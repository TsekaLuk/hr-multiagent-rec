"""Async LLM client with prompt-cache-aware message layout.

Design follows Claude Code's PREFIX-cached-then-DYNAMIC pattern (notes/04-Agent协调/06-Fork与提示词缓存优化.md):

    msgs = STATIC_PREFIX + DYNAMIC_TAIL

The STATIC_PREFIX is identical across a fan-out of N parallel calls
(same JD + rubric + few-shots), so SiliconFlow / DeepSeek / Anthropic
will report cache_read on every call after the first, drastically
cutting cost.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from hr_rec.agents.events import Usage

logger = logging.getLogger(__name__)


class AsyncLLMError(RuntimeError):
    pass


class AsyncLLM:
    """Async OpenAI-compatible chat client.

    Differences from the sync :class:`hr_rec.agents.llm.LLM`:

    * Uses ``httpx.AsyncClient`` so callers can ``asyncio.gather``.
    * Returns ``(text, Usage)`` so the orchestrator can stream usage
      events for cost/cache observability.
    * Accepts pre-split ``prefix`` and ``tail`` messages and concatenates
      them in a stable order, so providers can serve cache hits.
    """

    def __init__(
        self,
        *,
        model: str = "deepseek-ai/DeepSeek-V4-Flash",
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "siliconflow",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        # See hr_rec.agents.llm.LLM for the 180s rationale.
        timeout: float = 180.0,
        concurrency: int = 8,
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(concurrency)

        key = api_key or self._key_from_env(provider)
        base = base_url or self._base_url(provider)
        if not key:
            raise AsyncLLMError(
                f"No API key for provider '{provider}'. "
                f"Set the env var (e.g. SILICONFLOW_API_KEY)."
            )
        self._client = httpx.AsyncClient(
            base_url=base,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    # ---- core call ------------------------------------------------------

    async def chat(
        self,
        prefix: Sequence[dict[str, Any]],
        tail: Sequence[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> tuple[str, Usage]:
        """Returns (assistant_text, usage). Caches share `prefix`."""
        messages = list(prefix) + list(tail)
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        async with self.semaphore:
            last_err: Exception | None = None
            for attempt in range(retries + 1):
                t0 = time.time()
                try:
                    r = await self._client.post("/chat/completions", json=body)
                    r.raise_for_status()
                    data = r.json()
                    elapsed = time.time() - t0
                    text = data["choices"][0]["message"]["content"] or ""
                    usage = self._parse_usage(data.get("usage") or {}, elapsed)
                    return text, usage
                except httpx.HTTPError as e:
                    last_err = e
                    # Exponential backoff on transient failures.
                    await asyncio.sleep(min(2 ** attempt, 10))
                    continue
            raise AsyncLLMError(f"chat failed after {retries + 1} attempts: {last_err}")

    async def chat_json(
        self,
        prefix: Sequence[dict[str, Any]],
        tail: Sequence[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], Usage]:
        """Chat → JSON dict, with retry on parse failure (notes pattern: withhold + recover)."""
        cur_tail: list[dict[str, Any]] = list(tail)
        total_usage = Usage()
        for attempt in range(2):
            text, usage = await self.chat(
                prefix, cur_tail, temperature=temperature, max_tokens=max_tokens
            )
            total_usage = total_usage + usage
            obj = _safe_parse_json(text)
            if obj is not None:
                return obj, total_usage
            # Append assistant + corrective user message, retry once.
            cur_tail = [
                *cur_tail,
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Your previous response was not valid JSON. "
                                            "Reply with ONE valid JSON object only."},
            ]
        raise AsyncLLMError(f"Could not coerce LLM to JSON after 2 attempts: {text[:200]!r}")

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncLLM:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _key_from_env(provider: str) -> str | None:
        env_map = {
            "siliconflow": "SILICONFLOW_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "ollama": None,
        }
        var = env_map.get(provider)
        if var is None:
            return "ollama"
        return os.environ.get(var)

    @staticmethod
    def _base_url(provider: str) -> str:
        return {
            "siliconflow": "https://api.siliconflow.cn/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": "https://api.openai.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
            "ollama": "http://localhost:11434/v1",
        }[provider]

    @staticmethod
    def _parse_usage(d: dict[str, Any], elapsed: float) -> Usage:
        # OpenAI canonical + SiliconFlow's prompt_cache_hit_tokens extension
        details = d.get("prompt_tokens_details") or {}
        cache_read = (
            details.get("cached_tokens")
            or d.get("prompt_cache_hit_tokens")
            or 0
        )
        return Usage(
            input_tokens=int(d.get("prompt_tokens") or 0),
            output_tokens=int(d.get("completion_tokens") or 0),
            cache_read_tokens=int(cache_read),
            cache_write_tokens=0,
            calls=1,
            seconds=elapsed,
        )


def _safe_parse_json(s: str) -> dict[str, Any] | None:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Find first balanced { ... }
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
