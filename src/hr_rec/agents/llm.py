"""LLM backend abstraction. OpenAI-compatible for SiliconFlow, DeepSeek,
OpenRouter, vLLM, Ollama; native Gemini support via Google SDK.

We deliberately keep this thin — agents only need ``chat`` and ``chat_json``.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """LLM call failed after retries."""


class LLM:
    """OpenAI-compatible chat client.

    Works against:
    * SiliconFlow  (``https://api.siliconflow.cn/v1``)
    * DeepSeek     (``https://api.deepseek.com/v1``)
    * OpenRouter   (``https://openrouter.ai/api/v1``)
    * Ollama       (``http://localhost:11434/v1``)
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
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        key = api_key or self._key_from_env(provider)
        base = base_url or self._base_url(provider)
        if not key:
            raise LLMError(
                f"No API key for provider '{provider}'. "
                f"Set the env var (e.g. SILICONFLOW_API_KEY) or pass api_key=."
            )
        self._client = OpenAI(api_key=key, base_url=base, timeout=timeout)

    @staticmethod
    def _key_from_env(provider: str) -> str | None:
        env_map = {
            "siliconflow": "SILICONFLOW_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "ollama": None,  # no key needed
        }
        var = env_map.get(provider)
        if var is None:
            return "ollama"  # placeholder
        return os.environ.get(var)

    @staticmethod
    def _base_url(provider: str) -> str:
        return {
            "siliconflow": "https://api.siliconflow.cn/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": "https://api.openai.com/v1",
            "ollama": "http://localhost:11434/v1",
        }[provider]

    # ---- core API ------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
        except Exception as e:
            raise LLMError(f"chat call failed: {e}") from e
        return resp.choices[0].message.content or ""

    def chat_json(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Chat and force-parse the response as JSON.

        We *append* a hard JSON-only instruction. If parsing fails on
        the first try, we retry once with an even stricter prompt.
        """
        sys_msg = {
            "role": "system",
            "content": (
                "You are a precise JSON-only API. Respond with one valid JSON object "
                "and nothing else — no commentary, no markdown fences."
            ),
        }
        msgs = [sys_msg, *list(messages)]
        for attempt in range(2):
            raw = self.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            obj = _safe_parse_json(raw)
            if obj is not None:
                return obj
            msgs.append({"role": "assistant", "content": raw})
            msgs.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. Reply with ONE valid "
                        "JSON object only. No markdown, no prose."
                    ),
                }
            )
        raise LLMError(f"Could not coerce LLM output to JSON after 2 attempts: {raw[:200]!r}")


def _safe_parse_json(s: str) -> dict[str, Any] | None:
    s = s.strip()
    # Strip ```json ... ``` fences if present
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # Try to extract first balanced { ... }
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
                        break
                    except json.JSONDecodeError:
                        return None
        else:
            return None
    return obj if isinstance(obj, dict) else None
