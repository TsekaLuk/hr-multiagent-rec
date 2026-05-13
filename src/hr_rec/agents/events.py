"""Streaming events emitted by the async orchestrator.

Patterned after Claude Code's coordinator events (notes/04/05) so the
caller can render progress, accumulate usage, and cancel mid-flight.

Usage accounting follows Claude's dual-token model (notes/06-cost-token §3):
* **window_tokens** — what occupies the model's context window. Drives
  context-compaction decisions; counts cached tokens too because they
  still consume the window.
* **billed_tokens** — what the provider actually charges for. Excludes
  cached reads (most providers price them at ~1/10 of input tokens, or
  free, so we report cost separately).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    CANDIDATE_PROFILED = "candidate_profiled"
    USAGE = "usage"
    TRANSITION = "transition"     # max_tokens_retry / json_parse_retry / fallback_model
    ERROR = "error"
    CIRCUIT_BREAKER = "circuit_breaker"  # fail-fast triggered
    FINAL = "final"


@dataclass(frozen=True)
class Pricing:
    """Per-million-token pricing for a provider/model."""

    input_per_m: float = 0.0          # USD per 1M input tokens
    output_per_m: float = 0.0         # USD per 1M output tokens
    cache_read_per_m: float = 0.0     # USD per 1M cache-read tokens
    cache_write_per_m: float = 0.0    # USD per 1M cache-write tokens


# Known SiliconFlow / DeepSeek / OpenRouter prices (May 2026, USD/M tokens).
KNOWN_PRICING: dict[str, Pricing] = {
    "Qwen/Qwen3-8B": Pricing(),                      # free
    "Qwen/Qwen3.5-4B": Pricing(),                    # free
    "Qwen/Qwen3.5-9B": Pricing(),                    # free at time of writing
    "deepseek-ai/DeepSeek-V4-Flash": Pricing(
        input_per_m=0.145, output_per_m=1.74
    ),
    "deepseek-ai/DeepSeek-V3.2": Pricing(input_per_m=0.27, output_per_m=1.1),
    "google/gemini-2.5-flash": Pricing(input_per_m=0.30, output_per_m=2.50),
    "google/gemini-3.1-flash-lite": Pricing(input_per_m=0.25, output_per_m=1.50),
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0
    seconds: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            calls=self.calls + other.calls,
            seconds=self.seconds + other.seconds,
        )

    # ---- dual-token accounting ------------------------------------------

    def window_tokens(self) -> int:
        """Tokens that occupied the model's context window.

        Includes cache_read because cached tokens still take up the
        window even if priced cheaply. Used to drive context-compaction
        decisions.
        """
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens + self.output_tokens

    def billed_tokens(self) -> int:
        """Tokens that are billed at the *primary* input/output rate.

        Cache reads (and writes) get separate pricing; see
        :meth:`cost_usd`.
        """
        return self.input_tokens + self.output_tokens

    def cache_hit_rate(self) -> float:
        """Fraction of input tokens served from cache (0.0–1.0)."""
        total_input = self.input_tokens + self.cache_read_tokens
        if total_input == 0:
            return 0.0
        return self.cache_read_tokens / total_input

    # ---- cost -----------------------------------------------------------

    def cost_usd(self, pricing: Pricing | None) -> float:
        """USD spend at the given pricing."""
        if pricing is None:
            return 0.0
        return (
            self.input_tokens * pricing.input_per_m
            + self.output_tokens * pricing.output_per_m
            + self.cache_read_tokens * pricing.cache_read_per_m
            + self.cache_write_tokens * pricing.cache_write_per_m
        ) / 1_000_000.0

    def cost_estimate(
        self,
        *,
        input_price_per_m: float = 0.0,
        output_price_per_m: float = 0.0,
        cache_read_price_per_m: float = 0.0,
        cache_write_price_per_m: float = 0.0,
    ) -> float:
        """Legacy keyword-argument API; prefer :meth:`cost_usd` with a :class:`Pricing`."""
        return self.cost_usd(
            Pricing(
                input_per_m=input_price_per_m,
                output_per_m=output_price_per_m,
                cache_read_per_m=cache_read_price_per_m,
                cache_write_per_m=cache_write_price_per_m,
            )
        )


def lookup_pricing(model: str) -> Pricing | None:
    """Return known pricing for `model` (case-insensitive prefix match)."""
    if not model:
        return None
    m = model.lower()
    for k, v in KNOWN_PRICING.items():
        if m == k.lower() or m.startswith(k.lower()):
            return v
    return None


@dataclass
class Event:
    type: EventType
    agent: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    usage: Usage | None = None
