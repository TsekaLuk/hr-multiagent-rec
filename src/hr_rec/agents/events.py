"""Streaming events emitted by the async orchestrator.

Patterned after Claude Code's coordinator events (notes/04/05) so the
caller can render progress, accumulate usage, and cancel mid-flight.
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
    FINAL = "final"


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

    def cost_estimate(
        self,
        *,
        input_price_per_m: float = 0.0,
        output_price_per_m: float = 0.0,
        cache_read_price_per_m: float = 0.0,
    ) -> float:
        """USD-equivalent cost estimate.

        SiliconFlow Qwen3-8B is free (set all prices to 0.0); for
        Gemini/DeepSeek pass the documented per-million-token prices.
        """
        return (
            self.input_tokens * input_price_per_m
            + self.output_tokens * output_price_per_m
            + self.cache_read_tokens * cache_read_price_per_m
        ) / 1_000_000.0


@dataclass
class Event:
    type: EventType
    agent: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    usage: Usage | None = None
