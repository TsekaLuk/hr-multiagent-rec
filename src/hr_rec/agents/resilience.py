"""Typed retry + circuit breaker for LLM calls.

Ports two patterns from claude-code-notes:
* **Typed retry router** — classifies exceptions into recoverable vs fatal
  buckets; only burns attempts on recoverable ones; emits explicit
  telemetry events.
* **Circuit breaker** — after N consecutive failures of the same kind,
  trip the breaker and stop wasting calls.
"""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryClass(str, Enum):
    """Exception classes drawn from notes/06-services-api §3.6/3.8."""

    USER_ABORT = "user_abort"          # bail immediately, no retry
    CTX_OVERFLOW = "ctx_overflow"      # shrink prompt and retry (doesn't burn attempt)
    RATE_LIMIT = "rate_limit"          # backoff long, retry
    OVERLOAD = "overload"              # 503 / model overloaded; backoff
    NET = "net"                        # transient TCP / timeout / DNS
    JSON = "json"                      # response shape wrong
    FATAL = "fatal"                    # 400, schema, anything we shouldn't retry


def classify(e: BaseException) -> RetryClass:
    if isinstance(e, asyncio.CancelledError | KeyboardInterrupt):
        return RetryClass.USER_ABORT
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 429:
            return RetryClass.RATE_LIMIT
        if code in (502, 503, 504):
            return RetryClass.OVERLOAD
        if code == 413:
            return RetryClass.CTX_OVERFLOW
        if 500 <= code < 600:
            return RetryClass.OVERLOAD
        return RetryClass.FATAL
    if isinstance(e, httpx.TimeoutException | httpx.ConnectError | httpx.ReadError | httpx.WriteError):
        return RetryClass.NET
    if isinstance(e, httpx.HTTPError):
        return RetryClass.NET
    msg = (str(e) or "").lower()
    if "context_length" in msg or "context length" in msg or "maximum context" in msg:
        return RetryClass.CTX_OVERFLOW
    if "rate" in msg and "limit" in msg:
        return RetryClass.RATE_LIMIT
    if "json" in msg:
        return RetryClass.JSON
    return RetryClass.FATAL


@dataclass
class RetryReport:
    """Per-call observability record."""

    attempts: int = 0
    classes: list[RetryClass] = None  # type: ignore[assignment]
    breaker_tripped: bool = False

    def __post_init__(self) -> None:
        if self.classes is None:
            self.classes = []


class CircuitBreaker:
    """Trips after N consecutive failures *of the same class*.

    Once tripped, subsequent calls fail-fast for ``cool_down_seconds``
    before the breaker tries again.
    """

    def __init__(self, *, max_consecutive: int = 3, cool_down_seconds: float = 30.0) -> None:
        self.max_consecutive = max_consecutive
        self.cool_down_seconds = cool_down_seconds
        self.fails = 0
        self.last_class: RetryClass | None = None
        self.tripped_until = 0.0

    @property
    def is_tripped(self) -> bool:
        return self.fails >= self.max_consecutive

    def record(self, ok: bool, cls: RetryClass | None = None) -> None:
        if ok:
            self.fails = 0
            self.last_class = None
            self.tripped_until = 0.0
            return
        if cls is None or cls != self.last_class:
            self.fails = 1
        else:
            self.fails += 1
        self.last_class = cls
        if self.is_tripped:
            import time
            self.tripped_until = time.time() + self.cool_down_seconds

    def maybe_open(self) -> bool:
        """Returns True if the call should be aborted now (fail-fast)."""
        import time
        if not self.is_tripped:
            return False
        if time.time() >= self.tripped_until:
            # half-open: allow one trial call
            self.fails = self.max_consecutive - 1
            return False
        return True


async def with_typed_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 32.0,
    breaker: CircuitBreaker | None = None,
    background: bool = False,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    on_ctx_overflow: Callable[[], None] | None = None,
) -> tuple[T, RetryReport]:
    """Run `coro_factory()` with classified retry and optional breaker.

    Returns the awaited value plus a :class:`RetryReport`. Raises only
    on FATAL, USER_ABORT, or after exhausting `max_attempts`.

    When ``background=True``, RATE_LIMIT errors are *fatal* — we'd rather
    fail a background eval than block the user-facing path.
    """
    report = RetryReport()
    for attempt in range(max_attempts):
        if breaker and breaker.maybe_open():
            report.breaker_tripped = True
            if on_event:
                on_event({"event": "circuit_breaker_open", "class": "tripped"})
            raise RuntimeError("circuit breaker open")
        try:
            value = await coro_factory()
            if breaker:
                breaker.record(ok=True)
            report.attempts = attempt + 1
            return value, report
        except BaseException as e:
            cls = classify(e)
            report.classes.append(cls)
            if cls == RetryClass.USER_ABORT or cls == RetryClass.FATAL:
                if breaker:
                    breaker.record(ok=False, cls=cls)
                raise
            if cls == RetryClass.RATE_LIMIT and background:
                if breaker:
                    breaker.record(ok=False, cls=cls)
                raise
            if cls == RetryClass.CTX_OVERFLOW:
                if on_ctx_overflow:
                    on_ctx_overflow()
                if on_event:
                    on_event({"event": "retry", "class": cls.value, "attempt": attempt})
                continue   # don't count as attempt; caller shrunk the prompt
            if attempt == max_attempts - 1:
                if breaker:
                    breaker.record(ok=False, cls=cls)
                raise
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.random()
            if breaker:
                breaker.record(ok=False, cls=cls)
            if on_event:
                on_event({
                    "event": "retry",
                    "class": cls.value,
                    "attempt": attempt,
                    "delay": round(delay, 2),
                })
            logger.warning("retry attempt=%d class=%s delay=%.1fs", attempt, cls.value, delay)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable: loop exited without return")
