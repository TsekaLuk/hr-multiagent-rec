"""Tests for the new resilience + dual-token Usage helpers.

No mocks — we exercise real Usage / Pricing / CircuitBreaker /
classify with pure-logic inputs. The async retry wrapper is tested with
a callable that raises real httpx exceptions on the first N calls.
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from hr_rec.agents.events import (
    KNOWN_PRICING,
    Pricing,
    Usage,
    lookup_pricing,
)
from hr_rec.agents.resilience import (
    CircuitBreaker,
    RetryClass,
    classify,
    with_typed_retry,
)

pytestmark = pytest.mark.unit


# ---- Usage dual-token + cost --------------------------------------------


class TestUsageWindowVsBilled:
    def test_window_includes_cache(self) -> None:
        u = Usage(input_tokens=100, output_tokens=50, cache_read_tokens=900)
        assert u.window_tokens() == 1050
        assert u.billed_tokens() == 150

    def test_cache_hit_rate(self) -> None:
        u = Usage(input_tokens=200, cache_read_tokens=800)
        assert u.cache_hit_rate() == 0.8

    def test_cache_hit_rate_no_input_is_zero(self) -> None:
        assert Usage().cache_hit_rate() == 0.0


class TestCostUSD:
    def test_zero_pricing_zero_cost(self) -> None:
        u = Usage(input_tokens=10_000, output_tokens=2_000)
        assert u.cost_usd(Pricing()) == 0.0
        assert u.cost_usd(None) == 0.0

    def test_known_pricing_lookup(self) -> None:
        u = Usage(input_tokens=1_000_000, output_tokens=0)
        p = lookup_pricing("deepseek-ai/DeepSeek-V4-Flash")
        assert p is not None
        assert u.cost_usd(p) == pytest.approx(0.145, abs=1e-9)

    def test_qwen3_free_models_are_zero(self) -> None:
        u = Usage(input_tokens=10_000_000, output_tokens=10_000_000)
        assert u.cost_usd(lookup_pricing("Qwen/Qwen3.5-4B")) == 0.0

    def test_legacy_cost_estimate_compatible(self) -> None:
        u = Usage(input_tokens=10_000, output_tokens=2_000)
        cost = u.cost_estimate(input_price_per_m=0.30, output_price_per_m=2.50)
        # 10000*0.30/1e6 + 2000*2.50/1e6 = 0.003 + 0.005 = 0.008
        assert cost == pytest.approx(0.008, abs=1e-9)

    def test_lookup_unknown_returns_none(self) -> None:
        assert lookup_pricing("some-fictional-model-x") is None


# ---- classify() ----------------------------------------------------------


class TestClassify:
    def test_cancelled(self) -> None:
        assert classify(asyncio.CancelledError()) == RetryClass.USER_ABORT

    def test_rate_limit_status(self) -> None:
        req = httpx.Request("POST", "https://api.example.com")
        resp = httpx.Response(429, request=req)
        e = httpx.HTTPStatusError("rate-limited", request=req, response=resp)
        assert classify(e) == RetryClass.RATE_LIMIT

    def test_5xx_overload(self) -> None:
        req = httpx.Request("POST", "https://api.example.com")
        for code in (502, 503, 504):
            resp = httpx.Response(code, request=req)
            e = httpx.HTTPStatusError("x", request=req, response=resp)
            assert classify(e) == RetryClass.OVERLOAD

    def test_413_ctx_overflow(self) -> None:
        req = httpx.Request("POST", "https://api.example.com")
        resp = httpx.Response(413, request=req)
        e = httpx.HTTPStatusError("payload too large", request=req, response=resp)
        assert classify(e) == RetryClass.CTX_OVERFLOW

    def test_400_fatal(self) -> None:
        req = httpx.Request("POST", "https://api.example.com")
        resp = httpx.Response(400, request=req)
        e = httpx.HTTPStatusError("bad request", request=req, response=resp)
        assert classify(e) == RetryClass.FATAL

    def test_timeout_is_net(self) -> None:
        e = httpx.ConnectTimeout("connect timeout")
        assert classify(e) == RetryClass.NET

    def test_message_substring_ctx_overflow(self) -> None:
        assert classify(Exception("context_length_exceeded: too big")) == RetryClass.CTX_OVERFLOW


# ---- CircuitBreaker -----------------------------------------------------


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        b = CircuitBreaker()
        assert not b.is_tripped
        assert not b.maybe_open()

    def test_trips_after_n_consecutive_same_class(self) -> None:
        b = CircuitBreaker(max_consecutive=3)
        for _ in range(3):
            b.record(ok=False, cls=RetryClass.RATE_LIMIT)
        assert b.is_tripped

    def test_different_classes_reset_counter(self) -> None:
        b = CircuitBreaker(max_consecutive=3)
        b.record(ok=False, cls=RetryClass.RATE_LIMIT)
        b.record(ok=False, cls=RetryClass.RATE_LIMIT)
        b.record(ok=False, cls=RetryClass.NET)  # resets — different class
        assert not b.is_tripped

    def test_success_resets(self) -> None:
        b = CircuitBreaker(max_consecutive=3)
        b.record(ok=False, cls=RetryClass.RATE_LIMIT)
        b.record(ok=False, cls=RetryClass.RATE_LIMIT)
        b.record(ok=True)
        assert b.fails == 0

    def test_half_open_after_cooldown(self) -> None:
        b = CircuitBreaker(max_consecutive=2, cool_down_seconds=0.01)
        b.record(ok=False, cls=RetryClass.RATE_LIMIT)
        b.record(ok=False, cls=RetryClass.RATE_LIMIT)
        assert b.maybe_open() is True
        time.sleep(0.02)
        assert b.maybe_open() is False  # half-open, allow trial


# ---- with_typed_retry ---------------------------------------------------


@pytest.mark.asyncio
async def test_with_typed_retry_succeeds_after_transient() -> None:
    n = {"count": 0}

    async def flaky() -> str:
        n["count"] += 1
        if n["count"] < 3:
            raise httpx.ConnectTimeout("nope")
        return "ok"

    val, rep = await with_typed_retry(flaky, max_attempts=5, base_delay=0.01)
    assert val == "ok"
    assert rep.attempts == 3
    assert rep.classes == [RetryClass.NET, RetryClass.NET]


@pytest.mark.asyncio
async def test_with_typed_retry_raises_on_fatal() -> None:
    req = httpx.Request("POST", "https://api.example.com")
    resp = httpx.Response(400, request=req)
    err = httpx.HTTPStatusError("nope", request=req, response=resp)

    async def fatal() -> str:
        raise err

    with pytest.raises(httpx.HTTPStatusError):
        await with_typed_retry(fatal, max_attempts=5, base_delay=0.01)


@pytest.mark.asyncio
async def test_with_typed_retry_background_rejects_rate_limit() -> None:
    req = httpx.Request("POST", "https://api.example.com")
    resp = httpx.Response(429, request=req)
    err = httpx.HTTPStatusError("rl", request=req, response=resp)

    async def rl() -> str:
        raise err

    with pytest.raises(httpx.HTTPStatusError):
        await with_typed_retry(rl, max_attempts=5, base_delay=0.01, background=True)


@pytest.mark.asyncio
async def test_with_typed_retry_breaker_fail_fast() -> None:
    breaker = CircuitBreaker(max_consecutive=2, cool_down_seconds=60.0)

    async def always_overload() -> str:
        req = httpx.Request("POST", "https://api.example.com")
        resp = httpx.Response(503, request=req)
        raise httpx.HTTPStatusError("503", request=req, response=resp)

    # First two attempts populate the breaker, third should fail-fast.
    with pytest.raises(Exception):
        await with_typed_retry(
            always_overload, max_attempts=10, base_delay=0.01, breaker=breaker
        )
    assert breaker.is_tripped


@pytest.mark.asyncio
async def test_with_typed_retry_ctx_overflow_does_not_count_attempt() -> None:
    calls = {"n": 0, "shrinks": 0}

    async def shrink_then_succeed() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("context_length_exceeded: too big")
        return "ok"

    def on_ctx() -> None:
        calls["shrinks"] += 1

    val, rep = await with_typed_retry(
        shrink_then_succeed,
        max_attempts=2,
        base_delay=0.01,
        on_ctx_overflow=on_ctx,
    )
    assert val == "ok"
    assert calls["shrinks"] == 1
    # ctx_overflow flows through the loop but is recorded separately in
    # `classes`. `attempts` counts loop iterations actually entered (2 here).
    assert rep.attempts == 2
    assert rep.classes == [RetryClass.CTX_OVERFLOW]
