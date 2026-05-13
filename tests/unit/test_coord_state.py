"""Tests for the typed immutable CoordState pattern (no mocks, pure logic)."""
from __future__ import annotations

import dataclasses

import pytest

from hr_rec.agents.state import (
    CoordState,
    PhaseKind,
    TransitionEvent,
    TransitionReason,
)

pytestmark = pytest.mark.unit


class TestCoordState:
    def test_initial_is_init_phase(self) -> None:
        s = CoordState.initial()
        assert s.phase == PhaseKind.INIT
        assert s.turn == 0
        assert s.recovery_count == 0
        assert s.history == ()
        assert s.last_transition is None

    def test_advance_returns_new_state(self) -> None:
        s = CoordState.initial()
        s2 = s.advance(PhaseKind.JOB_ANALYSIS, TransitionReason.START)
        # Original unchanged
        assert s.phase == PhaseKind.INIT
        assert s.turn == 0
        # New state moved forward
        assert s2.phase == PhaseKind.JOB_ANALYSIS
        assert s2.turn == 1
        assert s2.last_transition == TransitionReason.START
        assert len(s2.history) == 1
        assert s2.history[0].from_phase == PhaseKind.INIT
        assert s2.history[0].to_phase == PhaseKind.JOB_ANALYSIS
        assert s2.history[0].reason == TransitionReason.START

    def test_advance_chains_immutably(self) -> None:
        s = (
            CoordState.initial()
            .advance(PhaseKind.JOB_ANALYSIS, TransitionReason.START)
            .advance(PhaseKind.CANDIDATE_FAN_OUT, TransitionReason.JOB_ANALYSED)
            .advance(PhaseKind.COORDINATION, TransitionReason.CANDIDATES_DONE)
            .advance(PhaseKind.EXPLANATION, TransitionReason.COORD_DONE)
            .advance(PhaseKind.TERMINAL, TransitionReason.EXPLAINED)
        )
        assert s.phase == PhaseKind.TERMINAL
        assert s.turn == 5
        assert len(s.history) == 5
        assert s.reasons() == [
            TransitionReason.START,
            TransitionReason.JOB_ANALYSED,
            TransitionReason.CANDIDATES_DONE,
            TransitionReason.COORD_DONE,
            TransitionReason.EXPLAINED,
        ]

    def test_advance_with_payload_kept_immutable(self) -> None:
        s = CoordState.initial().advance(
            PhaseKind.JOB_ANALYSIS,
            TransitionReason.START,
            job_id="J-42",
            n_candidates=15,
        )
        # payload is tuple-of-pairs (hashable, immutable)
        assert s.history[0].payload == (("job_id", "J-42"), ("n_candidates", 15))
        assert s.history[0].to_dict()["payload"] == {"job_id": "J-42", "n_candidates": 15}

    def test_with_recovery_stays_in_phase_but_counts(self) -> None:
        s = (
            CoordState.initial()
            .advance(PhaseKind.JOB_ANALYSIS, TransitionReason.START)
            .with_recovery(TransitionReason.JSON_PARSE_RETRY)
            .with_recovery(TransitionReason.MAX_TOKENS_RETRY)
        )
        assert s.phase == PhaseKind.JOB_ANALYSIS  # unchanged
        assert s.turn == 1                         # advance() incremented once
        assert s.recovery_count == 2
        assert s.last_transition == TransitionReason.MAX_TOKENS_RETRY
        assert len(s.history) == 3

    def test_dump_yields_serialisable_trace(self) -> None:
        s = CoordState.initial().advance(
            PhaseKind.JOB_ANALYSIS,
            TransitionReason.START,
            job_id="J-1",
        )
        dump = s.dump()
        assert isinstance(dump, list)
        assert isinstance(dump[0], dict)
        assert dump[0]["from"] == "init"
        assert dump[0]["to"] == "job_analysis"
        assert dump[0]["reason"] == "start"
        assert dump[0]["payload"] == {"job_id": "J-1"}

    def test_state_is_frozen(self) -> None:
        s = CoordState.initial()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.phase = PhaseKind.TERMINAL  # type: ignore[misc]

    def test_history_is_immutable_tuple(self) -> None:
        s = CoordState.initial().advance(PhaseKind.JOB_ANALYSIS, TransitionReason.START)
        assert isinstance(s.history, tuple)


class TestTransitionEvent:
    def test_event_is_frozen(self) -> None:
        ev = TransitionEvent(
            from_phase=PhaseKind.INIT,
            to_phase=PhaseKind.TERMINAL,
            reason=TransitionReason.START,
            timestamp=0.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.reason = TransitionReason.EXPLAINED  # type: ignore[misc]

    def test_to_dict_round_trips_keys(self) -> None:
        ev = TransitionEvent(
            from_phase=PhaseKind.JOB_ANALYSIS,
            to_phase=PhaseKind.CANDIDATE_FAN_OUT,
            reason=TransitionReason.JOB_ANALYSED,
            timestamp=1234.5,
            payload=(("a", 1), ("b", "x")),
        )
        d = ev.to_dict()
        assert d == {
            "from": "job_analysis",
            "to": "candidate_fan_out",
            "reason": "job_analysed",
            "ts": 1234.5,
            "payload": {"a": 1, "b": "x"},
        }


class TestEnumValues:
    """Stable enum string values matter for log parsing + paper figures."""

    def test_phase_kind_string_values(self) -> None:
        # Verifying the rendered diagram won't shift if we add a phase later.
        expected = {
            "init", "job_analysis", "candidate_fan_out",
            "coordination", "explanation", "terminal",
        }
        assert {p.value for p in PhaseKind} == expected

    def test_transition_reason_includes_recovery_vocab(self) -> None:
        values = {r.value for r in TransitionReason}
        for v in (
            "max_tokens_retry", "json_parse_retry", "ctx_overflow_shrink",
            "fallback_model", "circuit_breaker_open", "user_cancel",
        ):
            assert v in values
