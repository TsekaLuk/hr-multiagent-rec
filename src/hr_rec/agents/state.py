"""Pattern #4 from claude-code-notes/04-Agent协调/04-QueryEngine §4.1/4.2:
typed immutable ``CoordState`` + explicit ``transition.reason`` enum.

Why it matters here:

* Deterministic replay for paper figures (we can reconstruct each turn
  from the trace alone).
* A free per-turn telemetry stream — every transition records why we
  moved to the next state (max_tokens, ctx_overflow, fallback_model,
  candidate_done, …).
* A state-machine diagram we can paste directly into the thesis
  (auto-rendered by ``scripts/make_state_diagram.py``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class PhaseKind(str, Enum):
    """Top-level phases of the PJF coordinator loop."""

    INIT = "init"
    JOB_ANALYSIS = "job_analysis"
    CANDIDATE_FAN_OUT = "candidate_fan_out"
    COORDINATION = "coordination"
    EXPLANATION = "explanation"
    TERMINAL = "terminal"


class TransitionReason(str, Enum):
    """Why we moved to the next state.

    Each is *both* a telemetry tag *and* a state-machine edge label.
    Matches Claude Code's transition vocabulary from notes/04-QueryEngine §4.2.
    """

    START = "start"
    JOB_ANALYSED = "job_analysed"
    CANDIDATES_DONE = "candidates_done"
    COORD_DONE = "coord_done"
    EXPLAINED = "explained"

    MAX_TOKENS_RETRY = "max_tokens_retry"
    JSON_PARSE_RETRY = "json_parse_retry"
    CTX_OVERFLOW_SHRINK = "ctx_overflow_shrink"
    FALLBACK_MODEL = "fallback_model"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    USER_CANCEL = "user_cancel"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True)
class TransitionEvent:
    """A single edge in the state-machine trace."""

    from_phase: PhaseKind
    to_phase: PhaseKind
    reason: TransitionReason
    timestamp: float
    payload: tuple[tuple[str, Any], ...] = ()  # tuple-of-pairs so it stays hashable

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_phase.value,
            "to": self.to_phase.value,
            "reason": self.reason.value,
            "ts": self.timestamp,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class CoordState:
    """Immutable snapshot of the coordinator loop.

    Mutating operations return a *new* state via :meth:`advance`. This
    gives us free time-travel for debugging and a clean replay log for
    paper figures.
    """

    phase: PhaseKind
    turn: int = 0
    recovery_count: int = 0
    last_transition: TransitionReason | None = None
    history: tuple[TransitionEvent, ...] = field(default_factory=tuple)

    def advance(
        self,
        to_phase: PhaseKind,
        reason: TransitionReason,
        **payload: Any,
    ) -> CoordState:
        """Return a new state moved to ``to_phase`` with a recorded transition."""
        ev = TransitionEvent(
            from_phase=self.phase,
            to_phase=to_phase,
            reason=reason,
            timestamp=time.time(),
            payload=tuple(payload.items()),
        )
        return replace(
            self,
            phase=to_phase,
            turn=self.turn + 1,
            last_transition=reason,
            history=(*self.history, ev),
        )

    def with_recovery(self, reason: TransitionReason, **payload: Any) -> CoordState:
        """Stay in the current phase but record a recovery transition."""
        ev = TransitionEvent(
            from_phase=self.phase,
            to_phase=self.phase,
            reason=reason,
            timestamp=time.time(),
            payload=tuple(payload.items()),
        )
        return replace(
            self,
            recovery_count=self.recovery_count + 1,
            last_transition=reason,
            history=(*self.history, ev),
        )

    # ---- replay / analysis ----------------------------------------------

    def dump(self) -> list[dict[str, Any]]:
        """Serializable trace for paper appendix / case study tables."""
        return [e.to_dict() for e in self.history]

    def reasons(self) -> list[TransitionReason]:
        return [e.reason for e in self.history]

    @classmethod
    def initial(cls) -> CoordState:
        return cls(phase=PhaseKind.INIT)
