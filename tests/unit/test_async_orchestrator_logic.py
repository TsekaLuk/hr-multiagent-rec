"""Unit tests for the deterministic parts of the async orchestrator.

We test the synchronous helpers that don't need an LLM:
* `_cand_prefix` / `_cand_tail` produce the right cache-friendly layout
* `Coordinator._coordinate` adjusts scores deterministically
* `Usage` adds correctly + cost estimate
"""
from __future__ import annotations

import pytest

from hr_rec.agents.async_orchestrator import (
    AsyncOrchestrator,
    _cand_prefix,
    _cand_tail,
)
from hr_rec.agents.events import Event, EventType, Usage
from hr_rec.data.schemas import (
    EducationLevel,
    ExperienceLevel,
    Job,
    MatchEvidence,
    MatchScore,
    Resume,
    SalaryRange,
    Skill,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def job() -> Job:
    return Job(
        job_id="J-1",
        title="Backend Engineer",
        location="北京",
        salary=SalaryRange(min_cny=20_000, max_cny=30_000),
        required_education=EducationLevel.BACHELOR,
        required_experience=ExperienceLevel.Y1_3,
        required_skills=[Skill(name="Python"), Skill(name="Kafka")],
        preferred_skills=[Skill(name="Redis")],
        description="负责后端开发",
    )


@pytest.fixture
def resume() -> Resume:
    return Resume(
        resume_id="R-1",
        location="北京",
        expected_locations=["北京"],
        expected_salary=SalaryRange(min_cny=22_000, max_cny=28_000),
        experience_level=ExperienceLevel.Y3_5,
        skills=[Skill(name="Python"), Skill(name="Kafka")],
    )


# ---- prefix/tail cache layout -------------------------------------------


class TestCacheLayout:
    def test_prefix_is_stable_across_candidates(self, job: Job, resume: Resume) -> None:
        """Identical job + analysis MUST produce byte-identical prefix.

        This is the cache hit precondition.
        """
        ja = {
            "core_skills": ["Python", "Kafka"],
            "must_have_constraints": ["本科及以上"],
            "responsibility_summary": "后端开发",
        }
        p1 = _cand_prefix(job, ja)
        p2 = _cand_prefix(job, ja)
        assert p1 == p2

    def test_tail_differs_per_candidate(self) -> None:
        r1 = Resume(
            resume_id="R-A",
            location="北京",
            expected_locations=["北京"],
            experience_level=ExperienceLevel.Y1_3,
            skills=[Skill(name="Python")],
        )
        r2 = Resume(
            resume_id="R-B",
            location="上海",
            expected_locations=["上海"],
            experience_level=ExperienceLevel.Y3_5,
            skills=[Skill(name="Kafka")],
        )
        assert _cand_tail(r1) != _cand_tail(r2)

    def test_prefix_contains_no_candidate_specific_data(
        self, job: Job, resume: Resume
    ) -> None:
        """Prefix MUST NOT contain candidate-specific data — that breaks cache."""
        ja = {"core_skills": ["Python"]}
        prefix = _cand_prefix(job, ja)
        prefix_str = str(prefix)
        assert resume.resume_id not in prefix_str

    def test_prefix_message_roles_are_system_user_assistant(
        self, job: Job
    ) -> None:
        """For prompt caching, the role sequence must be deterministic."""
        ja = {"core_skills": ["X"]}
        prefix = _cand_prefix(job, ja)
        roles = [m["role"] for m in prefix]
        assert roles == ["system", "user", "assistant"]


# ---- Coordinator scoring -----------------------------------------------


class TestCoordinatorScoring:
    def _ms(self, rid: str, fused: float) -> MatchScore:
        return MatchScore(
            job_id="J", resume_id=rid,
            employer_score=fused, candidate_score=fused,
            fused_score=fused, semantic_similarity=0.5,
            rerank_score=None, evidence=None,
        )

    def test_high_fit_bumps_score(self) -> None:
        from hr_rec.agents.llm_async import AsyncLLM
        # We bypass network: only `_coordinate` is exercised.
        orch = AsyncOrchestrator.__new__(AsyncOrchestrator)
        out = orch._coordinate(
            [self._ms("A", 0.7)],
            {"A": {"overall_fit": "high", "risk_flags": []}},
        )
        assert out[0].fused_score == pytest.approx(0.78, abs=1e-6)

    def test_low_fit_penalises(self) -> None:
        orch = AsyncOrchestrator.__new__(AsyncOrchestrator)
        out = orch._coordinate(
            [self._ms("A", 0.7)],
            {"A": {"overall_fit": "low", "risk_flags": []}},
        )
        assert out[0].fused_score == pytest.approx(0.62, abs=1e-6)

    def test_risk_flags_subtract_capped_at_10pct(self) -> None:
        orch = AsyncOrchestrator.__new__(AsyncOrchestrator)
        # 5 flags × 0.03 = 0.15, but capped at 0.10
        out = orch._coordinate(
            [self._ms("A", 0.5)],
            {"A": {"overall_fit": "medium",
                   "risk_flags": ["a", "b", "c", "d", "e"]}},
        )
        assert out[0].fused_score == pytest.approx(0.40, abs=1e-6)

    def test_results_sorted_descending(self) -> None:
        orch = AsyncOrchestrator.__new__(AsyncOrchestrator)
        out = orch._coordinate(
            [self._ms("A", 0.5), self._ms("B", 0.9), self._ms("C", 0.7)],
            {},  # no analyses → no adjustment
        )
        ids = [m.resume_id for m in out]
        assert ids == ["B", "C", "A"]


# ---- Usage / cost --------------------------------------------------------


class TestUsage:
    def test_addition_is_componentwise(self) -> None:
        a = Usage(input_tokens=100, output_tokens=50, cache_read_tokens=20,
                  calls=1, seconds=1.5)
        b = Usage(input_tokens=200, output_tokens=80, cache_read_tokens=180,
                  calls=2, seconds=2.5)
        s = a + b
        assert s.input_tokens == 300
        assert s.output_tokens == 130
        assert s.cache_read_tokens == 200
        assert s.calls == 3
        assert s.seconds == 4.0

    def test_cost_estimate_with_cache(self) -> None:
        # $0.30 input, $2.50 output, $0.10 cache_read per M tokens
        u = Usage(input_tokens=10_000, output_tokens=2_000, cache_read_tokens=50_000)
        cost = u.cost_estimate(
            input_price_per_m=0.30,
            output_price_per_m=2.50,
            cache_read_price_per_m=0.10,
        )
        # 10000*0.30/1e6 + 2000*2.50/1e6 + 50000*0.10/1e6
        # = 0.003 + 0.005 + 0.005 = 0.013
        assert cost == pytest.approx(0.013, abs=1e-9)

    def test_zero_cost_for_free_provider(self) -> None:
        u = Usage(input_tokens=10_000, output_tokens=5_000)
        assert u.cost_estimate() == 0.0


# ---- Event dataclass ----------------------------------------------------


class TestEvent:
    def test_event_default_payload(self) -> None:
        e = Event(EventType.AGENT_START, agent="job_analyst")
        assert e.payload == {}
        assert e.usage is None

    def test_event_with_usage(self) -> None:
        u = Usage(calls=1)
        e = Event(EventType.USAGE, agent="x", usage=u)
        assert e.usage is u
