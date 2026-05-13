"""End-to-end agent tests against a real LLM API.

Skip cleanly when no API key is present — never mock.
"""
from __future__ import annotations

import os
from datetime import date

import pytest

from hr_rec.agents.base import AgentContext
from hr_rec.agents.candidate_analyst import CandidateAnalystAgent
from hr_rec.agents.coordinator import CoordinatorAgent
from hr_rec.agents.job_analyst import JobAnalystAgent
from hr_rec.agents.llm import LLM, LLMError
from hr_rec.data.schemas import (
    EducationEntry,
    EducationLevel,
    ExperienceLevel,
    Job,
    MatchEvidence,
    MatchScore,
    Resume,
    SalaryRange,
    Skill,
)

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


@pytest.fixture(scope="session")
def llm() -> LLM:
    if not os.environ.get("SILICONFLOW_API_KEY"):
        pytest.skip("SILICONFLOW_API_KEY not set — skipping real-LLM tests")
    try:
        return LLM(model="Qwen/Qwen3-8B", provider="siliconflow", max_tokens=512)
    except LLMError as e:
        pytest.skip(f"could not init LLM: {e}")


@pytest.fixture
def sample_job() -> Job:
    return Job(
        job_id="J-DEMO",
        title="高级后端开发工程师",
        company="字节跳动",
        location="北京",
        salary=SalaryRange(min_cny=25_000, max_cny=40_000),
        required_education=EducationLevel.BACHELOR,
        required_experience=ExperienceLevel.Y3_5,
        required_skills=[Skill(name="Java"), Skill(name="Kafka"), Skill(name="MySQL")],
        preferred_skills=[Skill(name="Redis"), Skill(name="Kubernetes")],
        description=(
            "负责字节跳动核心电商系统的后端开发，要求精通Java/Spring生态，"
            "有大规模分布式系统经验，熟悉Kafka和MySQL优化。"
        ),
    )


@pytest.fixture
def sample_resume() -> Resume:
    return Resume(
        resume_id="R-DEMO",
        summary="4年Java后端经验，熟悉分布式系统",
        location="北京",
        expected_locations=["北京", "上海"],
        expected_salary=SalaryRange(min_cny=28_000, max_cny=42_000),
        education=[
            EducationEntry(
                school="南京大学",
                major="计算机科学与技术",
                level=EducationLevel.BACHELOR,
                start=date(2019, 9, 1),
                end=date(2023, 6, 30),
            )
        ],
        experience_level=ExperienceLevel.Y3_5,
        skills=[
            Skill(name="Java"),
            Skill(name="Spring Boot"),
            Skill(name="Kafka"),
            Skill(name="MySQL"),
            Skill(name="Redis"),
        ],
        raw_text="4年Java后端经验，曾在阿里、美团工作，主导Kafka消息中间件改造。",
    )


# ---------- JobAnalystAgent ------------------------------------------------


def test_job_analyst_returns_structured_json(llm: LLM, sample_job: Job) -> None:
    agent = JobAnalystAgent(llm)
    ctx = AgentContext(job=sample_job, candidate_resumes=[])
    out = agent.run(ctx)
    assert out.job_analysis  # not empty
    core = out.job_analysis.get("core_skills") or []
    assert isinstance(core, list)
    # At least one of the explicit required skills should appear
    core_lc = " ".join(str(s) for s in core).lower()
    assert any(s in core_lc for s in ("java", "kafka", "mysql"))


# ---------- CandidateAnalystAgent -----------------------------------------


def test_candidate_analyst_profiles_each_resume(
    llm: LLM, sample_job: Job, sample_resume: Resume
) -> None:
    job_agent = JobAnalystAgent(llm)
    cand_agent = CandidateAnalystAgent(llm)
    ctx = AgentContext(job=sample_job, candidate_resumes=[sample_resume])
    ctx = job_agent.run(ctx)
    ctx = cand_agent.run(ctx)
    profile = ctx.candidate_analyses.get(sample_resume.resume_id)
    assert profile is not None
    assert profile.get("overall_fit") in {"high", "medium", "low"}


# ---------- CoordinatorAgent (no LLM round-trip) --------------------------


def test_coordinator_adjusts_for_high_fit(sample_job: Job, sample_resume: Resume) -> None:
    """Coordinator is deterministic — no LLM needed."""
    evidence = MatchEvidence(
        matched_skills=["Java", "Kafka"],
        missing_skills=[],
        salary_compatible=True,
        location_compatible=True,
        education_satisfied=True,
        experience_satisfied=True,
    )
    ms = MatchScore(
        job_id=sample_job.job_id,
        resume_id=sample_resume.resume_id,
        employer_score=0.8,
        candidate_score=0.8,
        fused_score=0.6,
        semantic_similarity=0.85,
        evidence=evidence,
    )
    ctx = AgentContext(
        job=sample_job,
        candidate_resumes=[sample_resume],
        pre_ranked=[ms],
        candidate_analyses={sample_resume.resume_id: {"overall_fit": "high", "risk_flags": []}},
    )

    # Real LLM not needed for coordinator; pass None-safe construction
    class _DummyLLM:  # NOT a mock — a real class that simply isn't called by Coordinator
        pass

    coord = CoordinatorAgent.__new__(CoordinatorAgent)
    coord.llm = _DummyLLM()  # type: ignore[assignment]
    coord.name = "coordinator"
    out = coord.run(ctx)
    assert out.final_ranking[0].fused_score == pytest.approx(0.68, abs=1e-6)


def test_coordinator_penalises_risk_flags(sample_job: Job, sample_resume: Resume) -> None:
    ms = MatchScore(
        job_id=sample_job.job_id,
        resume_id=sample_resume.resume_id,
        employer_score=0.7,
        candidate_score=0.7,
        fused_score=0.7,
        semantic_similarity=0.8,
        evidence=None,
    )
    ctx = AgentContext(
        job=sample_job,
        candidate_resumes=[sample_resume],
        pre_ranked=[ms],
        candidate_analyses={
            sample_resume.resume_id: {
                "overall_fit": "medium",
                "risk_flags": ["薪资倒挂", "地点不符", "经验不足"],
            }
        },
    )
    coord = CoordinatorAgent.__new__(CoordinatorAgent)
    coord.name = "coordinator"
    coord.llm = object()  # unused
    out = coord.run(ctx)
    # 3 flags × 0.03 each = 0.09, no fit bonus → fused = 0.7 - 0.09 = 0.61
    assert out.final_ranking[0].fused_score == pytest.approx(0.61, abs=1e-6)
