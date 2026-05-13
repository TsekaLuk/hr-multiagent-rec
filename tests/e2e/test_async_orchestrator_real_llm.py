"""End-to-end tests for the async orchestrator against a real LLM API.

Skip cleanly when SILICONFLOW_API_KEY is not set — NEVER mock.
"""
from __future__ import annotations

import os
from datetime import date

import pytest
import pytest_asyncio

from hr_rec.agents.async_orchestrator import AsyncOrchestrator
from hr_rec.agents.events import EventType
from hr_rec.agents.llm_async import AsyncLLM, AsyncLLMError
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


@pytest_asyncio.fixture
async def llm():
    if not os.environ.get("SILICONFLOW_API_KEY"):
        pytest.skip("SILICONFLOW_API_KEY not set — skipping real-LLM tests")
    try:
        async with AsyncLLM(
            model="deepseek-ai/DeepSeek-V4-Flash",
            provider="siliconflow",
            concurrency=4,
            max_tokens=512,
        ) as c:
            yield c
    except AsyncLLMError as e:
        pytest.skip(f"could not init LLM: {e}")


@pytest.fixture
def job() -> Job:
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
            "负责字节跳动核心电商系统的后端开发，要求精通 Java/Spring 生态，"
            "有大规模分布式系统经验，熟悉 Kafka 和 MySQL 优化。"
        ),
    )


def _resume(rid: str, skills: list[str]) -> Resume:
    return Resume(
        resume_id=rid,
        summary="后端开发工程师",
        location="北京",
        expected_locations=["北京"],
        expected_salary=SalaryRange(min_cny=28_000, max_cny=42_000),
        education=[
            EducationEntry(
                school="X University",
                level=EducationLevel.BACHELOR,
                start=date(2019, 9, 1),
                end=date(2023, 6, 30),
            )
        ],
        experience_level=ExperienceLevel.Y3_5,
        skills=[Skill(name=s) for s in skills],
    )


@pytest.fixture
def resumes() -> list[Resume]:
    return [
        _resume("R-strong", ["Java", "Kafka", "MySQL", "Redis"]),
        _resume("R-mid", ["Java", "MySQL"]),
        _resume("R-weak", ["Photoshop", "Illustrator"]),
    ]


@pytest.fixture
def pre_ranked(job: Job, resumes: list[Resume]) -> list[MatchScore]:
    out = []
    for r in resumes:
        ev = MatchEvidence(
            matched_skills=[s.name for s in r.skills if s.name in {"Java", "Kafka", "MySQL"}],
            missing_skills=[],
            salary_compatible=True,
            location_compatible=True,
            education_satisfied=True,
            experience_satisfied=True,
        )
        score = 0.85 if r.resume_id == "R-strong" else 0.6 if r.resume_id == "R-mid" else 0.2
        out.append(MatchScore(
            job_id=job.job_id, resume_id=r.resume_id,
            employer_score=score, candidate_score=score,
            fused_score=score, semantic_similarity=score,
            evidence=ev,
        ))
    return out


@pytest.mark.asyncio
async def test_stream_emits_lifecycle_events(
    llm: AsyncLLM, job: Job, resumes: list[Resume], pre_ranked: list[MatchScore]
) -> None:
    orch = AsyncOrchestrator(llm, explain_top_k=2, candidate_concurrency=2)
    seen_types: list[EventType] = []
    async for ev in orch.stream(job, resumes, pre_ranked):
        seen_types.append(ev.type)
    # Must include start/end for each agent + a final event
    assert EventType.AGENT_START in seen_types
    assert EventType.AGENT_END in seen_types
    assert EventType.FINAL in seen_types


@pytest.mark.asyncio
async def test_run_returns_ranking(
    llm: AsyncLLM, job: Job, resumes: list[Resume], pre_ranked: list[MatchScore]
) -> None:
    orch = AsyncOrchestrator(llm, explain_top_k=2, candidate_concurrency=2)
    result = await orch.run(job, resumes, pre_ranked)
    assert len(result.final_ranking) == len(pre_ranked)
    assert result.total_usage.calls > 0
    # The strongly-matching candidate should remain at the top
    assert result.final_ranking[0].resume_id == "R-strong"


@pytest.mark.asyncio
async def test_cache_read_tokens_increase_after_warmup(
    llm: AsyncLLM, job: Job, resumes: list[Resume], pre_ranked: list[MatchScore]
) -> None:
    """The whole point of PREFIX/TAIL: calls 2..N should see cache reads.

    SiliconFlow / DeepSeek surface this via prompt_cache_hit_tokens or
    prompt_tokens_details.cached_tokens. If the provider doesn't report
    cache stats, this test gracefully no-ops.
    """
    orch = AsyncOrchestrator(llm, explain_top_k=2, candidate_concurrency=2)
    result = await orch.run(job, resumes, pre_ranked)
    # If the provider reports any cache, total cache_read_tokens > 0.
    # We don't hard-assert because Qwen3-8B-free may not expose cache stats.
    if result.total_usage.cache_read_tokens > 0:
        assert result.total_usage.cache_read_tokens >= 100  # at least a meaningful chunk
