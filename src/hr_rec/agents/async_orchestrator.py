"""Async event-streaming orchestrator with parallel candidate analysis.

Architecture upgrades drawn from claude-code-notes/04-Agent协调/*:

1. **Async-generator coordinator** — yields events as soon as they're
   produced; caller can render progress / cancel mid-run.
2. **PREFIX/TAIL cache-friendly layout** — every CandidateAnalyst call
   shares the same JD+rubric+fewshot prefix; only the candidate's
   resume changes in the tail. SiliconFlow / DeepSeek / Anthropic
   prompt caches will charge cache-read pricing on calls 2..N.
3. **Parallel fan-out** — N candidates analysed via
   ``asyncio.gather`` with a semaphore (M4 16 GB: default 8).
4. **Warm-cache then fan-out** — fire the first candidate synchronously
   so the prefix is *written* to cache, then gather the rest as
   *cache reads*.
5. **Verification-style Explainer** — Explainer is a *separate* call
   that reads Coordinator's ranking; this counters the LLM
   self-confirmation bias that single-pass explainers fall prey to.
6. **Task registry + cooperative cancellation** — each candidate run
   is a tracked Task; ``asyncio.CancelledError`` propagates cleanly.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from hr_rec.agents.events import Event, EventType, Usage
from hr_rec.agents.llm_async import AsyncLLM, AsyncLLMError
from hr_rec.data.schemas import Job, MatchScore, Resume

logger = logging.getLogger(__name__)


# ---- agent prompts (extracted so PREFIX is stable for caching) ----------


_JOB_ANALYST_SYS = {
    "role": "system",
    "content": (
        "你是一名资深技术招聘官，输出严格的 JSON。\n"
        "字段：core_skills (≤8), nice_to_have (≤5), "
        "must_have_constraints, responsibility_summary, deal_breakers。"
    ),
}


def _job_analyst_user(job: Job) -> dict[str, Any]:
    body = (
        f"【岗位】{job.title} @ {job.company}\n"
        f"【地点】{job.location}\n"
        f"【描述】\n{job.description or job.raw_text or '(无)'}"
    )
    return {"role": "user", "content": body}


_CAND_ANALYST_SYS = {
    "role": "system",
    "content": (
        "你是经验丰富的猎头顾问，逐条评估候选人对岗位的匹配。\n"
        "返回严格 JSON：{strengths(≤4), gaps(≤4), risk_flags, "
        "overall_fit∈{high,medium,low}, rationale}。仅返回 JSON。"
    ),
}


def _cand_prefix(job: Job, job_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Cache-friendly stable prefix shared across all candidate calls."""
    core = "、".join(job_analysis.get("core_skills") or [s.name for s in job.required_skills])
    constraints = "、".join(job_analysis.get("must_have_constraints") or []) or "(无)"
    summary = job_analysis.get("responsibility_summary") or job.title
    job_block = {
        "role": "user",
        "content": (
            "本批次共享的岗位背景（请记住，下条用户消息会给出每位候选人简历）：\n"
            f"【岗位关键技能】{core}\n"
            f"【硬约束】{constraints}\n"
            f"【职责】{summary}"
        ),
    }
    ack = {
        "role": "assistant",
        "content": "明白了，请按上述岗位标准评估接下来给出的候选人。",
    }
    return [_CAND_ANALYST_SYS, job_block, ack]


def _cand_tail(r: Resume) -> list[dict[str, Any]]:
    salary = (
        f"{r.expected_salary.min_cny}-{r.expected_salary.max_cny}"
        if r.expected_salary else "(未指明)"
    )
    body = (
        f"【候选人 ID】{r.resume_id}\n"
        f"【城市/期望】{r.location} / { '、'.join(r.expected_locations) or '(未指明)' }\n"
        f"【期望薪资】{salary}\n"
        f"【经验】{r.experience_level.value}\n"
        f"【教育】{ '、'.join(f'{e.school}({e.level.value})' for e in r.education) or '(未提供)' }\n"
        f"【技能】{ '、'.join(s.name for s in r.skills) or '(未提供)' }\n"
        f"【简历摘要】\n{r.summary or r.raw_text[:500] or '(无)'}"
    )
    return [{"role": "user", "content": body}]


_EXPLAINER_SYS = {
    "role": "system",
    "content": (
        "你是 *验证型* 解释 Agent：根据下游打分与结构化证据，"
        "用 80 字以内中文写出推荐理由，先讲匹配亮点，再点出最大风险，"
        "**不重复人云亦云，必须引证据**。"
    ),
}


# ---- coordinator state ---------------------------------------------------


@dataclass
class CandidateTask:
    resume_id: str
    status: str = "pending"  # pending / running / done / error / cancelled
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    cancel: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class AsyncOrchestratorResult:
    final_ranking: list[MatchScore]
    explanations: dict[str, str]
    job_analysis: dict[str, Any]
    candidate_analyses: dict[str, dict[str, Any]]
    tasks: dict[str, CandidateTask]
    total_usage: Usage


# ---- orchestrator --------------------------------------------------------


_FIT_BONUS = {"high": 0.08, "medium": 0.0, "low": -0.08}


class AsyncOrchestrator:
    """Async event-stream PJF orchestrator.

    Usage::

        async with AsyncLLM() as llm:
            orch = AsyncOrchestrator(llm)
            async for ev in orch.stream(job, resumes, pre_ranked):
                print(ev.type, ev.agent)
            result = orch.last_result
    """

    def __init__(
        self,
        llm: AsyncLLM,
        *,
        explain_top_k: int = 10,
        candidate_concurrency: int = 8,
    ) -> None:
        self.llm = llm
        self.explain_top_k = explain_top_k
        self.candidate_concurrency = candidate_concurrency
        self.last_result: AsyncOrchestratorResult | None = None

    async def stream(
        self,
        job: Job,
        resumes: list[Resume],
        pre_ranked: list[MatchScore],
    ) -> AsyncIterator[Event]:
        total_usage = Usage()
        tasks: dict[str, CandidateTask] = {r.resume_id: CandidateTask(r.resume_id) for r in resumes}

        # ---- 1. Job-Analyst ---------------------------------------------
        yield Event(EventType.AGENT_START, agent="job_analyst")
        ja, ja_usage = await self._job_analyst(job)
        total_usage = total_usage + ja_usage
        yield Event(EventType.USAGE, agent="job_analyst", usage=ja_usage)
        yield Event(EventType.AGENT_END, agent="job_analyst",
                    payload={"core_skills": ja.get("core_skills") or []})

        # ---- 2. Candidate-Analyst (warm cache + parallel fan-out) ------
        yield Event(EventType.AGENT_START, agent="candidate_analyst")
        ca_prefix = _cand_prefix(job, ja)
        analyses: dict[str, dict[str, Any]] = {}

        async def analyse_one(r: Resume) -> None:
            t = tasks[r.resume_id]
            t.status = "running"
            if t.cancel.is_set():
                t.status = "cancelled"
                return
            try:
                obj, usage = await self.llm.chat_json(ca_prefix, _cand_tail(r))
                t.usage = usage
                t.status = "done"
                analyses[r.resume_id] = obj
            except AsyncLLMError as e:
                t.status = "error"
                t.error = str(e)

        # WARM the cache with the first candidate (sequential).
        if resumes:
            await analyse_one(resumes[0])
            yield Event(
                EventType.CANDIDATE_PROFILED,
                agent="candidate_analyst",
                payload={"resume_id": resumes[0].resume_id, "cache_warming": True},
                usage=tasks[resumes[0].resume_id].usage,
            )
            total_usage = total_usage + tasks[resumes[0].resume_id].usage

        # Then FAN OUT the rest — each call reads the cached prefix.
        sem = asyncio.Semaphore(self.candidate_concurrency)

        async def gated(r: Resume) -> None:
            async with sem:
                await analyse_one(r)

        results = await asyncio.gather(
            *(gated(r) for r in resumes[1:]),
            return_exceptions=True,
        )
        for r, exc in zip(resumes[1:], results, strict=True):
            if isinstance(exc, Exception):
                yield Event(EventType.ERROR, agent="candidate_analyst",
                            payload={"resume_id": r.resume_id, "error": repr(exc)})
                continue
            yield Event(
                EventType.CANDIDATE_PROFILED,
                agent="candidate_analyst",
                payload={"resume_id": r.resume_id},
                usage=tasks[r.resume_id].usage,
            )
            total_usage = total_usage + tasks[r.resume_id].usage
        yield Event(EventType.AGENT_END, agent="candidate_analyst",
                    payload={"profiled": sum(1 for t in tasks.values() if t.status == "done")})

        # ---- 3. Coordinator (deterministic; no LLM call) ----------------
        yield Event(EventType.AGENT_START, agent="coordinator")
        adjusted = self._coordinate(pre_ranked, analyses)
        yield Event(EventType.AGENT_END, agent="coordinator",
                    payload={"n_ranked": len(adjusted)})

        # ---- 4. Explainer (verification-style; parallel per-candidate) -
        yield Event(EventType.AGENT_START, agent="explainer")
        top_to_explain = adjusted[: self.explain_top_k]
        explanations = await self._explain_parallel(job, top_to_explain, analyses)
        # explain_parallel returns (text_map, usage); record usage
        explanations_text, exp_usage = explanations
        total_usage = total_usage + exp_usage
        yield Event(EventType.USAGE, agent="explainer", usage=exp_usage)
        yield Event(EventType.AGENT_END, agent="explainer",
                    payload={"n_explained": len(explanations_text)})

        # ---- final --------------------------------------------------------
        self.last_result = AsyncOrchestratorResult(
            final_ranking=adjusted,
            explanations=explanations_text,
            job_analysis=ja,
            candidate_analyses=analyses,
            tasks=tasks,
            total_usage=total_usage,
        )
        yield Event(EventType.FINAL, payload={
            "total_calls": total_usage.calls,
            "cache_read_tokens": total_usage.cache_read_tokens,
            "total_seconds": total_usage.seconds,
        }, usage=total_usage)

    async def run(
        self,
        job: Job,
        resumes: list[Resume],
        pre_ranked: list[MatchScore],
    ) -> AsyncOrchestratorResult:
        """Convenience: consume the stream and return the final result."""
        async for _ in self.stream(job, resumes, pre_ranked):
            pass
        assert self.last_result is not None
        return self.last_result

    # ---- internals ----------------------------------------------------

    async def _job_analyst(self, job: Job) -> tuple[dict[str, Any], Usage]:
        try:
            obj, usage = await self.llm.chat_json(
                prefix=[_JOB_ANALYST_SYS],
                tail=[_job_analyst_user(job)],
            )
            return obj, usage
        except AsyncLLMError as e:
            logger.warning("Job-Analyst LLM call failed: %s — using structured fallback", e)
            return (
                {
                    "core_skills": [s.name for s in job.required_skills],
                    "nice_to_have": [s.name for s in job.preferred_skills],
                    "must_have_constraints": [],
                    "responsibility_summary": job.title,
                    "deal_breakers": [],
                },
                Usage(),
            )

    def _coordinate(
        self,
        pre_ranked: list[MatchScore],
        analyses: dict[str, dict[str, Any]],
    ) -> list[MatchScore]:
        out: list[MatchScore] = []
        for ms in pre_ranked:
            profile = analyses.get(ms.resume_id, {})
            fit = (profile.get("overall_fit") or "medium").lower()
            risk = profile.get("risk_flags") or []
            bonus = _FIT_BONUS.get(fit, 0.0)
            penalty = min(0.10, 0.03 * len(risk))
            new = max(0.0, min(1.0, ms.fused_score + bonus - penalty))
            out.append(
                MatchScore(
                    job_id=ms.job_id,
                    resume_id=ms.resume_id,
                    employer_score=ms.employer_score,
                    candidate_score=ms.candidate_score,
                    fused_score=new,
                    semantic_similarity=ms.semantic_similarity,
                    rerank_score=ms.rerank_score,
                    evidence=ms.evidence,
                )
            )
        out.sort(key=lambda x: x.fused_score, reverse=True)
        return out

    async def _explain_parallel(
        self,
        job: Job,
        top: list[MatchScore],
        analyses: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, str], Usage]:
        if not top:
            return {}, Usage()
        prefix = [
            _EXPLAINER_SYS,
            {
                "role": "user",
                "content": (
                    f"本批解释的岗位（共享背景）：\n"
                    f"【岗位】{job.title} @ {job.company or '(未指明)'} | {job.location}\n"
                    f"【必需技能】{ '、'.join(s.name for s in job.required_skills) }"
                ),
            },
            {"role": "assistant", "content": "明白，请告诉我每位候选人的得分与证据。"},
        ]

        async def explain_one(ms: MatchScore) -> tuple[str, str, Usage]:
            profile = analyses.get(ms.resume_id, {})
            ev = ms.evidence.rationale if ms.evidence else ""
            tail = [{
                "role": "user",
                "content": (
                    f"【候选人 ID】{ms.resume_id}\n"
                    f"【综合得分】{ms.fused_score:.2f}\n"
                    f"【证据】{ev}\n"
                    f"【优势】{ '、'.join(profile.get('strengths') or []) or '(待补)' }\n"
                    f"【缺口】{ '、'.join(profile.get('gaps') or []) or '(无)' }"
                ),
            }]
            try:
                text, usage = await self.llm.chat(prefix, tail, max_tokens=200)
                return ms.resume_id, text.strip(), usage
            except AsyncLLMError:
                return ms.resume_id, ev, Usage()

        # Warm-then-fan-out so calls 2..N benefit from prompt cache.
        first_id, first_text, first_usage = await explain_one(top[0])
        explanations = {first_id: first_text}
        total = first_usage
        if len(top) > 1:
            results = await asyncio.gather(
                *(explain_one(ms) for ms in top[1:]),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    continue
                rid, text, u = r
                explanations[rid] = text
                total = total + u
        return explanations, total
