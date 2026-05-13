"""Sync façade over :class:`hr_rec.agents.AsyncOrchestrator`.

Lets the existing sync :class:`hr_rec.pipeline.Pipeline` use the new
async orchestrator without changes — each ``run()`` call spawns its
own event loop, runs the async orchestrator to completion, then
returns. Avoids leaking event-loop concerns into the rest of the
codebase.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from hr_rec.agents.async_orchestrator import AsyncOrchestrator
from hr_rec.agents.events import Usage, lookup_pricing
from hr_rec.agents.llm_async import AsyncLLM
from hr_rec.data.schemas import Job, MatchScore, Resume

logger = logging.getLogger(__name__)


@dataclass
class _SyncResult:
    final_ranking: list[MatchScore]
    explanations: dict[str, str]
    job_analysis: dict
    candidate_analyses: dict
    trace: list[dict]
    total_usage: Usage = field(default_factory=Usage)


class AsyncOrchestratorFacade:
    """Pretends to be a sync ``Orchestrator`` for ``Pipeline``."""

    def __init__(
        self,
        *,
        model: str = "deepseek-ai/DeepSeek-V4-Flash",
        provider: str = "siliconflow",
        temperature: float = 0.2,
        concurrency: int = 4,
        explain_top_k: int = 5,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.concurrency = concurrency
        self.explain_top_k = explain_top_k
        self.timeout = timeout

    def run(
        self,
        job: Job,
        resumes: list[Resume],
        pre_ranked: list[MatchScore],
    ) -> _SyncResult:
        async def _go() -> _SyncResult:
            async with AsyncLLM(
                model=self.model,
                provider=self.provider,
                temperature=self.temperature,
                concurrency=self.concurrency,
                timeout=self.timeout,
            ) as llm:
                orch = AsyncOrchestrator(
                    llm,
                    explain_top_k=self.explain_top_k,
                    candidate_concurrency=self.concurrency,
                )
                result = await orch.run(job, resumes, pre_ranked)
                pricing = lookup_pricing(self.model)
                cost = result.total_usage.cost_usd(pricing)
                logger.info(
                    "multi-agent job=%s calls=%d in=%d out=%d cache_read=%d "
                    "cache_hit_rate=%.1f%% seconds=%.1f cost_usd=%.5f",
                    job.job_id,
                    result.total_usage.calls,
                    result.total_usage.input_tokens,
                    result.total_usage.output_tokens,
                    result.total_usage.cache_read_tokens,
                    100 * result.total_usage.cache_hit_rate(),
                    result.total_usage.seconds,
                    cost,
                )
                return _SyncResult(
                    final_ranking=result.final_ranking,
                    explanations=result.explanations,
                    job_analysis=result.job_analysis,
                    candidate_analyses=result.candidate_analyses,
                    trace=[],
                    total_usage=result.total_usage,
                )

        return asyncio.run(_go())
