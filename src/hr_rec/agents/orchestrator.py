"""Hand-rolled orchestrator — runs the four agents sequentially.

Why hand-rolled?
* Avoids the CrewAI ↔ LiteLLM ↔ Ollama version churn documented in
  crewAIInc/crewAI#3031 / #2932 / #3811 (May 2026).
* Keeps the call graph explicit and testable end-to-end.
* Same OpenAI-compatible LLM interface still lets us swap backends.
"""
from __future__ import annotations

from dataclasses import dataclass

from hr_rec.agents.base import AgentContext
from hr_rec.agents.candidate_analyst import CandidateAnalystAgent
from hr_rec.agents.coordinator import CoordinatorAgent
from hr_rec.agents.explainer import ExplainerAgent
from hr_rec.agents.job_analyst import JobAnalystAgent
from hr_rec.agents.llm import LLM
from hr_rec.data.schemas import Job, MatchScore, Resume


@dataclass
class OrchestratorResult:
    final_ranking: list[MatchScore]
    explanations: dict[str, str]
    job_analysis: dict
    candidate_analyses: dict
    trace: list[dict]


class Orchestrator:
    """Run Job-Analyst → Candidate-Analyst → Coordinator → Explainer."""

    def __init__(
        self,
        llm: LLM,
        *,
        explain_top_k: int = 10,
        skip_candidate_analyst: bool = False,
        skip_explainer: bool = False,
    ) -> None:
        self.job_analyst = JobAnalystAgent(llm)
        self.candidate_analyst = CandidateAnalystAgent(llm)
        self.coordinator = CoordinatorAgent(llm)
        self.explainer = ExplainerAgent(llm, top_k=explain_top_k)
        self.skip_candidate_analyst = skip_candidate_analyst
        self.skip_explainer = skip_explainer

    def run(
        self,
        job: Job,
        resumes: list[Resume],
        pre_ranked: list[MatchScore],
    ) -> OrchestratorResult:
        ctx = AgentContext(job=job, candidate_resumes=resumes, pre_ranked=pre_ranked)
        ctx = self.job_analyst.run(ctx)
        if not self.skip_candidate_analyst:
            ctx = self.candidate_analyst.run(ctx)
        ctx = self.coordinator.run(ctx)
        if not self.skip_explainer:
            ctx = self.explainer.run(ctx)
        return OrchestratorResult(
            final_ranking=ctx.final_ranking,
            explanations=ctx.explanations,
            job_analysis=ctx.job_analysis,
            candidate_analyses=ctx.candidate_analyses,
            trace=ctx.trace,
        )
