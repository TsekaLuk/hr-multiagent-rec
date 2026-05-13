"""Coordinator agent: fuses bidirectional scores + LLM judgements into a final ranking."""
from __future__ import annotations

from hr_rec.agents.base import Agent, AgentContext
from hr_rec.data.schemas import MatchScore


_FIT_BONUS = {"high": 0.08, "medium": 0.0, "low": -0.08}


class CoordinatorAgent(Agent):
    name = "coordinator"

    def run(self, ctx: AgentContext) -> AgentContext:
        # Start from the algorithmic pre-ranking and adjust by Candidate-Analyst signals.
        adjusted: list[MatchScore] = []
        for ms in ctx.pre_ranked:
            profile = ctx.candidate_analyses.get(ms.resume_id, {})
            fit = (profile.get("overall_fit") or "medium").lower()
            risk_flags = profile.get("risk_flags") or []

            bonus = _FIT_BONUS.get(fit, 0.0)
            # Each distinct risk flag costs a small amount.
            risk_penalty = min(0.10, 0.03 * len(risk_flags))
            new_fused = max(0.0, min(1.0, ms.fused_score + bonus - risk_penalty))

            adjusted.append(
                MatchScore(
                    job_id=ms.job_id,
                    resume_id=ms.resume_id,
                    employer_score=ms.employer_score,
                    candidate_score=ms.candidate_score,
                    fused_score=new_fused,
                    semantic_similarity=ms.semantic_similarity,
                    rerank_score=ms.rerank_score,
                    evidence=ms.evidence,
                )
            )

        adjusted.sort(key=lambda x: x.fused_score, reverse=True)
        ctx.final_ranking = adjusted
        self._log(ctx, "ranked", n=len(adjusted))
        return ctx
