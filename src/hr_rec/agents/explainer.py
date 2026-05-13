"""Explainer agent: produces a human-readable rationale per recommended candidate."""
from __future__ import annotations

from hr_rec.agents.base import Agent, AgentContext


_PROMPT = """请用 80 字以内中文为以下推荐生成一段招聘官友好的推荐理由：

【岗位】{job_title} @ {company}
【候选人ID】{resume_id}
【综合得分】{fused:.2f}
【证据】{evidence}
【优势】{strengths}
【缺口】{gaps}

要求：先讲匹配亮点，再点出最大风险，不写废话，不使用 markdown。"""


class ExplainerAgent(Agent):
    name = "explainer"

    def __init__(self, llm, *, top_k: int = 10) -> None:  # type: ignore[no-untyped-def]
        super().__init__(llm)
        self.top_k = top_k

    def run(self, ctx: AgentContext) -> AgentContext:
        for ms in ctx.final_ranking[: self.top_k]:
            profile = ctx.candidate_analyses.get(ms.resume_id, {})
            ev = ms.evidence.rationale if ms.evidence else ""
            prompt = _PROMPT.format(
                job_title=ctx.job.title,
                company=ctx.job.company or "(未指明公司)",
                resume_id=ms.resume_id,
                fused=ms.fused_score,
                evidence=ev,
                strengths="、".join(profile.get("strengths") or []) or "(待补充)",
                gaps="、".join(profile.get("gaps") or []) or "(无)",
            )
            try:
                text = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=200)
            except Exception as e:
                self._log(ctx, "llm_error", resume_id=ms.resume_id, error=str(e))
                text = ev
            ctx.explanations[ms.resume_id] = text.strip()
        self._log(ctx, "explained", n=len(ctx.explanations))
        return ctx
