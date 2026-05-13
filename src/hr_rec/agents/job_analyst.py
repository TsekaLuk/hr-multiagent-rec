"""Job-Analyst agent: extracts structured requirements from a job posting."""
from __future__ import annotations

import json

from hr_rec.agents.base import Agent, AgentContext


_PROMPT = """你是一名资深技术招聘官。请从下面的岗位描述中抽取结构化要求，并以 JSON 输出。

【岗位标题】{title}
【公司】{company}
【地点】{location}
【描述】
{description}

输出 JSON 格式：
{{
  "core_skills": ["..."],         // 核心必需技能 (最多 8 个)
  "nice_to_have": ["..."],         // 加分项 (最多 5 个)
  "must_have_constraints": ["..."],// 硬约束 (学历、经验、资质)
  "responsibility_summary": "...", // 一句话总结岗位职责
  "deal_breakers": ["..."]         // 哪些情况会直接淘汰
}}
"""


class JobAnalystAgent(Agent):
    name = "job_analyst"

    def run(self, ctx: AgentContext) -> AgentContext:
        j = ctx.job
        prompt = _PROMPT.format(
            title=j.title,
            company=j.company,
            location=j.location,
            description=j.description or j.raw_text or "(无)",
        )
        try:
            analysis = self.llm.chat_json([{"role": "user", "content": prompt}])
        except Exception as e:
            self._log(ctx, "llm_error", error=str(e))
            # Conservative fallback: derive from the structured fields directly.
            analysis = {
                "core_skills": [s.name for s in j.required_skills],
                "nice_to_have": [s.name for s in j.preferred_skills],
                "must_have_constraints": [],
                "responsibility_summary": j.title,
                "deal_breakers": [],
            }
        ctx.job_analysis = analysis
        self._log(ctx, "analyzed", n_core=len(analysis.get("core_skills") or []))
        return ctx
