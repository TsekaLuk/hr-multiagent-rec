"""Candidate-Analyst agent: profiles each candidate against the job spec."""
from __future__ import annotations

from hr_rec.agents.base import Agent, AgentContext


_PROMPT = """你是一名经验丰富的猎头顾问。请评估候选人对岗位的匹配情况。

【岗位关键技能】{core_skills}
【岗位硬约束】{constraints}
【岗位职责】{summary}

【候选人 ID】{resume_id}
【所在城市】{location}
【期望城市】{expected_cities}
【期望薪资 (CNY/月)】{expected_salary}
【经验水平】{exp}
【教育】{edu}
【技能】{skills}
【简历摘要】
{summary_text}

输出 JSON：
{{
  "strengths": ["..."],          // 优势 (最多 4 条)
  "gaps": ["..."],                // 缺口 (最多 4 条)
  "risk_flags": ["..."],          // 风险点 (薪资倒挂、地点不符等)
  "overall_fit": "high/medium/low",
  "rationale": "..."              // 简短理由
}}
"""


class CandidateAnalystAgent(Agent):
    name = "candidate_analyst"

    def run(self, ctx: AgentContext) -> AgentContext:
        ja = ctx.job_analysis or {}
        core = "、".join(ja.get("core_skills") or [s.name for s in ctx.job.required_skills])
        constraints = "、".join(ja.get("must_have_constraints") or [])
        summary = ja.get("responsibility_summary") or ctx.job.title

        for r in ctx.candidate_resumes:
            prompt = _PROMPT.format(
                core_skills=core or "(未提供)",
                constraints=constraints or "(无)",
                summary=summary,
                resume_id=r.resume_id,
                location=r.location,
                expected_cities="、".join(r.expected_locations) or "(未指明)",
                expected_salary=(
                    f"{r.expected_salary.min_cny}-{r.expected_salary.max_cny}"
                    if r.expected_salary
                    else "(未指明)"
                ),
                exp=r.experience_level.value,
                edu="、".join(f"{e.school}({e.level.value})" for e in r.education) or "(未提供)",
                skills="、".join(s.name for s in r.skills) or "(未提供)",
                summary_text=r.summary or r.raw_text[:500] or "(无)",
            )
            try:
                profile = self.llm.chat_json([{"role": "user", "content": prompt}])
            except Exception as e:
                self._log(ctx, "llm_error", resume_id=r.resume_id, error=str(e))
                profile = {
                    "strengths": [],
                    "gaps": [],
                    "risk_flags": [],
                    "overall_fit": "medium",
                    "rationale": "fallback (LLM unavailable)",
                }
            ctx.candidate_analyses[r.resume_id] = profile
        self._log(ctx, "profiled", n=len(ctx.candidate_analyses))
        return ctx
