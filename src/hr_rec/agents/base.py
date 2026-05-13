"""Agent base contract.

Every agent is a typed, stateless function:

    Agent.run(input: AgentInput) -> AgentOutput

Errors are raised, not swallowed; the orchestrator decides retry/abort.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hr_rec.agents.llm import LLM
from hr_rec.data.schemas import Job, MatchScore, Resume


@dataclass
class AgentContext:
    """Shared blackboard the orchestrator passes between agents."""

    job: Job
    candidate_resumes: list[Resume]
    pre_ranked: list[MatchScore] = field(default_factory=list)
    job_analysis: dict[str, Any] = field(default_factory=dict)
    candidate_analyses: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_ranking: list[MatchScore] = field(default_factory=list)
    explanations: dict[str, str] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)


class Agent(ABC):
    name: str

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    @abstractmethod
    def run(self, ctx: AgentContext) -> AgentContext:
        ...

    def _log(self, ctx: AgentContext, event: str, **kw: Any) -> None:
        ctx.trace.append({"agent": self.name, "event": event, **kw})
