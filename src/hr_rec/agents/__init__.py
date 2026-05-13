"""Multi-agent layer.

Two orchestrators are provided:

* :class:`Orchestrator` — original sync sequential implementation
  (kept for backward compatibility with ``scripts/run_experiments.py``).
* :class:`AsyncOrchestrator` — recommended path; streams events,
  parallelises Candidate-Analyst, uses prompt-cache-aware message
  layout for SiliconFlow / DeepSeek / Anthropic.
"""
from hr_rec.agents.async_orchestrator import (
    AsyncOrchestrator,
    AsyncOrchestratorResult,
    CandidateTask,
)
from hr_rec.agents.base import Agent, AgentContext
from hr_rec.agents.candidate_analyst import CandidateAnalystAgent
from hr_rec.agents.coordinator import CoordinatorAgent
from hr_rec.agents.events import Event, EventType, Usage
from hr_rec.agents.explainer import ExplainerAgent
from hr_rec.agents.job_analyst import JobAnalystAgent
from hr_rec.agents.llm import LLM, LLMError
from hr_rec.agents.llm_async import AsyncLLM, AsyncLLMError
from hr_rec.agents.orchestrator import Orchestrator, OrchestratorResult

__all__ = [
    # sync (legacy)
    "Agent",
    "AgentContext",
    "CandidateAnalystAgent",
    "CoordinatorAgent",
    "ExplainerAgent",
    "JobAnalystAgent",
    "LLM",
    "LLMError",
    "Orchestrator",
    "OrchestratorResult",
    # async (recommended)
    "AsyncOrchestrator",
    "AsyncOrchestratorResult",
    "AsyncLLM",
    "AsyncLLMError",
    "CandidateTask",
    "Event",
    "EventType",
    "Usage",
]
