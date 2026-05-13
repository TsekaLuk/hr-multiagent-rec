# Design Decisions (ADR-style)

A running log of why we made the calls we did. Each entry is a small
ADR (Architecture Decision Record): the context, the choice, the
trade-offs, and what would force us to revisit.

---

## ADR-001 — Same-family Qwen3 stack across encoding / rerank / reasoning

**Context.** A typical 2024-stack glues a BGE encoder, a Cohere
reranker, and a GPT chat LLM. Their tokenizers, pre-training
distributions and embedding spaces diverge.

**Decision.** Use Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B +
Qwen3-8B/DeepSeek-V4-Flash (both Qwen-family at inference time).

**Trade-offs.**
- **+** Shared tokenizer and pre-training distribution improve
  numerical alignment of recall ↔ rerank scores; we ablate this with
  MRL dim 128/256 tests.
- **+** Single vendor reduces operational surface (one HF/ModelScope
  account, one inference style, one billing line).
- **−** Locks us into Qwen3 quality ceiling.
- **−** If Qwen3 changes its API (e.g. Qwen4), we have to migrate
  three components at once.

**Revisit if.** A higher-MTEB encoder or a substantially better
reranker emerges within the same vendor; or if Qwen4 drops three
components in lock-step.

---

## ADR-002 — Hand-rolled orchestrator over CrewAI / LangGraph

**Context.** CrewAI ↔ LiteLLM ↔ Ollama compatibility broke at least
three times in Q1–Q2 2026 (crewAIInc/crewAI #2932, #3031, #3811).

**Decision.** Implement the four-agent pipeline directly on top of
`asyncio` + an OpenAI-compatible client (`httpx.AsyncClient`). No
LiteLLM, no framework adapter.

**Trade-offs.**
- **+** Zero third-party agent-framework dependency.
- **+** Easy to inspect call ordering and message layout for
  prompt-cache exploitation.
- **+** Surface ~150 LOC; entire orchestrator is testable.
- **−** Lose CrewAI's CLI conveniences and observability plug-ins.
- **−** We don't get "free" multi-provider routing — but we don't
  need it (one provider per experiment).

**Revisit if.** A stable agent framework with first-class prompt-cache
support emerges.

---

## ADR-003 — Multiplicative hard-floor penalties in scoring

**Context.** Additive penalties (`base - 0.3`) allow strong technical
skill matches to overcome a missing education requirement. That's
fine for ML metrics but does not reflect real HR workflow where
education / experience are *gates*.

**Decision.** Failing a hard floor *multiplies* the score by 0.4–0.5.
Skill-perfect candidates who fail education go from 1.0 → 0.4 and
sink well below mid-skill candidates who pass.

**Trade-offs.**
- **+** Matches HR mental model.
- **+** Robust to outliers — no need to tune separate "penalty
  weights" per dataset.
- **−** Penalty is hard to interpret on a 0–1 scale.

**Revisit if.** A downstream learn-to-rank head can subsume this
heuristic; or if a domain expert disagrees with multiplicative
combination for a specific country/industry.

---

## ADR-004 — Severe-salary-inversion as a negative sub-score

**Context.** Pure unit-interval scoring under-penalises severe salary
inversions (candidate expects ¥30k, job offers ¥10–15k). After convex
combination with a perfect location match, the candidate still scores
~0.55 — they would never accept the role.

**Decision.** For inversions exceeding 20% of the candidate's
expected ceiling, set the salary sub-score to **−1.0**, then clip the
final candidate-side score to `[0, 1]`. This effectively removes the
candidate from the top of the list.

**Trade-offs.**
- **+** Hard, transparent rule that mirrors a real disqualifier.
- **+** Negative-score-then-clip is a single line of code and easy
  to ablate.
- **−** Not differentiable; downstream learning-to-rank would have
  to special-case it.

**Revisit if.** We move to a fully neural ranker; or if HR partners
report want-tier-2 (medium inversion) cases.

---

## ADR-005 — Reranker on CPU, Embedder on MPS

**Context.** On a 16 GB M4, loading Qwen3-Embedding-0.6B (≈1.2 GB,
MPS) and Qwen3-Reranker-0.6B (≈1.2 GB, MPS) simultaneously kills the
process via OOM in the middle of an experiment grid.

**Decision.** Pin `Reranker(device="cpu")` in `run_experiments.py`.
Embedder stays on MPS for batched corpus encoding (large speed-up).
The Reranker does only ~50 cross-encoder calls per job (manageable
on CPU).

**Trade-offs.**
- **+** Eliminates the OOM kill.
- **+** Embedder's batched encode still benefits from MPS.
- **−** Reranker per-call latency ~3× slower on CPU than MPS.

**Revisit if.** We get a Mac with 32+ GB unified memory, or move to
a Linux/CUDA box.

---

## ADR-006 — Prompt-cache-aware PREFIX/TAIL message layout

**Context.** Candidate-Analyst makes N calls per job — one per
candidate — but only one variable (the candidate's resume)
changes. The system prompt, the JD analysis, and the few-shot
exemplars are identical.

**Decision.** Split every chat call into a *static* `prefix` (system
+ JD analysis + ack) and a *dynamic* `tail` (candidate-specific
block). Concatenate them in a deterministic order so providers can
serve cache reads.

**Trade-offs.**
- **+** Confirmed working: SiliconFlow / DeepSeek / Anthropic all
  surface `prompt_tokens_details.cached_tokens`.
- **+** Saves ~70-80% of input-token cost on paid providers once
  warm.
- **+** Forces a stable, auditable prompt structure.
- **−** Slightly more verbose call site.
- **−** Cache benefit hidden if provider doesn't expose stats.

**Revisit if.** A provider changes its cache key heuristic; or if we
add per-candidate adaptive few-shots (would invalidate the prefix).
