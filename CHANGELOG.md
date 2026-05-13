# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] – Real API milestone (2026-05-14)

### Added
- **Async event-streaming orchestrator** (`hr_rec.agents.async_orchestrator.AsyncOrchestrator`)
  - `stream() → AsyncIterator[Event]` for live progress observation.
  - Per-agent and total `Usage` accounting (input/output/cache_read tokens, calls, seconds).
  - `CandidateTask` registry with `asyncio.Event` for cooperative cancellation.
- **AsyncLLM** (`hr_rec.agents.llm_async.AsyncLLM`)
  - `httpx.AsyncClient` based OpenAI-compatible chat client.
  - PREFIX/TAIL message API for prompt-cache exploitation.
  - JSON-coerce + retry pattern.
- **Verification-style Explainer** — separate API call that reads the
  Coordinator's ranking + structured evidence; counters LLM
  self-confirmation bias seen in single-pass explainers.
- **Cache-warm-then-fan-out** — first candidate sequential (writes
  cache), remaining N−1 via `asyncio.gather` (cache reads).
- **`scripts/demo_async.py`** — live event-streaming CLI demo.

### Changed
- Default LLM model: `Qwen/Qwen3-8B` → `deepseek-ai/DeepSeek-V4-Flash`
  (validated working against SiliconFlow's free tier).
- `Reranker` rewritten as a Qwen3-Reranker causal-LM yes/no scorer
  (`sigmoid(logit_yes - logit_no)`); the sentence-transformers
  `CrossEncoder` is incompatible with the generative-style reranker.
- Reranker pinned to CPU device by default in experiment harness to
  avoid OOM on 16GB M4 when Embedder is on MPS.

### Verified
- **3/3 end-to-end multi-agent tests pass** against the real
  SiliconFlow API in 189 s (`tests/e2e/test_async_orchestrator_real_llm.py`).
- 13 new unit tests on async-orchestrator deterministic logic
  (cache-layout invariants, coordinator scoring, usage arithmetic).

## [0.1.0] – Foundations (2026-05-13)

### Added
- Pydantic data contracts: `Resume`, `Job`, `Skill`, `SalaryRange`,
  `MatchScore`, `MatchEvidence`.
- Deterministic Tianchi-style synthetic corpus generator
  (300 jobs × 800 resumes → 24,145 ground-truth pairs at seed 42).
- Real Qwen3-Embedding-0.6B wrapper with MPS auto-detect, MRL
  dimension truncation, ModelScope path resolution.
- FAISS-cpu indexer (flat + IVF) with save/load round-trip and
  ≥85% recall@10 verification.
- Bidirectional matching scorer with multiplicative hard-floor
  penalties on education and experience, and a severe-salary-inversion
  negative-score branch.
- Information-retrieval metrics: P@K, R@K, nDCG@K (graded), MRR.
- BM25 and TF-IDF baselines over jieba-tokenised Chinese.
- 10-config ablation harness (`configs/experiments.yaml`,
  `scripts/run_experiments.py`).
- arXiv paper skeleton (ACL-style) + 11 cited works.
- 8-document thesis-materials pack (algorithms, architecture,
  data-flow, experimental setup, related-work summary, case studies,
  defense FAQ).

### Bench
- BM25:                                 P@10=0.967  nDCG@10=0.829  MRR=0.983
- TF-IDF:                               P@10=0.917  nDCG@10=0.783  MRR=0.944
- Qwen3-Embedding only:                 P@10=0.930  nDCG@10=0.795  MRR=0.967
- Qwen3-Embedding + Bidirectional:      P@10=0.947  nDCG@10=0.889  MRR=1.000

### Engineering
- Strict no-mocks policy enforced in tests; pytest skips cleanly when
  models / API keys are unavailable.
- Hand-rolled orchestrator (sync) — avoids the CrewAI ↔ LiteLLM ↔
  Ollama compatibility regressions of Q1/Q2 2026
  (crewAIInc/crewAI #2932, #3031, #3811).
