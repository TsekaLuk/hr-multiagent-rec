# Architecture

This document is the load-bearing technical reference for HR-MultiAgent-Rec.
Companion documents:

* [`decisions.md`](decisions.md) — six ADRs explaining the load-bearing
  choices (same-family stack, hand-rolled orchestrator, multiplicative
  hard floors, salary-inversion negative-score, MPS vs CPU device split,
  PREFIX/TAIL cache layout).
* [`agent_upgrade.md`](agent_upgrade.md) — the async-coordinator
  upgrade derived from `claude-code-notes/04-Agent协调/*`, with
  every pattern's port mapping.

## Layered view

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer                | Responsibility                | Module             │
├──────────────────────────────────────────────────────────────────────────┤
│ 1. Data contracts    | Validated typed Resume/Job    | hr_rec.data.schemas│
│                      | + synthetic + Tianchi loader  | hr_rec.data.*      │
├──────────────────────────────────────────────────────────────────────────┤
│ 2. Semantic encoding | Qwen3-Embedding wrapper       | hr_rec.encoding.   │
│                      | + FAISS IVF/flat indexer      |   embedder, indexer│
├──────────────────────────────────────────────────────────────────────────┤
│ 3. Matching          | Qwen3-Reranker yes/no scorer  | hr_rec.matching.*  │
│                      | + bidirectional scoring       |                    │
├──────────────────────────────────────────────────────────────────────────┤
│ 4. Multi-Agent       | Job-Analyst / Cand-Analyst /  | hr_rec.agents.*    │
│   (sync orchestrator)| Coordinator / Explainer       | orchestrator.py    │
│   (async upgrade)    | + event stream + cache layout | async_orchestrator │
├──────────────────────────────────────────────────────────────────────────┤
│ 5. Pipeline          | Composes layers 1–4 with      | hr_rec.pipeline    │
│                      | ablation switches             |                    │
├──────────────────────────────────────────────────────────────────────────┤
│ 6. Baselines         | BM25 / TF-IDF                 | hr_rec.baselines   │
├──────────────────────────────────────────────────────────────────────────┤
│ 7. Evaluation        | P@K / R@K / nDCG@K / MRR      | hr_rec.evaluation  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Why same-family models?

A typical 2024-stack glues a `bge-large-zh` encoder, a Cohere reranker and a
GPT-4o reasoner — three different pre-training distributions, three
tokenizers, three semantic spaces. We use **Qwen3-Embedding-0.6B**,
**Qwen3-Reranker-0.6B** and **DeepSeek-V4-Flash (Qwen3-family) via
SiliconFlow**. The recall vector, rerank logit and agent reasoning
therefore share a tokenizer and the underlying representation manifold.

The Qwen3-Reranker is *not* a regression-head cross-encoder — it's a
causal LM that scores `sigmoid(logit_yes - logit_no)`. Our
implementation in `hr_rec.matching.reranker` uses this generative
formulation directly; the standard sentence-transformers `CrossEncoder`
class is incompatible and threw a tensor-reshape error during our
initial integration.

## Why hand-roll the orchestrator?

CrewAI 1.x ships native integrations for OpenAI/Anthropic/Gemini/Bedrock
and falls back to **LiteLLM** for everything else, including Ollama and
OpenAI-compatible China endpoints (SiliconFlow, DeepSeek). Between Jan
and May 2026, this LiteLLM fallback broke three times
(crewAIInc/crewAI #2932, #3031, #3811). Our **sync** orchestrator is a
60-line sequential runner that calls the OpenAI SDK directly; our
**async** orchestrator (recommended) is `httpx.AsyncClient` + a small
event-stream coordinator (~150 LOC). Both are robust to LiteLLM
regressions and trivially support any OpenAI-compatible endpoint.

## Bidirectional scoring formula

For job *j* and resume *r*:

$$
s_e(j, r) = \underbrace{(w_R\,\mathrm{cov}_R + w_P\,\mathrm{cov}_P)}_{\text{skill base}}
            \cdot \pi_{\text{edu}}(j, r) \cdot \pi_{\text{exp}}(j, r)
$$

$$
s_c(j, r) = w_L\,\ell(j, r) + w_S\,\sigma(j, r)
$$

$$
\hat{s}(j, r) = \alpha \cdot s_e + (1-\alpha) \cdot s_c
$$

with multiplicative hard-floor penalties $\pi_{\text{edu}}=0.4$ /
$\pi_{\text{exp}}=0.5$ if the candidate is below the floor, and
$\sigma(j, r)=-1.0$ on a *severe* (≥20%) salary inversion to make
inversions effectively disqualifying after clipping to $[0, 1]$. We
choose $\alpha=0.6$ — the employer side dominates because the corpus is
recruiter-led. See [`decisions.md`](decisions.md) ADR-003 and ADR-004
for the rationale.

## Layer-by-layer test coverage (May 2026 snapshot)

| Layer        | Unit tests | Integration | E2E (real API) | What's exercised                          |
|--------------|-----------:|------------:|---------------:|-------------------------------------------|
| schemas      |        28  |          –  |              – | All validators + edge cases               |
| synthetic    |        26  |          –  |              – | Determinism, distribution, GT consistency |
| metrics      |        21  |          –  |              – | P/R/nDCG edge cases + DCG formula sanity  |
| scoring      |        23  |          –  |              – | Long-tail business cases (no mocks)       |
| llm_json     |         9  |          –  |              – | JSON coercion robustness                  |
| async-orch   |        13  |          –  |              3 | Cache layout, coord math, lifecycle       |
| indexer      |         –  |          8  |              – | Flat + IVF + roundtrip + recall bound     |
| baselines    |         –  |          3  |              – | BM25/TF-IDF on synthetic corpus           |
| embedder     |         –  |         17  |              – | Real Qwen3-Embedding (MPS-fallback)       |
| agents (sync)|         –  |          –  |              3 | Real SiliconFlow JSON contracts           |
| **Total**    |   **120**  |     **28**  |          **6** |                                           |
