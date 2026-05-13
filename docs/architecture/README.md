# Architecture

This document is the load-bearing technical reference for HR-MultiAgent-Rec.
It deliberately leaves the *narrative* of the work to the
[arXiv paper](../../paper/main.tex) and the *how-to* to
[reproduce/](../reproduce/README.md).

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
│ 3. Matching          | Qwen3-Reranker cross-encoder  | hr_rec.matching.*  │
│                      | + bidirectional scoring       |                    │
├──────────────────────────────────────────────────────────────────────────┤
│ 4. Multi-Agent       | Job-Analyst / Cand-Analyst /  | hr_rec.agents.*    │
│                      | Coordinator / Explainer       |                    │
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
**Qwen3-Reranker-0.6B** and **Qwen3-8B** all built on the same Qwen3
foundation. The recall vector, rerank logit and agent reasoning therefore
share a tokenizer and the underlying representation manifold. The
ablation experiment **MRL truncation @ 256/128** stress-tests this:
truncating dimensions still preserves the ordering produced by the
reranker because both were trained jointly under MRL.

## Why hand-roll the orchestrator?

CrewAI 1.x ships native integrations for OpenAI/Anthropic/Gemini/Bedrock
and falls back to **LiteLLM** for everything else, including Ollama and
OpenAI-compatible China endpoints (SiliconFlow, DeepSeek). Between Jan
and May 2026, this LiteLLM fallback broke three times
(crewAIInc/crewAI #2932, #3031, #3811). Our orchestrator is a 60-line
sequential runner that calls the OpenAI SDK directly; it is robust to
LiteLLM regressions and trivially supports any OpenAI-compatible
endpoint.

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
recruiter-led.

## Layer-by-layer test coverage (May 2026 snapshot)

| Layer        | Unit tests | Integration | What's exercised                          |
|--------------|-----------:|------------:|-------------------------------------------|
| schemas      |        28  |          –  | All validators + edge cases               |
| synthetic    |        26  |          –  | Determinism, distribution, GT consistency |
| metrics      |        21  |          –  | P/R/nDCG edge cases + DCG formula sanity  |
| scoring      |        23  |          –  | Long-tail business cases (no mocks)       |
| llm_json     |         9  |          –  | JSON coercion robustness                  |
| indexer      |         –  |          8  | Flat + IVF + roundtrip + recall bound     |
| baselines    |         –  |          3  | BM25/TF-IDF on synthetic corpus           |
| embedder     |         –  |         14  | Real Qwen3-Embedding (MPS-fallback)       |
| **Total**    |   **107**  |     **25**  |                                           |
