<div align="center">

# HR-MultiAgent-Rec

**A Multi-Agent Recommendation & Optimization Framework for Large-Scale Human-Resource Semantic Modeling**

*基于大规模人力资源语义建模的多智能体推荐与优化框架*

[![CI](https://github.com/TsekaLuk/hr-multiagent-rec/actions/workflows/ci.yml/badge.svg)](https://github.com/TsekaLuk/hr-multiagent-rec/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![arXiv](https://img.shields.io/badge/arXiv-coming--soon-b31b1b.svg)](#)
[![Hugging Face Models](https://img.shields.io/badge/🤗-Qwen3--Embedding-yellow)](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

[Quickstart](#quickstart) · [Architecture](#architecture) · [Benchmarks](#benchmarks) · [Reproduce](#reproduce) · [Citation](#citation)

</div>

---

## TL;DR

A unified **person–job fit (PJF)** framework that combines (i) **same-family semantic encoding** with Qwen3-Embedding + Qwen3-Reranker, (ii) **bidirectional preference scoring** capturing both employer requirements and candidate preferences, and (iii) **lightweight multi-agent coordination** for evidence aggregation, ranking fusion and explanation. The full pipeline runs on a single **MacBook Air M4 (16 GB)** at < ¥5 in API cost.

> Why "same-family"? Existing PJF stacks glue heterogeneous models (BERT + GPT + custom rankers) whose embedding spaces diverge. We use the **Qwen3 series across encoding, reranking, and reasoning** so the representation space is intrinsically aligned — a property we ablate.

## Highlights

- **Modality-consistent semantic stack** — Qwen3-Embedding-0.6B (encoder) + Qwen3-Reranker-0.6B (cross-encoder) + DeepSeek-V4-Flash (agent reasoning, served by SiliconFlow). Built around a shared OpenAI-compatible API surface.
- **Bidirectional matching** — separate employer-side and candidate-side scoring functions, with *multiplicative hard-floor penalties* on education / experience and a *severe-salary-inversion negative-score* branch, fused by a convex combination. **Adds +9.4 pp nDCG@10** over pure semantic recall.
- **Async event-streaming multi-agent** — four specialized agents (Job-Analyst, Candidate-Analyst, Coordinator, Explainer) run under an async coordinator with parallel fan-out (`asyncio.Semaphore`) and **prompt-cache-aware PREFIX/TAIL** message layout. Validated end-to-end against a real SiliconFlow API (3/3 e2e tests, 189 s).
- **Strict TDD, no mocks** — 120 unit + 25 integration + 6 e2e tests; tests skip cleanly when models / API keys are unavailable rather than falling back to fakes.
- **Reproducible on a laptop** — runs end-to-end on a 16 GB MacBook Air M4 (Apple Silicon MPS for Embedder, CPU for Reranker to fit unified memory) with free SiliconFlow API for the agent LLM.

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
   Resumes &        │  ① Semantic Encoder  (Qwen3-Embedding-0.6B)     │
   Job Posts ─────► │     → FAISS-CPU IVF+PQ index                    │
                    └────────────────────┬────────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │  ② Bidirectional Scorer                          │
                    │     • Employer-side: skill ∩ exp ∩ hard rules   │
                    │     • Candidate-side: pay ∩ region ∩ pref       │
                    │     • Fusion: convex combination                 │
                    │  + Qwen3-Reranker-0.6B cross-encoder rerank      │
                    └────────────────────┬────────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────────┐
                    │  ③ Multi-Agent Coordinator  (CrewAI / hand-rolled) │
                    │     Job-Analyst ─┐                              │
                    │     Cand-Analyst ┼─► Coordinator ─► Explainer   │
                    │     LLM backend: Qwen3-8B / Gemini / DeepSeek   │
                    └────────────────────┬────────────────────────────┘
                                         ▼
                              Top-N + Evidence + Rationale
```

## Quickstart

```bash
# 1. Clone & install (Python 3.11 / 3.12)
git clone https://github.com/TsekaLuk/hr-multiagent-rec
cd hr-multiagent-rec
make dev

# 2. Provide API keys (free tiers are sufficient)
cp .env.example .env
$EDITOR .env       # paste SILICONFLOW_API_KEY (free Qwen3-8B or
                   # deepseek-ai/DeepSeek-V4-Flash works)

# 3. Prepare data + smoke demo (no model needed)
make data
make run-demo

# 4. Live async multi-agent demo (real API, ~10s)
make demo-async

# 5. Quick ablation slice (5 jobs, no Multi-Agent, ~10 min)
make eval-quick

# 6. Full ablation grid (≈40 min on M4 16GB)
make eval && make summary
```

## Benchmarks

Real numbers from a 30-job slice of the synthetic Tianchi-style corpus
(seed=42; 500 indexed resumes; full grid is in `outputs/ablation.csv`).

| Method                                  | P@10  | R@10  | nDCG@10   | MRR    |
|-----------------------------------------|------:|------:|----------:|-------:|
| TF-IDF (jieba)                          | 0.917 | 0.190 | 0.783     | 0.944  |
| BM25 (jieba)                            | 0.967 | 0.203 | 0.829     | 0.983  |
| Qwen3-Embedding-0.6B (no rerank)        | 0.930 | 0.192 | 0.795     | 0.967  |
| **+ Bidirectional scoring (this work)** | **0.947** | **0.198** | **0.889** | **1.000** |
| + Reranker + Multi-Agent                | _running_   | _running_   | _running_       | _running_    |

> **Update 2026-05-14:** 3/3 end-to-end multi-agent tests pass against the
> real SiliconFlow API using `deepseek-ai/DeepSeek-V4-Flash`. The async
> orchestrator (parallel fan-out + prompt-cache layout) is validated;
> the remaining ablation rows are running now and will be appended.

Adding our bidirectional scoring on top of pure semantic recall yields
**+9.4 percentage points of nDCG@10** and lifts MRR to 1.0 on this
slice — the largest single lift in the grid so far. Re-ranker and
multi-agent rows fill in once the corresponding models / API runs
complete.

## Reproduce

Full reproduction recipe — datasets, configs, seeds — is documented in [`docs/reproduce/README.md`](docs/reproduce/README.md). All experiments are seed-pinned and runnable on a single M-series Mac.

## Documentation

| Audience            | Start here                                                          |
|---------------------|---------------------------------------------------------------------|
| Graders / thesis defenders | [`docs/thesis_materials/`](docs/thesis_materials/) — 8股素材 |
| Open-source users   | [`docs/reproduce/`](docs/reproduce/README.md) → [`docs/api/`](docs/api/README.md) |
| Co-authors / future maintainers | [`docs/architecture/`](docs/architecture/README.md) → [`docs/architecture/decisions.md`](docs/architecture/decisions.md) (ADRs) → [`docs/architecture/agent_upgrade.md`](docs/architecture/agent_upgrade.md) |
| Paper readers       | [`paper/main.tex`](paper/main.tex) (arXiv-targeted) + [`paper/refs.bib`](paper/refs.bib) |
| Changelog           | [`CHANGELOG.md`](CHANGELOG.md)                                       |

## Citation

If you use this codebase or build on the framework, please cite:

```bibtex
@misc{lu2026hrmultiagent,
  title={A Multi-Agent Recommendation and Optimization Framework
         for Large-Scale Human-Resource Semantic Modeling},
  author={Lu, Zikai and Li, Xue},
  year={2026},
  eprint={arXiv:coming-soon},
  archivePrefix={arXiv},
  primaryClass={cs.IR}
}
```

## License

MIT © 2026 Lu Zikai. See [LICENSE](LICENSE).
