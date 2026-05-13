# Reproduce

A clean reproduction of every number in the [arXiv paper](../../paper/main.tex)
on an unmodified MacBook Air M4 (16 GB) running macOS 26.

## 1. Install

```bash
git clone https://github.com/TsekaLuk/hr-multiagent-rec
cd hr-multiagent-rec
make dev
```

`make dev` creates `.venv`, installs the package with all extras, and
wires up pre-commit hooks. Total: ~3–4 min, ~600 MB.

## 2. API keys

The full ablation grid uses one paid LLM provider. Free tiers are
sufficient for the standard configurations.

```bash
cp .env.example .env
# Pick at least one and fill it in:
#   SILICONFLOW_API_KEY=sk-...    (free Qwen3-8B — recommended)
#   GEMINI_API_KEY=AIza...        (free tier, 1500 req/day)
#   DEEPSEEK_API_KEY=sk-...       (pay-as-you-go, ¥0.14/M)
```

If no API key is set, the pipeline runs everything *except* the agent
layer; the `bm25`, `tfidf`, `semantic_only`, `semantic_plus_reranker`,
`semantic_plus_bidirectional` and `full_no_agent` rows still produce
numbers.

## 3. Data

```bash
make data   # builds the deterministic synthetic corpus
```

This generates `data/synthetic/{jobs,resumes,pairs}.json` —
300 jobs × 800 resumes × 24{,}145 ground-truth pairs (seed=42).

## 4. Models

The first integration run downloads:

| Model                       | Size  | Source           |
|-----------------------------|------:|------------------|
| Qwen/Qwen3-Embedding-0.6B   | ~1.2 GB | ModelScope or HF |
| Qwen/Qwen3-Reranker-0.6B    | ~1.2 GB | ModelScope or HF |

We default to ModelScope for users on Chinese networks (set
`HF_ENDPOINT=https://hf-mirror.com` or `MODELSCOPE_API_HUB=…` as
needed).

## 5. Run

```bash
make eval         # full grid (~25 min on M4)
make run-demo     # 10-second smoke demo, no models needed
```

Results land in:

```
outputs/ablation.csv         <- the canonical table
outputs/ablation.json        <- machine-readable copy
```

## 6. Compile the paper

```bash
make paper        # emits paper/main.pdf
```

## 7. Verifying reproducibility

Every random source is seeded:

```yaml
seed: 42                       # see configs/experiments.yaml
```

The synthetic corpus is deterministic — same seed produces byte-identical
JSON (we test for this in `tests/unit/test_synthetic.py::TestDeterminism`).
LLM agent outputs are *not* fully deterministic; numeric metrics are
reported as the mean over 3 runs, with std-dev in parentheses in the
paper appendix.

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `LocalEntryNotFoundError` for Qwen3-* | HF unreachable | `HF_ENDPOINT=https://hf-mirror.com` |
| `MPS placeholder storage` error | Old PyTorch | `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `LLMError: No API key` | Forgot to set provider key | `cp .env.example .env` and fill |
| `CrewAI ImportError` | Optional extra | `pip install -e ".[agents]"` |
