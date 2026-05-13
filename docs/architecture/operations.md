# Operations guide

Running this project in practice on a 16 GB MacBook Air M4. The
patterns below are what we *actually* found work; deviations have a
documented failure mode.

## Memory budget on M4 16GB

| Process                              | Footprint | Notes                                          |
|--------------------------------------|----------:|------------------------------------------------|
| macOS + Slack / browser baseline     |    ~6 GB  | The harsh truth — only ~10 GB usable for ML.   |
| Qwen3-Embedding-0.6B on MPS          |  ~1.4 GB  | Stays loaded during corpus index + retrieval.  |
| Qwen3-Reranker-0.6B on MPS           |  ~1.5 GB  | **Co-loading on MPS exceeds budget → OOM kill**|
| Qwen3-Reranker-0.6B on CPU           |  ~1.5 GB  | Recommended path; ~3× slower than MPS.         |
| FAISS index (500 resumes × 1024 dim) |  <0.5 GB  | Effectively free.                              |
| sentence-transformers framework      |  ~0.5 GB  | Cached after first import.                     |
| Active Python heap                   |  ~0.5 GB  | Adds up across DataFrames / lists.             |

**Default in `scripts/run_experiments.py`**: Embedder on auto-detect
(MPS), Reranker pinned to CPU. This is the only configuration we have
seen complete a multi-job ablation grid without an OOM kill on 16 GB.

## Avoiding stale background processes

Several iterations of debugging produced multiple concurrent
experiment processes (each consuming ~1.5 GB), which compounded the
memory pressure. To clean them up:

```bash
# List active runs
ps aux | grep run_experiments | grep -v grep

# Kill duplicates by PID
kill -9 <PID> <PID> ...
```

Future fix: use `pid_file=outputs/run.pid` semantics in
`run_experiments.py` to refuse a second invocation while one is alive.

## Resuming a partial ablation

`run_experiments.py` currently overwrites `outputs/ablation.csv`. To
preserve a result we backed up:

```bash
cp outputs/ablation.csv outputs/ablation_<descriptor>.csv
cp outputs/ablation.json outputs/ablation_<descriptor>.json
```

Then the next run starts fresh. We have not yet implemented an
incremental `--append` mode; the workflow is "snapshot, then rerun".

## Real-API smoke checklist

When validating against SiliconFlow / DeepSeek / OpenRouter:

```bash
# 1. Verify the key reaches the provider
.venv/bin/python -c "
import os
from openai import OpenAI
c = OpenAI(
    api_key=os.environ['SILICONFLOW_API_KEY'],
    base_url='https://api.siliconflow.cn/v1',
)
r = c.chat.completions.create(
    model='deepseek-ai/DeepSeek-V4-Flash',
    messages=[{'role':'user', 'content':'reply OK'}],
    max_tokens=10,
)
print(r.choices[0].message.content, r.usage)
"

# 2. Run the async e2e suite (3 tests)
set -a && source .env && set +a
.venv/bin/pytest tests/e2e/test_async_orchestrator_real_llm.py -v --timeout=180
```

If both pass, the multi-agent layer is healthy.

## Cache-friendly call pattern (for any new agent)

Use `AsyncLLM.chat(prefix, tail)` — **never** concatenate everything
into a single `chat([msg1, msg2, ...])` call when you're about to
make N parallel calls that share most context. The split is what
makes the provider's cache key stable across calls 2..N.

See `hr_rec.agents.async_orchestrator._cand_prefix` for the
production example.
