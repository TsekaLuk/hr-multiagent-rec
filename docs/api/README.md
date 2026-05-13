# API Reference

Cliffsnotes for the public surface. For full signatures, see the docstrings
in each module.

## `hr_rec.data`

```python
from hr_rec.data.schemas import Resume, Job, Skill, SalaryRange, MatchScore
from hr_rec.data.synthetic import build_corpus, make_resume, make_job
from hr_rec.data.loaders import (
    load_synthetic, load_tianchi_jobs, load_tianchi_resumes, load_jobsdf_skills,
    dump_corpus_json, load_corpus_json,
)
```

## `hr_rec.encoding`

```python
from hr_rec.encoding.embedder import Embedder
from hr_rec.encoding.indexer import VectorIndex, SearchHit

emb = Embedder("Qwen/Qwen3-Embedding-0.6B")  # auto-detects MPS / CUDA / CPU
v  = emb.encode("Python 后端工程师")             # (D,) float32
vs = emb.encode_batch(["A", "B", "C"])           # (3, D)

idx = VectorIndex(dim=emb.dim, index_type="flat")
idx.add(vs, ["a", "b", "c"])
hits: list[list[SearchHit]] = idx.search(v.reshape(1, -1), top_k=5)
```

## `hr_rec.matching`

```python
from hr_rec.matching import bidirectional_score
from hr_rec.matching.reranker import Reranker

result = bidirectional_score(job, resume, alpha=0.6)
# -> BidirectionalScore(employer, candidate, fused, evidence)

# IMPORTANT: pin to CPU on 16GB M4 if Embedder is on MPS, or you'll OOM.
rk = Reranker("Qwen/Qwen3-Reranker-0.6B", device="cpu")
ranked: list[tuple[str, float]] = rk.rerank(
    query="JD 文本",
    candidates=[("rid_1", "resume text 1"), ("rid_2", "resume text 2")],
    top_k=10,
)
```

## `hr_rec.agents` — sync (legacy)

```python
from hr_rec.agents.llm import LLM
from hr_rec.agents.orchestrator import Orchestrator

llm = LLM(model="deepseek-ai/DeepSeek-V4-Flash", provider="siliconflow")
orch = Orchestrator(llm, explain_top_k=10)
result = orch.run(job, top_resumes, pre_ranked_match_scores)
# -> OrchestratorResult(final_ranking, explanations, ...)
```

## `hr_rec.agents` — async (recommended)

```python
import asyncio
from hr_rec.agents import AsyncLLM, AsyncOrchestrator, EventType

async def run_pjf(job, resumes, pre_ranked):
    async with AsyncLLM(
        model="deepseek-ai/DeepSeek-V4-Flash",
        provider="siliconflow",
        concurrency=8,                  # parallel Candidate-Analyst calls
        max_tokens=512,
    ) as llm:
        orch = AsyncOrchestrator(llm, explain_top_k=10, candidate_concurrency=8)

        # Option A: consume stream for live progress
        async for ev in orch.stream(job, resumes, pre_ranked):
            if ev.type == EventType.CANDIDATE_PROFILED and ev.usage:
                print(f"  {ev.payload['resume_id']}  cache_read={ev.usage.cache_read_tokens}")
        result = orch.last_result

        # Option B: just wait for the end
        # result = await orch.run(job, resumes, pre_ranked)

    print(result.final_ranking[0].resume_id, result.explanations)
    print(f"total {result.total_usage.calls} calls, "
          f"{result.total_usage.cache_read_tokens} cache_read tokens")
```

### Cache-friendly message layout (under the hood)

`AsyncLLM.chat(prefix, tail)` takes pre-split message lists so the
*prefix* is bit-identical across N concurrent calls. SiliconFlow,
DeepSeek and Anthropic all reward this with `cache_read_tokens` in
their `usage` response. See
[`docs/architecture/agent_upgrade.md`](../architecture/agent_upgrade.md)
for the protocol details.

## `hr_rec.pipeline`

```python
from hr_rec.pipeline import Pipeline, PipelineConfig

cfg = PipelineConfig(
    use_reranker=True,
    use_bidirectional=True,
    use_multi_agent=True,
    alpha=0.6,
    top_n_recall=50,
    top_m_agent=15,
)
pipe = Pipeline(embedder, reranker=reranker, orchestrator=orch, config=cfg)
pipe.index_resumes(resumes)
top: list[MatchScore] = pipe.match_one(job)
```

## `hr_rec.evaluation`

```python
from hr_rec.evaluation import precision_at_k, recall_at_k, ndcg_at_k, mean_reciprocal_rank
```

## `hr_rec.baselines`

```python
from hr_rec.baselines import BM25Baseline, TfidfBaseline

bm = BM25Baseline()
bm.index(resumes)
top: list[MatchScore] = bm.match(job, top_k=50)
```
