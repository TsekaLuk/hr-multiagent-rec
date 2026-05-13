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

emb = Embedder("Qwen/Qwen3-Embedding-0.6B")
v  = emb.encode("Python 后端工程师")          # (D,)
vs = emb.encode_batch(["A", "B", "C"])         # (3, D)

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

rk = Reranker("Qwen/Qwen3-Reranker-0.6B")
ranked = rk.rerank(query="JD 文本", candidates=[("rid", "resume text"), ...])
```

## `hr_rec.agents`

```python
from hr_rec.agents.llm import LLM
from hr_rec.agents.orchestrator import Orchestrator

llm = LLM(model="Qwen/Qwen3-8B", provider="siliconflow")
orch = Orchestrator(llm, explain_top_k=10)
result = orch.run(job, top_resumes, pre_ranked_match_scores)
# -> OrchestratorResult(final_ranking, explanations, ...)
```

## `hr_rec.pipeline`

```python
from hr_rec.pipeline import Pipeline, PipelineConfig

cfg = PipelineConfig(use_reranker=True, use_bidirectional=True, use_multi_agent=True)
pipe = Pipeline(embedder, reranker=reranker, orchestrator=orch, config=cfg)
pipe.index_resumes(resumes)
top = pipe.match_one(job)   # -> list[MatchScore]
```

## `hr_rec.evaluation`

```python
from hr_rec.evaluation import precision_at_k, recall_at_k, ndcg_at_k, mean_reciprocal_rank
```
