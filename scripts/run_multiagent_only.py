#!/usr/bin/env python3
"""Run the Multi-Agent ablation slice in isolation.

Why separate? On a 16GB M4 the Embedder + Reranker pair already
saturates MPS / unified memory. The Multi-Agent stage only needs the
Embedder (for the recall stage) plus the SiliconFlow / Qwen LLM API —
no Reranker. By isolating this experiment we (a) avoid OOM and (b)
get a clean cost / cache-read measurement per agent.

Outputs ``outputs/ablation_multiagent.csv`` and prints a summary table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_rec.data.loaders import load_synthetic  # noqa: E402
from hr_rec.evaluation.metrics import (  # noqa: E402
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _build_relevance(pairs):  # type: ignore[no-untyped-def]
    rel: dict[str, dict[str, int]] = defaultdict(dict)
    for j, r, g in pairs:
        rel[j][r] = g
    return rel


def _evaluate(rankings, rel_map, ks):  # type: ignore[no-untyped-def]
    p_at: dict[int, list[float]] = {k: [] for k in ks}
    r_at: dict[int, list[float]] = {k: [] for k in ks}
    n_at: dict[int, list[float]] = {k: [] for k in ks}
    rels_per_q: list[list[str]] = []
    rankings_per_q: list[list[str]] = []
    for job_id, ranked in rankings.items():
        relmap = rel_map.get(job_id, {})
        rel_ids = set(relmap.keys())
        ids = [m.resume_id for m in ranked]
        for k in ks:
            p_at[k].append(precision_at_k(ids, rel_ids, k))
            r_at[k].append(recall_at_k(ids, rel_ids, k))
            n_at[k].append(ndcg_at_k(ids, relmap, k))
        rels_per_q.append(list(rel_ids))
        rankings_per_q.append(ids)
    mrr = mean_reciprocal_rank(rankings_per_q, rels_per_q)
    return {
        **{f"P@{k}": (sum(v) / len(v) if v else 0.0) for k, v in p_at.items()},
        **{f"R@{k}": (sum(v) / len(v) if v else 0.0) for k, v in r_at.items()},
        **{f"nDCG@{k}": (sum(v) / len(v) if v else 0.0) for k, v in n_at.items()},
        "MRR": mrr,
    }


def main() -> None:
    import asyncio

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=5)
    ap.add_argument("--n-resumes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-m", type=int, default=10, help="candidates fed to Multi-Agent")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--explain-top-k", type=int, default=3)
    ap.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "ablation_multiagent.csv"),
    )
    args = ap.parse_args()

    if not os.environ.get("SILICONFLOW_API_KEY"):
        print("SILICONFLOW_API_KEY not set — aborting.", file=sys.stderr)
        sys.exit(1)

    from hr_rec.agents.async_orchestrator import AsyncOrchestrator
    from hr_rec.agents.events import EventType
    from hr_rec.agents.llm_async import AsyncLLM
    from hr_rec.encoding.embedder import Embedder
    from hr_rec.pipeline import Pipeline, PipelineConfig

    print(f"Loading corpus  jobs={args.n_jobs} resumes={args.n_resumes} seed={args.seed}")
    jobs, resumes, pairs = load_synthetic(args.n_jobs * 20, args.n_resumes, args.seed)
    jobs = jobs[: args.n_jobs]
    rel_map = _build_relevance(pairs)

    embedder = Embedder()
    pipe = Pipeline(
        embedder,
        reranker=None,
        orchestrator=None,
        config=PipelineConfig(
            top_n_recall=30, top_m_agent=args.top_m,
            use_reranker=False, use_bidirectional=True,
            use_multi_agent=False,
        ),
    )
    pipe.index_resumes(resumes)

    async def run_multiagent() -> tuple[dict, dict]:
        async with AsyncLLM(
            provider="siliconflow",
            concurrency=args.concurrency,
            max_tokens=512,
        ) as llm:
            orch = AsyncOrchestrator(
                llm,
                explain_top_k=args.explain_top_k,
                candidate_concurrency=args.concurrency,
            )

            rankings: dict[str, list] = {}
            agg_usage = {"input": 0, "output": 0, "cache_read": 0, "calls": 0, "seconds": 0.0}
            t0 = time.time()
            for i, job in enumerate(jobs):
                # Stage 1: vector recall + bidirectional (no Reranker — saves OOM).
                base = pipe.match_one(job)
                # Stage 2: feed top-M into the Multi-Agent orchestrator.
                top_m = base[: args.top_m]
                rd = {r.resume_id: r for r in resumes}
                top_resumes = [rd[ms.resume_id] for ms in top_m]
                print(f"  job {i+1}/{len(jobs)}: {job.title[:30]:30s}", end=" ", flush=True)
                tj = time.time()
                async for ev in orch.stream(job, top_resumes, top_m):
                    if ev.type == EventType.FINAL and ev.usage:
                        agg_usage["input"] += ev.usage.input_tokens
                        agg_usage["output"] += ev.usage.output_tokens
                        agg_usage["cache_read"] += ev.usage.cache_read_tokens
                        agg_usage["calls"] += ev.usage.calls
                        agg_usage["seconds"] += ev.usage.seconds
                result = orch.last_result
                assert result is not None
                # Splice agent-adjusted top-M back, keep tail unchanged.
                full = list(result.final_ranking) + base[args.top_m :]
                rankings[job.job_id] = full
                print(f"  {time.time() - tj:.1f}s  agent_calls={ev.usage.calls if ev.usage else 0}")
            agg_usage["wall_seconds"] = time.time() - t0
            return rankings, agg_usage

    rankings, agg_usage = asyncio.run(run_multiagent())
    metrics = _evaluate(rankings, rel_map, ks=[1, 3, 5, 10, 20])
    metrics["experiment"] = "full_multiagent"
    metrics["seconds"] = round(agg_usage["wall_seconds"], 1)

    print()
    print(f"  P@10 = {metrics['P@10']:.3f}")
    print(f"  R@10 = {metrics['R@10']:.3f}")
    print(f"  nDCG@10 = {metrics['nDCG@10']:.3f}")
    print(f"  MRR    = {metrics['MRR']:.3f}")
    print(f"  total agent calls: {agg_usage['calls']}")
    print(f"  total input tokens: {agg_usage['input']}")
    print(f"  total output tokens: {agg_usage['output']}")
    print(f"  total cache_read tokens: {agg_usage['cache_read']}")
    if agg_usage["input"]:
        print(f"  cache hit rate: {agg_usage['cache_read'] / agg_usage['input']:.1%}")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()) + list(agg_usage.keys()))
        w.writeheader()
        w.writerow({**metrics, **agg_usage})
    Path(str(out_p).replace(".csv", ".json")).write_text(
        json.dumps({**metrics, **agg_usage}, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
