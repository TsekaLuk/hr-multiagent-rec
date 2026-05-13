#!/usr/bin/env python3
"""Run the full ablation grid and write a CSV + JSON report.

This script is *idempotent* — it caches encoded vectors and reuses them
across experiments to keep the wall-clock under 30 minutes on M4 Air.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_rec.baselines import BM25Baseline, TfidfBaseline  # noqa: E402
from hr_rec.data.loaders import load_synthetic  # noqa: E402
from hr_rec.evaluation.metrics import (  # noqa: E402
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("experiments")


def _build_relevance(pairs):  # type: ignore[no-untyped-def]
    """Returns: dict[job_id] -> dict[resume_id] -> grade(1|2)."""
    rel: dict[str, dict[str, int]] = defaultdict(dict)
    for j, r, g in pairs:
        rel[j][r] = g
    return rel


def _evaluate(rankings, rel_map, ks):  # type: ignore[no-untyped-def]
    """Compute P@K, R@K, nDCG@K, MRR averaged over jobs."""
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


def _run_bm25(jobs, resumes, top_k):  # type: ignore[no-untyped-def]
    bm = BM25Baseline()
    bm.index(resumes)
    return {j.job_id: bm.match(j, top_k=top_k) for j in jobs}


def _run_tfidf(jobs, resumes, top_k):  # type: ignore[no-untyped-def]
    tf = TfidfBaseline()
    tf.index(resumes)
    return {j.job_id: tf.match(j, top_k=top_k) for j in jobs}


def _run_pipeline(jobs, resumes, cfg, llm_cfg, exp):  # type: ignore[no-untyped-def]
    """Lazy-import heavy pipeline so BM25/TF-IDF runs work without models."""
    from hr_rec.encoding.embedder import Embedder
    from hr_rec.matching.reranker import Reranker
    from hr_rec.pipeline import Pipeline, PipelineConfig

    embedder = Embedder()
    reranker = Reranker() if exp.get("use_reranker", True) else None

    orchestrator = None
    if exp.get("use_multi_agent", True):
        from hr_rec.agents.llm import LLM, LLMError
        from hr_rec.agents.orchestrator import Orchestrator

        provider = (exp.get("llm_override") or {}).get("provider") or llm_cfg["provider"]
        model = (exp.get("llm_override") or {}).get("model") or llm_cfg["model"]
        try:
            llm = LLM(model=model, provider=provider, temperature=llm_cfg.get("temperature", 0.2))
            orchestrator = Orchestrator(llm, explain_top_k=5)
        except LLMError as e:
            log.warning("LLM unavailable (%s) — running pipeline without agent layer", e)
            orchestrator = None

    pc = PipelineConfig(
        top_n_recall=cfg["data"]["top_n_recall"],
        top_m_agent=cfg["data"]["top_m_agent"],
        use_reranker=bool(reranker is not None),
        use_bidirectional=exp.get("use_bidirectional", True),
        use_multi_agent=bool(orchestrator is not None),
        dim=exp.get("dim"),
    )
    pipe = Pipeline(embedder, reranker=reranker, orchestrator=orchestrator, config=pc)
    pipe.index_resumes(resumes)
    return {j.job_id: pipe.match_one(j) for j in jobs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments.yaml"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "ablation.csv"))
    ap.add_argument("--max-jobs", type=int, default=None, help="Cap jobs for a quick smoke run")
    ap.add_argument(
        "--filter",
        default=None,
        help="Only run experiments whose name contains any of these (comma-separated) substrings",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["seed"]
    n_jobs = cfg["data"]["n_jobs"]
    n_resumes = cfg["data"]["n_resumes"]
    log.info("loading synthetic corpus  n_jobs=%d n_resumes=%d seed=%d", n_jobs, n_resumes, seed)
    jobs, resumes, pairs = load_synthetic(n_jobs, n_resumes, seed)
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    rel_map = _build_relevance(pairs)
    ks = cfg["data"]["eval_k"]
    top_k = max(ks)

    rows: list[dict] = []
    filters = [f.strip() for f in (args.filter or "").split(",") if f.strip()]
    for exp in cfg["experiments"]:
        if filters and not any(f in exp["name"] for f in filters):
            continue
        method = exp["method"]
        log.info("▶ running experiment: %s (method=%s)", exp["name"], method)
        t0 = time.time()
        if method == "bm25":
            rankings = _run_bm25(jobs, resumes, top_k)
        elif method == "tfidf":
            rankings = _run_tfidf(jobs, resumes, top_k)
        elif method == "pipeline":
            rankings = _run_pipeline(jobs, resumes, cfg, cfg["llm"], exp)
        else:
            log.error("unknown method %r — skipping", method)
            continue
        elapsed = time.time() - t0
        metrics = _evaluate(rankings, rel_map, ks)
        metrics["experiment"] = exp["name"]
        metrics["seconds"] = round(elapsed, 1)
        log.info(
            "  %s: P@10=%.3f R@10=%.3f nDCG@10=%.3f MRR=%.3f  (%.1fs)",
            exp["name"],
            metrics["P@10"],
            metrics["R@10"],
            metrics["nDCG@10"],
            metrics["MRR"],
            elapsed,
        )
        rows.append(metrics)

    if not rows:
        log.warning("no experiments ran")
        return

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "seconds", *[k for k in rows[0] if k not in ("experiment", "seconds")]]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    out_json = out_csv.with_suffix(".json")
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    log.info("wrote %s and %s", out_csv, out_json)


if __name__ == "__main__":
    main()
