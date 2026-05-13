#!/usr/bin/env python3
"""End-to-end smoke demo using BM25 baseline (no model required).

Runs against the synthetic corpus and prints a top-5 list with
bidirectional-score evidence for one job.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hr_rec.baselines import BM25Baseline  # noqa: E402
from hr_rec.data.loaders import load_synthetic  # noqa: E402
from hr_rec.matching.scoring import bidirectional_score  # noqa: E402


def main() -> None:
    print("Loading synthetic corpus…")
    jobs, resumes, _ = load_synthetic(n_jobs=30, n_resumes=120, seed=2026)
    rd = {r.resume_id: r for r in resumes}
    bm = BM25Baseline()
    bm.index(resumes)

    job = jobs[0]
    print(f"\nJob: {job.title} @ {job.company} | {job.location}")
    print(f"Required: {[s.name for s in job.required_skills]}")
    print(f"Preferred: {[s.name for s in job.preferred_skills]}")

    hits = bm.match(job, top_k=5)
    print("\nTop-5 by BM25 + bidirectional scoring:")
    print("-" * 80)
    for h in hits:
        r = rd[h.resume_id]
        bi = bidirectional_score(job, r)
        print(f"{h.resume_id:>12}  bm25={h.fused_score:.3f}  emp={bi.employer:.2f}  cand={bi.candidate:.2f}  fused={bi.fused:.2f}")
        print(f"             skills={[s.name for s in r.skills]} loc={r.location} exp={r.experience_level.value}")
        print(f"             {bi.evidence.rationale if bi.evidence else ''}")
        print()


if __name__ == "__main__":
    main()
