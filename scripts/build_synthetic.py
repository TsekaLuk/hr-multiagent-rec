#!/usr/bin/env python3
"""Build deterministic synthetic Tianchi-style corpus and write to data/synthetic/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hr_rec.data.loaders import dump_corpus_json, load_synthetic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=300)
    ap.add_argument("--n-resumes", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic",
    )
    args = ap.parse_args()

    print(f"Generating {args.n_jobs} jobs × {args.n_resumes} resumes (seed={args.seed})…")
    jobs, resumes, pairs = load_synthetic(args.n_jobs, args.n_resumes, args.seed)
    print(f"Ground-truth positive pairs: {len(pairs)}")
    dump_corpus_json(jobs, resumes, pairs, args.out)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
