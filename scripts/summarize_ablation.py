#!/usr/bin/env python3
"""Render the ablation results as a polished Markdown table.

Used to refresh the README / paper benchmark sections from `outputs/ablation.csv`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


PRETTY_NAMES = {
    "bm25": "BM25 (jieba)",
    "tfidf": "TF-IDF (jieba)",
    "semantic_only": "Qwen3-Embedding only",
    "semantic_plus_reranker": "  + Qwen3-Reranker",
    "semantic_plus_bidirectional": "  + Bidirectional scoring",
    "full_no_agent": "  + Reranker + Bidirectional",
    "full": "**Full (+ Multi-Agent)**",
    "full_mrl_dim_256": "  MRL dim=256",
    "full_mrl_dim_128": "  MRL dim=128",
    "full_gemini": "  LLM = Gemini-2.5-Flash",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "ablation.csv"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"# Ablation results not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("# (no experiments yet)")
        return

    cols = [("P@10", "P@10"), ("R@10", "R@10"), ("nDCG@10", "nDCG@10"), ("MRR", "MRR")]
    print("| Method | " + " | ".join(c[0] for c in cols) + " | sec |")
    print("|---|" + "|".join("---:" for _ in cols) + "|---:|")
    for r in rows:
        name = PRETTY_NAMES.get(r["experiment"], r["experiment"])
        cells = []
        for _, key in cols:
            v = r.get(key, "")
            try:
                cells.append(f"{float(v):.3f}")
            except ValueError:
                cells.append("—")
        cells.append(f"{float(r.get('seconds', 0) or 0):.1f}")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print()
    # Best-config callout
    best_ndcg = max(rows, key=lambda r: float(r.get("nDCG@10", 0) or 0))
    bn = PRETTY_NAMES.get(best_ndcg["experiment"], best_ndcg["experiment"])
    print(f"> **Best nDCG@10:** {bn} — {float(best_ndcg['nDCG@10']):.3f}.")


if __name__ == "__main__":
    main()
