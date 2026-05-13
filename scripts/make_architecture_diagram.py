#!/usr/bin/env python3
"""Render the layered system architecture as a real publication-grade figure.

Replaces the ASCII art in README with a vector PDF + PNG suitable for
the thesis chapter 5 "system architecture" figure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LAYERS = [
    ("① Data Layer",         "Tianchi · Job-SDF · Synthetic · ESCO  →  Pydantic schemas",          "#DAE8FC"),
    ("② Semantic Encoding",  "Qwen3-Embedding-0.6B (MPS)  →  FAISS-CPU flat / IVF",               "#D5E8D4"),
    ("③ Matching",           "Qwen3-Reranker-0.6B (CPU yes/no)  +  Bidirectional scoring",         "#FFF2CC"),
    ("④ Multi-Agent",        "JobAnalyst · CandidateAnalyst · Coordinator · Explainer  (async)",   "#F8CECC"),
    ("⑤ Evaluation",         "P@K · R@K · nDCG@K · MRR  +  Cost / Cache Telemetry",                "#E1D5E7"),
]


def render(out_pdf: Path) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(LAYERS) + 1)
    ax.set_axis_off()

    for i, (title, body, color) in enumerate(reversed(LAYERS)):
        y = i + 0.6
        box = mpatches.FancyBboxPatch(
            (0.4, y), 9.2, 0.85,
            boxstyle="round,pad=0.02",
            linewidth=1.2, edgecolor="#333", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(0.7, y + 0.55, title, fontsize=11, fontweight="bold", va="center")
        ax.text(0.7, y + 0.22, body, fontsize=9, va="center", color="#333")
        if i < len(LAYERS) - 1:
            ax.annotate(
                "",
                xy=(5, y + 0.97), xytext=(5, y + 0.85 + 0.2),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#666"),
            )

    ax.set_title("HR-MultiAgent-Rec — Layered architecture", fontsize=12, pad=10)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=180)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(ROOT / "paper" / "figures" / "architecture.pdf"),
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out)
    print(f"wrote {out}")
    # Also write PNG sibling for README embedding
    png = out.with_suffix(".png")
    render(png)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
