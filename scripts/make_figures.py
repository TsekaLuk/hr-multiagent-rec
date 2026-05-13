#!/usr/bin/env python3
"""Generate paper figures from the ablation CSV.

Outputs:
    paper/figures/arch.pdf              — system architecture (matplotlib)
    paper/figures/ablation_bars.pdf     — per-config nDCG@10 bar chart
    paper/figures/topk_curves.pdf       — nDCG@K curves
    paper/tables/ablation.tex           — booktabs LaTeX table
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _bars(rows: list[dict[str, str]], out: Path) -> None:
    rows = [r for r in rows if "nDCG@10" in r and r["nDCG@10"]]
    labels = [r["experiment"] for r in rows]
    vals = [float(r["nDCG@10"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(labels, vals, color="#4C72B0")
    for b, v in zip(bars, vals, strict=True):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8)
    ax.set_xlabel("nDCG@10")
    ax.set_xlim(0, max(vals) * 1.18 if vals else 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def _curves(rows: list[dict[str, str]], out: Path, ks: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for r in rows:
        ys = [float(r.get(f"nDCG@{k}", 0) or 0) for k in ks]
        ax.plot(ks, ys, marker="o", label=r["experiment"])
    ax.set_xlabel("K")
    ax.set_ylabel("nDCG@K")
    ax.set_xticks(ks)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def _table(rows: list[dict[str, str]], out: Path) -> None:
    cols = ["P@10", "R@10", "nDCG@10", "MRR"]
    lines: list[str] = []
    lines.append("\\begin{tabular}{lcccc}\n\\toprule")
    lines.append("Configuration & " + " & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        cells = [r["experiment"].replace("_", r"\_")]
        for c in cols:
            v = r.get(c, "—")
            cells.append(f"{float(v):.3f}" if v not in ("", "—") else "—")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule\n\\end{tabular}")
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "ablation.csv"))
    ap.add_argument(
        "--figures-dir",
        default=str(ROOT / "paper" / "figures"),
    )
    ap.add_argument(
        "--tables-dir",
        default=str(ROOT / "paper" / "tables"),
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ablation CSV not found at {csv_path}", file=sys.stderr)
        print("run `make eval` first", file=sys.stderr)
        sys.exit(1)

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = Path(args.tables_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_csv(csv_path)
    _bars(rows, fig_dir / "ablation_bars.pdf")
    _curves(rows, fig_dir / "topk_curves.pdf", ks=[1, 3, 5, 10, 20])
    _table(rows, tab_dir / "ablation.tex")
    print(f"wrote {fig_dir / 'ablation_bars.pdf'}")
    print(f"wrote {fig_dir / 'topk_curves.pdf'}")
    print(f"wrote {tab_dir / 'ablation.tex'}")


if __name__ == "__main__":
    main()
