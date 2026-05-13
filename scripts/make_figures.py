#!/usr/bin/env python3
"""Generate paper figures from the ablation CSV.

Outputs (under paper/figures/ unless overridden):
    architecture.pdf            — vector architecture stack
    state_machine.pdf           — Coordinator state machine
    ablation_bars.pdf           — per-config nDCG@10 bar chart
    topk_curves.pdf             — nDCG@K curves (K ∈ {1, 3, 5, 10, 20})
    cost_vs_quality.pdf         — Pareto: nDCG@10 vs latency, colored by cost
    per_agent_timing.pdf        — bar chart of Multi-Agent stage breakdown (if telemetry available)
    ablation.tex                — booktabs LaTeX table (re-emitted)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


PRETTY = {
    "bm25": "BM25",
    "tfidf": "TF-IDF",
    "semantic_only": "Qwen3-Emb only",
    "semantic_plus_reranker": "+ Reranker",
    "semantic_plus_bidirectional": "+ Bidirectional",
    "full_no_agent": "+ Reranker + Bidirectional",
    "full_multiagent": "+ Multi-Agent (full)",
    "full": "+ Multi-Agent (full)",
}

COLORS = {
    "bm25": "#999999",
    "tfidf": "#BBBBBB",
    "semantic_only": "#A6CEE3",
    "semantic_plus_bidirectional": "#1F78B4",
    "semantic_plus_reranker": "#33A02C",
    "full_no_agent": "#FF7F00",
    "full_multiagent": "#E31A1C",
    "full": "#E31A1C",
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _fnum(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _bars(rows: list[dict[str, str]], out: Path) -> None:
    rows = [r for r in rows if _fnum(r.get("nDCG@10"))]
    rows.sort(key=lambda r: _fnum(r["nDCG@10"]) or 0)
    labels = [PRETTY.get(r["experiment"], r["experiment"]) for r in rows]
    vals = [_fnum(r["nDCG@10"]) or 0 for r in rows]
    colors = [COLORS.get(r["experiment"], "#4C72B0") for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.barh(labels, vals, color=colors, edgecolor="#222", linewidth=0.4)
    for b, v in zip(bars, vals, strict=True):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlabel("nDCG@10 (higher is better)", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.18 if vals else 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Ablation — nDCG@10 by configuration", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _curves(rows: list[dict[str, str]], out: Path, ks: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for r in rows:
        if not _fnum(r.get("nDCG@10")):
            continue
        ys = [_fnum(r.get(f"nDCG@{k}")) or 0 for k in ks]
        ax.plot(
            ks, ys,
            marker="o", linewidth=1.6,
            label=PRETTY.get(r["experiment"], r["experiment"]),
            color=COLORS.get(r["experiment"]),
        )
    ax.set_xlabel("K")
    ax.set_ylabel("nDCG@K")
    ax.set_xticks(ks)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_title("nDCG@K curves", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _cost_quality(rows: list[dict[str, str]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in rows:
        nd = _fnum(r.get("nDCG@10"))
        sec = _fnum(r.get("seconds"))
        if nd is None or sec is None:
            continue
        cost = _fnum(r.get("cost_usd")) or 0.0
        size = 80 + cost * 50_000  # scale dot by $ (free → 80)
        color = COLORS.get(r["experiment"], "#4C72B0")
        ax.scatter(sec, nd, s=size, color=color, edgecolor="#222",
                   linewidth=0.6, alpha=0.85)
        ax.annotate(
            PRETTY.get(r["experiment"], r["experiment"]),
            xy=(sec, nd), xytext=(6, 6), textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Wall-clock seconds (lower is better)")
    ax.set_ylabel("nDCG@10 (higher is better)")
    ax.set_xscale("log")
    ax.set_title("Cost vs. Quality (dot size ∝ USD cost)", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _per_agent_timing(rows: list[dict[str, str]], out: Path) -> None:
    """For Multi-Agent rows, plot in / out / cache_read tokens stacked."""
    rows = [r for r in rows if r.get("llm_calls")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    labels = [PRETTY.get(r["experiment"], r["experiment"]) for r in rows]
    in_ = [_fnum(r.get("input_tokens")) or 0 for r in rows]
    out_ = [_fnum(r.get("output_tokens")) or 0 for r in rows]
    cache = [_fnum(r.get("cache_read_tokens")) or 0 for r in rows]
    x = list(range(len(labels)))
    ax.bar(x, in_, label="input", color="#A6CEE3")
    ax.bar(x, out_, bottom=in_, label="output", color="#1F78B4")
    bottoms = [a + b for a, b in zip(in_, out_, strict=True)]
    ax.bar(x, cache, bottom=bottoms, label="cache_read", color="#33A02C")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Tokens (stacked)")
    ax.set_title("Multi-Agent token usage (in + out + cache_read)", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _table(rows: list[dict[str, str]], out: Path) -> None:
    cols = ["P@10", "R@10", "nDCG@10", "MRR"]
    lines = []
    lines.append("\\begin{tabular}{lcccc}\n\\toprule")
    lines.append("Configuration & " + " & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        cells = [PRETTY.get(r["experiment"], r["experiment"]).replace("_", r"\_")]
        for c in cols:
            v = r.get(c, "—")
            cells.append(f"{float(v):.3f}" if v not in ("", "—") else "—")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule\n\\end{tabular}")
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "ablation.csv"))
    ap.add_argument("--figures-dir", default=str(ROOT / "paper" / "figures"))
    ap.add_argument("--tables-dir", default=str(ROOT / "paper" / "tables"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ablation CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = Path(args.tables_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_csv(csv_path)

    _bars(rows, fig_dir / "ablation_bars.pdf")
    print(f"wrote {fig_dir / 'ablation_bars.pdf'}")
    _curves(rows, fig_dir / "topk_curves.pdf", ks=[1, 3, 5, 10, 20])
    print(f"wrote {fig_dir / 'topk_curves.pdf'}")
    _cost_quality(rows, fig_dir / "cost_vs_quality.pdf")
    print(f"wrote {fig_dir / 'cost_vs_quality.pdf'}")
    _per_agent_timing(rows, fig_dir / "per_agent_timing.pdf")
    print(f"wrote {fig_dir / 'per_agent_timing.pdf'}")
    _table(rows, tab_dir / "ablation.tex")
    print(f"wrote {tab_dir / 'ablation.tex'}")


if __name__ == "__main__":
    main()
