#!/usr/bin/env python3
"""Render the Coordinator state machine to PDF/SVG (and the canonical Mermaid source).

Used to embed a deterministic state-machine figure in the thesis +
arXiv paper (replaces hand-drawn diagrams which rot when phases
change). The Mermaid source is also written so the reader can copy it
into Typora/Notion.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hr_rec.agents.state import PhaseKind, TransitionReason  # noqa: E402


HAPPY_PATH = [
    (PhaseKind.INIT,                TransitionReason.START,            PhaseKind.JOB_ANALYSIS),
    (PhaseKind.JOB_ANALYSIS,        TransitionReason.JOB_ANALYSED,     PhaseKind.CANDIDATE_FAN_OUT),
    (PhaseKind.CANDIDATE_FAN_OUT,   TransitionReason.CANDIDATES_DONE,  PhaseKind.COORDINATION),
    (PhaseKind.COORDINATION,        TransitionReason.COORD_DONE,       PhaseKind.EXPLANATION),
    (PhaseKind.EXPLANATION,         TransitionReason.EXPLAINED,        PhaseKind.TERMINAL),
]

RECOVERY_EDGES = [
    (PhaseKind.JOB_ANALYSIS,      TransitionReason.JSON_PARSE_RETRY),
    (PhaseKind.CANDIDATE_FAN_OUT, TransitionReason.JSON_PARSE_RETRY),
    (PhaseKind.CANDIDATE_FAN_OUT, TransitionReason.MAX_TOKENS_RETRY),
    (PhaseKind.CANDIDATE_FAN_OUT, TransitionReason.CTX_OVERFLOW_SHRINK),
    (PhaseKind.EXPLANATION,       TransitionReason.JSON_PARSE_RETRY),
]

ERROR_EDGES = [
    (PhaseKind.JOB_ANALYSIS,      TransitionReason.CIRCUIT_BREAKER_OPEN, PhaseKind.TERMINAL),
    (PhaseKind.CANDIDATE_FAN_OUT, TransitionReason.USER_CANCEL,          PhaseKind.TERMINAL),
    (PhaseKind.CANDIDATE_FAN_OUT, TransitionReason.PROVIDER_ERROR,       PhaseKind.TERMINAL),
]


def render_mermaid() -> str:
    lines = [
        "stateDiagram-v2",
        "    [*] --> init",
    ]
    for src, reason, dst in HAPPY_PATH:
        lines.append(f"    {src.value} --> {dst.value}: {reason.value}")
    for phase, reason in RECOVERY_EDGES:
        lines.append(f"    {phase.value} --> {phase.value}: {reason.value}")
    for src, reason, dst in ERROR_EDGES:
        lines.append(f"    {src.value} --> {dst.value}: {reason.value}")
    lines.append("    terminal --> [*]")
    return "\n".join(lines)


def render_matplotlib(out: Path) -> None:
    """Render a programmatic state-machine figure via matplotlib."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_axis_off()

    # Phase nodes arranged horizontally
    phases = list(HAPPY_PATH)
    nodes = [HAPPY_PATH[0][0], *[t[2] for t in HAPPY_PATH]]
    positions = {p: (1.0 + 2.2 * i, 4) for i, p in enumerate(nodes)}

    for phase, (x, y) in positions.items():
        color = "#FFE6CC" if phase != PhaseKind.TERMINAL else "#D5E8D4"
        fancy = mpatches.FancyBboxPatch(
            (x - 0.9, y - 0.35), 1.8, 0.7,
            boxstyle="round,pad=0.02",
            linewidth=1.2, edgecolor="#333", facecolor=color,
        )
        ax.add_patch(fancy)
        ax.text(x, y, phase.value, ha="center", va="center", fontsize=9)

    # Happy-path edges
    for src, reason, dst in HAPPY_PATH:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        ax.annotate(
            "",
            xy=(dx - 0.95, dy), xytext=(sx + 0.95, sy),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"),
        )
        ax.text((sx + dx) / 2, sy + 0.35, reason.value,
                ha="center", fontsize=7, color="#333")

    # Recovery self-loops (drawn above each phase)
    for phase, reason in RECOVERY_EDGES:
        x, y = positions[phase]
        # tiny self-loop arc
        loop = mpatches.FancyArrowPatch(
            (x - 0.2, y + 0.45), (x + 0.2, y + 0.45),
            connectionstyle="arc3,rad=-1.4",
            arrowstyle="->", lw=1.0, color="#888",
        )
        ax.add_patch(loop)

    # Error edges (drawn below, dashed)
    for src, reason, dst in ERROR_EDGES:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        ax.annotate(
            "",
            xy=(dx - 0.7, dy - 1.5), xytext=(sx, sy - 0.4),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#B85450", linestyle="--"),
        )

    # Legend
    happy = mpatches.Patch(color="#FFE6CC", label="Phase")
    term = mpatches.Patch(color="#D5E8D4", label="Terminal")
    recovery = mpatches.FancyArrowPatch(
        (0, 0), (1, 0),
        arrowstyle="->", color="#888", lw=1.0,
    )
    error = mpatches.FancyArrowPatch(
        (0, 0), (1, 0),
        arrowstyle="->", color="#B85450", lw=1.0, linestyle="--",
    )
    ax.legend(
        handles=[happy, term, recovery, error],
        labels=["Phase", "Terminal", "Recovery (self-loop)", "Error → terminal"],
        loc="lower center", ncol=4, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    ax.set_title("Multi-Agent Coordinator state machine", fontsize=11, pad=6)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mmd-out",
        default=str(ROOT / "paper" / "figures" / "state_machine.mmd"),
    )
    ap.add_argument(
        "--pdf-out",
        default=str(ROOT / "paper" / "figures" / "state_machine.pdf"),
    )
    args = ap.parse_args()

    mmd_path = Path(args.mmd_out)
    pdf_path = Path(args.pdf_out)
    mmd_path.parent.mkdir(parents=True, exist_ok=True)

    mmd_path.write_text(render_mermaid(), encoding="utf-8")
    print(f"wrote {mmd_path}")

    try:
        render_matplotlib(pdf_path)
        print(f"wrote {pdf_path}")
    except Exception as e:
        print(f"matplotlib render failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
