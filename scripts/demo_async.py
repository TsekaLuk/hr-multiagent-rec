#!/usr/bin/env python3
"""Live event-stream demo of the async multi-agent orchestrator.

Run with:

    set -a && source .env && set +a
    python scripts/demo_async.py

Demonstrates:
* AsyncOrchestrator.stream() yielding events as agents progress
* Parallel candidate-analyst fan-out
* Cache-friendly PREFIX/TAIL message layout
* Per-agent + total usage accounting
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hr_rec.agents.async_orchestrator import AsyncOrchestrator  # noqa: E402
from hr_rec.agents.events import EventType  # noqa: E402
from hr_rec.agents.llm_async import AsyncLLM  # noqa: E402
from hr_rec.baselines import BM25Baseline  # noqa: E402
from hr_rec.data.loaders import load_synthetic  # noqa: E402

logging.basicConfig(level=logging.WARNING)


async def main() -> None:
    print("Loading synthetic corpus…")
    jobs, resumes, _ = load_synthetic(n_jobs=10, n_resumes=80, seed=2026)
    rd = {r.resume_id: r for r in resumes}

    # Pre-rank with BM25 to get a non-trivial pre_ranked input.
    bm = BM25Baseline()
    bm.index(resumes)
    job = jobs[0]
    pre_ranked = bm.match(job, top_k=10)
    top_resumes = [rd[ms.resume_id] for ms in pre_ranked]

    print(f"\nJob: {job.title} @ {job.company} | {job.location}")
    print(f"  required: {[s.name for s in job.required_skills]}")
    print(f"  pre-ranked top-10 candidates (BM25)\n")

    async with AsyncLLM(
        model="deepseek-ai/DeepSeek-V4-Flash",
        provider="siliconflow",
        concurrency=4,
        max_tokens=512,
    ) as llm:
        orch = AsyncOrchestrator(llm, explain_top_k=3, candidate_concurrency=4)
        async for ev in orch.stream(job, top_resumes, pre_ranked):
            if ev.type == EventType.AGENT_START:
                print(f"  ▶ {ev.agent} starting…")
            elif ev.type == EventType.AGENT_END:
                print(f"  ✓ {ev.agent} done  {ev.payload}")
            elif ev.type == EventType.CANDIDATE_PROFILED:
                cache = ev.usage.cache_read_tokens if ev.usage else 0
                rid = ev.payload.get("resume_id", "?")
                tag = "(warm-up)" if ev.payload.get("cache_warming") else ""
                print(f"    • profiled {rid:>14}  in={ev.usage.input_tokens if ev.usage else 0:5d} "
                      f"out={ev.usage.output_tokens if ev.usage else 0:4d} cache_read={cache:5d}  {tag}")
            elif ev.type == EventType.FINAL:
                u = ev.usage
                print("\n" + "=" * 60)
                print(f"TOTAL calls: {u.calls}  in={u.input_tokens}  out={u.output_tokens}")
                if u.cache_read_tokens:
                    rate = u.cache_read_tokens / max(1, u.input_tokens) * 100
                    print(f"      cache_read: {u.cache_read_tokens} tokens ({rate:.1f}% of input)")
                print(f"      wall-clock: {u.seconds:.1f}s")

        result = orch.last_result
        assert result is not None
        print("\nTop-3 final ranking with rationale:")
        print("-" * 60)
        for ms in result.final_ranking[:3]:
            r = rd[ms.resume_id]
            ex = result.explanations.get(ms.resume_id, "(no explanation)")
            print(f"  {ms.resume_id:>14} score={ms.fused_score:.3f}  skills={[s.name for s in r.skills][:4]}")
            print(f"                 {ex}\n")


if __name__ == "__main__":
    asyncio.run(main())
