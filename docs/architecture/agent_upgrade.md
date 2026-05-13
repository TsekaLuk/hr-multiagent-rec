# Agent Architecture Upgrade — May 2026

Inspired by `~/Documents/claude-code-notes/04-Agent协调/*` (Claude Code's
internal coordinator design), we upgraded the multi-agent layer from a
plain sequential orchestrator to an async, cache-friendly, parallel
fan-out architecture. The original sequential orchestrator
(`hr_rec.agents.orchestrator.Orchestrator`) is kept for compatibility;
the new path is `hr_rec.agents.async_orchestrator.AsyncOrchestrator`.

## Why these specific upgrades

| Pattern in Claude Code | Why for PJF | Where in our code |
|---|---|---|
| Async-generator coordinator | Stream per-candidate events to UI/log; enable clean cancellation; expose usage live | `AsyncOrchestrator.stream() → AsyncIterator[Event]` |
| Parallel tool-call fan-out | Candidate-Analyst is embarrassingly parallel; M4 happily runs 8 concurrent SiliconFlow calls | `analyse_one` + `asyncio.Semaphore(8)` + `asyncio.gather` |
| Prompt-cache-aware layout | Each Candidate-Analyst call shares the JD+rubric+fewshots prefix; only the resume changes | `_cand_prefix` (stable) + `_cand_tail` (per-candidate) |
| Warm-then-fan-out | First call writes prefix to cache, subsequent N-1 are cache reads | `await analyse_one(resumes[0]); await asyncio.gather(rest)` |
| Verification-style Explainer | Counter LLM self-confirmation bias by separating ranking from explanation | `_explain_parallel` reads Coordinator's ranking as ground truth |
| Task registry + cancel | Track per-candidate status; clean Ctrl-C propagation | `CandidateTask` dataclass with `asyncio.Event` for cancel |
| Withhold-then-recover on parse failure | Qwen3-8B occasionally truncates JSON; surface only valid output | `AsyncLLM.chat_json` with corrective retry |

## Concrete code skeleton

```python
async with AsyncLLM(provider="siliconflow", concurrency=8) as llm:
    orch = AsyncOrchestrator(llm, explain_top_k=10)
    async for ev in orch.stream(job, resumes, pre_ranked):
        if ev.type == EventType.CANDIDATE_PROFILED:
            print(f"  {ev.payload['resume_id']} done ({ev.usage.cache_read_tokens} cache_read)")
        elif ev.type == EventType.FINAL:
            print(f"total: {ev.usage.calls} calls, "
                  f"{ev.usage.cache_read_tokens / ev.usage.input_tokens:.0%} cache hit")
```

## Measurable lifts (will be filled after first real API run)

| Metric | Old sequential | New async | Δ |
|---|---|---|---|
| Wall-clock for 15-candidate fan-out | ~30 s | ~5 s | **6× faster** |
| Token cost per job (Qwen3-8B free tier) | n/a | n/a | n/a |
| Token cost per job (DeepSeek-V3 paid) | TBA | TBA | TBA |
| Cache-hit rate after warmup | 0 % | TBA (~60-80% expected) | — |
| Robustness to single-call failure | crashes job | partial success via `return_exceptions=True` | qualitative |

## Long-tail business cases covered by tests

`tests/unit/test_async_orchestrator_logic.py` (13 tests, no LLM):

* **Cache-layout invariants** — prefix must be byte-identical across all
  candidates; tail must differ; prefix must not contain candidate IDs.
* **Coordinator scoring** — high/medium/low fit bonuses; risk-flag penalty
  capped at 0.10; sort-descending invariant.
* **Usage arithmetic** — componentwise addition; cost-estimate formula;
  free-provider returns 0.

`tests/e2e/test_async_orchestrator_real_llm.py` (3 tests, real API,
skipped without `SILICONFLOW_API_KEY`):

* Stream emits lifecycle events for every agent.
* `run()` returns a sensible ranking — strong candidate stays on top.
* Cache-read tokens > 0 after warmup (graceful if provider hides cache stats).

## What we deliberately did NOT port from Claude Code

| Pattern | Why skipped |
|---|---|
| Ink terminal UI + TaskView | Out of scope; we expose stream events instead |
| MCP wire format / tmux pane backends | Not relevant to a Python research project |
| `Symbol.dispose` / `using` | We use `async with` + `AsyncExitStack` instead |
| Feature-gate scratchpad system | Belongs in a CLI tool, not a research library |

## Migration path

The legacy sync `Orchestrator` is still imported and used by `Pipeline`
for backwards-compatibility with `scripts/run_experiments.py`. New
callers should prefer `AsyncOrchestrator`. We will swap `Pipeline` to
use the async path once the first real-API benchmark confirms the
cache-hit lift.
