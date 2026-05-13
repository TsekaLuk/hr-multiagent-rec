# Contributing

Thanks for considering a contribution! HR-MultiAgent-Rec is an academic-grade
codebase: small, fully-typed, no mocks in the test suite, and reproducible
end-to-end on a laptop.

## Quick start

```bash
git clone https://github.com/TsekaLuk/hr-multiagent-rec
cd hr-multiagent-rec
make dev
make test          # 100+ unit tests, < 2 s
make test-int      # heavier integration suite, < 1 min
make lint type     # ruff + mypy
```

## Engineering rules

1. **No mocks.** Tests hit real models, real FAISS, real Pydantic. If a
   dependency is genuinely unavailable (no network / no GPU), `pytest.skip` —
   never invent fake behaviour.
2. **Tests come first.** New behaviour ships with the test that motivated
   it. PRs without coverage are sent back. Long-tail business edge cases
   (salary inversion, Unicode emoji, education hard floors) are part of the
   contract.
3. **One module, one purpose.** ≤ 400 lines per file. If you need more, you
   need a new module.
4. **Pydantic at the boundary.** Anything that enters the system from disk,
   network, or LLM output is parsed through a typed schema.
5. **Determinism.** Every random source takes a seed parameter.

## What we look for

* New baselines (e.g. DPR, ColBERT) under `hr_rec/baselines.py`.
* Additional dataset adapters under `hr_rec/data/loaders.py`.
* Performance optimisations on the FAISS indexer or batch encoding.
* Re-ranker backbones beyond Qwen3-Reranker.
* New agents (e.g. a salary-negotiation agent, a culture-fit agent) following
  the `hr_rec.agents.base.Agent` contract.

## What we do not accept

* Vendored proprietary data.
* Breaking the `--no-mocks` test policy.
* Adding heavyweight dependencies (Spark, Ray, vLLM) without an opt-in extra
  and a clear motivation in the PR description.

## Release checklist (maintainers)

- [ ] `make check` is green
- [ ] `make eval --max-jobs 10` smoke run succeeds
- [ ] Changelog entry added
- [ ] arXiv `main.tex` re-builds with `make paper`
- [ ] Git tag matches `pyproject.toml` version
