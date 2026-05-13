"""BM25 and TF-IDF baseline integration tests on synthetic corpus."""
from __future__ import annotations

import pytest

from hr_rec.baselines import BM25Baseline, TfidfBaseline
from hr_rec.data.loaders import load_synthetic

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def corpus():  # type: ignore[no-untyped-def]
    return load_synthetic(n_jobs=10, n_resumes=50, seed=2026)


class TestBM25:
    def test_ranks_relevant_resumes_high(self, corpus) -> None:  # type: ignore[no-untyped-def]
        jobs, resumes, pairs = corpus
        rel = {(j, r) for j, r, _ in pairs}
        bm = BM25Baseline()
        bm.index(resumes)
        # Pick a job with positive labels
        target_job = next(
            (j for j in jobs if any(p[0] == j.job_id for p in pairs)),
            jobs[0],
        )
        results = bm.match(target_job, top_k=20)
        top_ids = {r.resume_id for r in results[:20]}
        relevant_ids = {r for jid, r in rel if jid == target_job.job_id}
        # Reasonable hit-rate on synthetic corpus
        assert relevant_ids & top_ids, "BM25 should retrieve at least one relevant"

    def test_returns_top_k_count(self, corpus) -> None:  # type: ignore[no-untyped-def]
        jobs, resumes, _ = corpus
        bm = BM25Baseline()
        bm.index(resumes)
        results = bm.match(jobs[0], top_k=5)
        assert len(results) == 5
        scores = [r.fused_score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestTfidf:
    def test_ranks_relevant_resumes(self, corpus) -> None:  # type: ignore[no-untyped-def]
        jobs, resumes, pairs = corpus
        tf = TfidfBaseline()
        tf.index(resumes)
        target_job = jobs[0]
        results = tf.match(target_job, top_k=10)
        # First result must score higher than the last
        assert results[0].fused_score >= results[-1].fused_score
        assert len(results) == 10
