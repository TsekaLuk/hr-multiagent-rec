"""Tests for the synthetic-corpus generator.

We test *properties* (determinism, distribution, ground-truth sanity)
rather than specific values, since the generator is RNG-seeded.
"""
from __future__ import annotations

import pytest

from hr_rec.data.schemas import ExperienceLevel, Job, Resume
from hr_rec.data.synthetic import (
    SKILL_VOCAB,
    build_corpus,
    make_ground_truth_pairs,
    make_job,
    make_resume,
)

pytestmark = pytest.mark.unit


class TestDeterminism:
    def test_same_seed_same_resume(self) -> None:
        r1 = make_resume("seed-A", domain="ml")
        r2 = make_resume("seed-A", domain="ml")
        assert r1.model_dump() == r2.model_dump()

    def test_same_seed_same_job(self) -> None:
        j1 = make_job("seed-X", domain="backend")
        j2 = make_job("seed-X", domain="backend")
        assert j1.model_dump() == j2.model_dump()

    def test_different_seed_different_output(self) -> None:
        r1 = make_resume("A")
        r2 = make_resume("B")
        assert r1.model_dump() != r2.model_dump()


class TestResumeProperties:
    @pytest.mark.parametrize("domain", list(SKILL_VOCAB.keys()))
    def test_all_domains_generate_valid_resumes(self, domain: str) -> None:
        r = make_resume(f"test-{domain}", domain=domain)
        assert isinstance(r, Resume)
        assert r.skills, "must have skills"
        assert all(s.name in SKILL_VOCAB[domain] for s in r.skills)

    def test_fresh_grad_has_no_work_history(self) -> None:
        # Re-roll until we get a fresh-grad; bounded loop
        for i in range(200):
            r = make_resume(f"fresh-{i}")
            if r.experience_level == ExperienceLevel.FRESH:
                assert r.work_history == []
                return
        pytest.skip("RNG never produced a fresh grad — extremely unlikely")

    def test_expected_locations_include_home(self) -> None:
        for i in range(50):
            r = make_resume(f"loc-{i}")
            assert r.location in r.expected_locations

    def test_salary_well_formed(self) -> None:
        for i in range(50):
            r = make_resume(f"sal-{i}")
            assert r.expected_salary is not None
            assert r.expected_salary.min_cny <= r.expected_salary.max_cny


class TestJobProperties:
    @pytest.mark.parametrize("domain", list(SKILL_VOCAB.keys()))
    def test_all_domains_generate_valid_jobs(self, domain: str) -> None:
        j = make_job(f"test-{domain}", domain=domain)
        assert isinstance(j, Job)
        assert j.required_skills, "must have required skills"

    def test_required_and_preferred_disjoint(self) -> None:
        for i in range(50):
            j = make_job(f"disj-{i}")
            req = {s.name for s in j.required_skills}
            pref = {s.name for s in j.preferred_skills}
            assert req.isdisjoint(pref), f"overlap: {req & pref}"


class TestCorpusAndGroundTruth:
    def test_corpus_sizes_respected(self) -> None:
        jobs, resumes = build_corpus(n_jobs=20, n_resumes=30, seed=7)
        assert len(jobs) == 20
        assert len(resumes) == 30

    def test_corpus_deterministic(self) -> None:
        j1, r1 = build_corpus(n_jobs=10, n_resumes=10, seed=99)
        j2, r2 = build_corpus(n_jobs=10, n_resumes=10, seed=99)
        assert [j.job_id for j in j1] == [j.job_id for j in j2]
        assert [r.resume_id for r in r1] == [r.resume_id for r in r2]

    def test_ground_truth_has_positive_pairs(self) -> None:
        jobs, resumes = build_corpus(n_jobs=50, n_resumes=100, seed=42)
        pairs = make_ground_truth_pairs(jobs, resumes)
        assert pairs, "expected at least some positive pairs"
        assert all(rel in (1, 2) for _, _, rel in pairs)

    def test_ground_truth_relevance_distribution(self) -> None:
        """Strong matches should be rarer than weak matches."""
        jobs, resumes = build_corpus(n_jobs=50, n_resumes=200, seed=42)
        pairs = make_ground_truth_pairs(jobs, resumes)
        n_strong = sum(1 for _, _, r in pairs if r == 2)
        n_weak = sum(1 for _, _, r in pairs if r == 1)
        # strong matches are stricter, must be ≤ weak ones (almost surely)
        assert n_strong <= n_weak

    def test_ground_truth_ids_exist_in_corpus(self) -> None:
        jobs, resumes = build_corpus(n_jobs=30, n_resumes=60, seed=11)
        job_ids = {j.job_id for j in jobs}
        res_ids = {r.resume_id for r in resumes}
        for j_id, r_id, _ in make_ground_truth_pairs(jobs, resumes):
            assert j_id in job_ids
            assert r_id in res_ids
