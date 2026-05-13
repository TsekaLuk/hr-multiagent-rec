"""Bidirectional matching scorer — long-tail business scenario tests.

We test the *scoring functions* in isolation (no model loading). Edge cases:
- salary inversion (candidate expected > job offered)
- city mismatch (Chinese tier-1 vs tier-2)
- skill overlap with case/whitespace variations
- education hard-floor violations
- experience over-/under-qualification
- empty required skills (job description with no extracted skills)
- candidate with no expected salary / location
- fully missing optional fields
"""
from __future__ import annotations

import pytest

from hr_rec.data.schemas import (
    EducationEntry,
    EducationLevel,
    ExperienceLevel,
    Job,
    Resume,
    SalaryRange,
    Skill,
)
from hr_rec.matching.scoring import (
    BidirectionalScore,
    bidirectional_score,
    candidate_side_score,
    employer_side_score,
    fuse_scores,
)

pytestmark = pytest.mark.unit


# ---------- helpers --------------------------------------------------------


def _resume(
    *,
    location: str = "南京",
    expected_locations: list[str] | None = None,
    expected_salary: SalaryRange | None = None,
    education: EducationLevel = EducationLevel.BACHELOR,
    experience: ExperienceLevel = ExperienceLevel.Y1_3,
    skills: list[str] | None = None,
) -> Resume:
    return Resume(
        resume_id="R",
        location=location,
        expected_locations=expected_locations or [location],
        expected_salary=expected_salary,
        education=[EducationEntry(school="X", level=education)],
        experience_level=experience,
        skills=[Skill(name=s) for s in (skills or ["Python", "MySQL"])],
    )


def _job(
    *,
    location: str = "南京",
    salary: SalaryRange | None = None,
    required_education: EducationLevel | None = EducationLevel.BACHELOR,
    required_experience: ExperienceLevel | None = ExperienceLevel.Y1_3,
    required_skills: list[str] | None = None,
    preferred_skills: list[str] | None = None,
) -> Job:
    return Job(
        job_id="J",
        title="Backend Engineer",
        location=location,
        salary=salary,
        required_education=required_education,
        required_experience=required_experience,
        required_skills=[Skill(name=s) for s in (required_skills or ["Python", "MySQL"])],
        preferred_skills=[Skill(name=s) for s in (preferred_skills or [])],
    )


# ---------- Employer-side score -------------------------------------------


class TestEmployerSide:
    def test_perfect_match_score_high(self) -> None:
        r = _resume(skills=["Python", "MySQL", "Redis"])
        j = _job(required_skills=["Python", "MySQL"])
        s = employer_side_score(j, r)
        assert s > 0.85

    def test_no_skill_overlap_score_low(self) -> None:
        r = _resume(skills=["Photoshop", "Illustrator"])
        j = _job(required_skills=["Python", "MySQL"])
        s = employer_side_score(j, r)
        assert s < 0.30

    def test_case_insensitive_skill_match(self) -> None:
        r = _resume(skills=["python", "MYSQL"])
        j = _job(required_skills=["Python", "MySQL"])
        assert employer_side_score(j, r) > 0.85

    def test_education_below_required_penalized(self) -> None:
        r = _resume(education=EducationLevel.HIGH_SCHOOL, skills=["Python", "MySQL"])
        j = _job(required_education=EducationLevel.BACHELOR)
        s = employer_side_score(j, r)
        assert s < 0.6  # hard penalty

    def test_education_above_required_not_penalized(self) -> None:
        r = _resume(education=EducationLevel.MASTER, skills=["Python", "MySQL"])
        j = _job(required_education=EducationLevel.BACHELOR)
        assert employer_side_score(j, r) > 0.85

    def test_experience_above_required_not_penalized(self) -> None:
        r = _resume(experience=ExperienceLevel.Y5_10)
        j = _job(required_experience=ExperienceLevel.Y1_3)
        assert employer_side_score(j, r) > 0.7

    def test_experience_below_required_penalized(self) -> None:
        r = _resume(experience=ExperienceLevel.FRESH)
        j = _job(required_experience=ExperienceLevel.Y5_10)
        assert employer_side_score(j, r) < 0.7

    def test_preferred_skills_bonus(self) -> None:
        r1 = _resume(skills=["Python", "MySQL"])
        r2 = _resume(skills=["Python", "MySQL", "Kafka"])
        j = _job(required_skills=["Python", "MySQL"], preferred_skills=["Kafka"])
        assert employer_side_score(j, r2) > employer_side_score(j, r1)

    def test_empty_required_skills_returns_neutral(self) -> None:
        r = _resume()
        j = _job(required_skills=[])
        s = employer_side_score(j, r)
        assert 0.4 <= s <= 1.0  # no required skills → can't fail on skills

    def test_score_in_unit_interval(self) -> None:
        r = _resume()
        j = _job()
        s = employer_side_score(j, r)
        assert 0.0 <= s <= 1.0


# ---------- Candidate-side score ------------------------------------------


class TestCandidateSide:
    def test_location_in_expected_high(self) -> None:
        r = _resume(location="南京", expected_locations=["南京", "上海"])
        j = _job(location="南京")
        assert candidate_side_score(j, r) > 0.8

    def test_location_not_in_expected_penalized(self) -> None:
        r = _resume(location="南京", expected_locations=["南京"])
        j = _job(location="深圳")
        assert candidate_side_score(j, r) < 0.6

    def test_salary_overlap_high(self) -> None:
        r = _resume(expected_salary=SalaryRange(min_cny=15_000, max_cny=25_000))
        j = _job(salary=SalaryRange(min_cny=18_000, max_cny=22_000))
        assert candidate_side_score(j, r) > 0.8

    def test_salary_inversion_low(self) -> None:
        """Candidate wants ¥30k+, job offers ¥10-15k → strong penalty."""
        r = _resume(expected_salary=SalaryRange(min_cny=30_000, max_cny=40_000))
        j = _job(salary=SalaryRange(min_cny=10_000, max_cny=15_000))
        assert candidate_side_score(j, r) < 0.5

    def test_salary_unspecified_treated_neutral(self) -> None:
        """No expected_salary should not crash and gives moderate score."""
        r = _resume(expected_salary=None)
        j = _job(salary=SalaryRange(min_cny=10_000, max_cny=15_000))
        s = candidate_side_score(j, r)
        assert 0.3 <= s <= 1.0

    def test_score_in_unit_interval(self) -> None:
        r = _resume()
        j = _job()
        s = candidate_side_score(j, r)
        assert 0.0 <= s <= 1.0


# ---------- Fusion --------------------------------------------------------


class TestFusion:
    def test_convex_combination_bounds(self) -> None:
        for a in (0.0, 0.3, 0.5, 0.7, 1.0):
            for b in (0.0, 0.3, 0.5, 0.7, 1.0):
                f = fuse_scores(a, b, alpha=0.6)
                assert min(a, b) <= f <= max(a, b)

    def test_alpha_zero_returns_candidate_side(self) -> None:
        assert fuse_scores(0.9, 0.2, alpha=0.0) == pytest.approx(0.2)

    def test_alpha_one_returns_employer_side(self) -> None:
        assert fuse_scores(0.9, 0.2, alpha=1.0) == pytest.approx(0.9)

    def test_alpha_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            fuse_scores(0.5, 0.5, alpha=1.5)


# ---------- End-to-end bidirectional score --------------------------------


class TestBidirectional:
    def test_returns_evidence(self) -> None:
        r = _resume(skills=["Python", "MySQL"], expected_locations=["南京"])
        j = _job(required_skills=["Python", "MySQL"], location="南京")
        result = bidirectional_score(j, r, alpha=0.6)
        assert isinstance(result, BidirectionalScore)
        assert result.evidence is not None
        assert "Python" in result.evidence.matched_skills
        assert result.evidence.location_compatible is True

    def test_missing_skills_reported(self) -> None:
        r = _resume(skills=["Python"])
        j = _job(required_skills=["Python", "MySQL", "Redis"])
        result = bidirectional_score(j, r)
        assert set(result.evidence.missing_skills) == {"MySQL", "Redis"}

    def test_score_monotone_in_overlap(self) -> None:
        """More skill overlap → higher fused score, all else equal."""
        j = _job(required_skills=["Python", "MySQL", "Redis", "Kafka"])
        r1 = _resume(skills=["Python"])
        r2 = _resume(skills=["Python", "MySQL"])
        r3 = _resume(skills=["Python", "MySQL", "Redis"])
        r4 = _resume(skills=["Python", "MySQL", "Redis", "Kafka"])
        s1 = bidirectional_score(j, r1).fused
        s2 = bidirectional_score(j, r2).fused
        s3 = bidirectional_score(j, r3).fused
        s4 = bidirectional_score(j, r4).fused
        assert s1 < s2 < s3 < s4
