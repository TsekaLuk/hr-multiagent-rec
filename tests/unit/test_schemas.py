"""Data-contract tests. Strict validation — no mocks.

Long-tail business cases included: salary inversion, empty skills,
duplicate skills, future-dated work, Unicode normalization, etc.
"""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from hr_rec.data.schemas import (
    EducationEntry,
    EducationLevel,
    ExperienceLevel,
    Job,
    MatchEvidence,
    MatchScore,
    Resume,
    SalaryRange,
    Skill,
    WorkEntry,
)

pytestmark = pytest.mark.unit


# ---------- Skill ----------------------------------------------------------


class TestSkill:
    def test_minimum_valid(self) -> None:
        s = Skill(name="Python")
        assert s.name == "Python"
        assert s.proficiency is None

    def test_strips_whitespace(self) -> None:
        assert Skill(name="  Java  ").name == "Java"

    def test_proficiency_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Skill(name="Go", proficiency=1.5)
        with pytest.raises(ValidationError):
            Skill(name="Go", proficiency=-0.01)

    def test_years_unreasonable_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Skill(name="C", years=100.0)

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Skill(name="")

    def test_immutable(self) -> None:
        s = Skill(name="Rust")
        with pytest.raises(ValidationError):
            s.name = "Zig"  # type: ignore[misc]


# ---------- SalaryRange ----------------------------------------------------


class TestSalaryRange:
    def test_ordered(self) -> None:
        r = SalaryRange(min_cny=10_000, max_cny=20_000)
        assert r.min_cny <= r.max_cny

    def test_inversion_rejected(self) -> None:
        with pytest.raises(ValidationError, match="min_cny must be ≤ max_cny"):
            SalaryRange(min_cny=30_000, max_cny=20_000)

    def test_equal_bounds_allowed(self) -> None:
        SalaryRange(min_cny=15_000, max_cny=15_000)

    def test_overlap_logic(self) -> None:
        a = SalaryRange(min_cny=10_000, max_cny=20_000)
        b = SalaryRange(min_cny=18_000, max_cny=25_000)
        c = SalaryRange(min_cny=25_000, max_cny=30_000)
        assert a.overlaps(b)
        assert not a.overlaps(c)
        assert b.overlaps(a)  # symmetric

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SalaryRange(min_cny=-1, max_cny=10_000)

    def test_unreasonably_large_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SalaryRange(min_cny=0, max_cny=10_000_001)


# ---------- Education / Work entries --------------------------------------


class TestEducationEntry:
    def test_dates_ordered(self) -> None:
        EducationEntry(
            school="江苏海洋大学",
            level=EducationLevel.BACHELOR,
            start=date(2022, 9, 1),
            end=date(2026, 6, 30),
        )

    def test_start_after_end_rejected(self) -> None:
        with pytest.raises(ValidationError, match="education.start"):
            EducationEntry(
                school="X University",
                level=EducationLevel.MASTER,
                start=date(2026, 6, 30),
                end=date(2022, 9, 1),
            )

    def test_no_dates_allowed(self) -> None:
        EducationEntry(school="X", level=EducationLevel.BACHELOR)


class TestWorkEntry:
    def test_current_job_no_end(self) -> None:
        w = WorkEntry(company="ByteDance", title="SWE", start=date(2024, 1, 1))
        assert w.end is None

    def test_start_after_end_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkEntry(
                company="X",
                title="Y",
                start=date(2024, 6, 1),
                end=date(2024, 1, 1),
            )


# ---------- Resume ---------------------------------------------------------


class TestResume:
    def _ok_kwargs(self) -> dict[str, object]:
        return {
            "resume_id": "R-0001",
            "location": "南京",
            "experience_level": ExperienceLevel.FRESH,
        }

    def test_minimum_valid(self) -> None:
        r = Resume(**self._ok_kwargs())  # type: ignore[arg-type]
        assert r.resume_id == "R-0001"

    def test_duplicate_skills_deduped(self) -> None:
        r = Resume(
            **self._ok_kwargs(),  # type: ignore[arg-type]
            skills=[
                Skill(name="Python"),
                Skill(name="python"),  # casefold dup
                Skill(name="  Python "),  # whitespace dup
                Skill(name="Java"),
            ],
        )
        names = [s.name.lower() for s in r.skills]
        assert names.count("python") == 1
        assert "java" in names

    def test_empty_resume_id_rejected(self) -> None:
        kw = self._ok_kwargs()
        kw["resume_id"] = ""
        with pytest.raises(ValidationError):
            Resume(**kw)  # type: ignore[arg-type]

    def test_chinese_unicode_preserved(self) -> None:
        r = Resume(
            **self._ok_kwargs(),  # type: ignore[arg-type]
            summary="精通分布式系统 🚀 与机器学习",
            skills=[Skill(name="分布式系统"), Skill(name="机器学习")],
        )
        assert "分布式系统" in r.summary
        assert r.skills[0].name == "分布式系统"


# ---------- Job ------------------------------------------------------------


class TestJob:
    def _ok_kwargs(self) -> dict[str, object]:
        return {"job_id": "J-0001", "title": "ML Engineer", "location": "上海"}

    def test_minimum_valid(self) -> None:
        j = Job(**self._ok_kwargs())  # type: ignore[arg-type]
        assert j.job_id == "J-0001"

    def test_required_and_preferred_independent_dedup(self) -> None:
        j = Job(
            **self._ok_kwargs(),  # type: ignore[arg-type]
            required_skills=[Skill(name="Python"), Skill(name="python")],
            preferred_skills=[Skill(name="Python"), Skill(name="GPU")],
        )
        assert len(j.required_skills) == 1
        # preferred kept independently — dedup is per-list
        assert {s.name for s in j.preferred_skills} == {"Python", "GPU"}

    def test_long_description_allowed(self) -> None:
        j = Job(
            **self._ok_kwargs(),  # type: ignore[arg-type]
            description="岗位职责" * 5000,
        )
        assert len(j.description) >= 20_000


# ---------- MatchScore -----------------------------------------------------


class TestMatchScore:
    def test_score_bounds(self) -> None:
        ms = MatchScore(
            job_id="J", resume_id="R",
            employer_score=0.8, candidate_score=0.7,
            fused_score=0.75, semantic_similarity=0.9,
        )
        assert 0.0 <= ms.fused_score <= 1.0
        assert -1.0 <= ms.semantic_similarity <= 1.0

    def test_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MatchScore(
                job_id="J", resume_id="R",
                employer_score=1.1, candidate_score=0.5,
                fused_score=0.5, semantic_similarity=0.5,
            )

    def test_semantic_similarity_can_be_negative(self) -> None:
        """Cosine similarity can legitimately be negative."""
        MatchScore(
            job_id="J", resume_id="R",
            employer_score=0.0, candidate_score=0.0,
            fused_score=0.0, semantic_similarity=-0.42,
        )


# ---------- MatchEvidence --------------------------------------------------


class TestMatchEvidence:
    def test_evidence_immutable(self) -> None:
        ev = MatchEvidence(
            matched_skills=["Python"],
            missing_skills=["Rust"],
            salary_compatible=True,
            location_compatible=False,
            education_satisfied=True,
            experience_satisfied=True,
        )
        with pytest.raises(ValidationError):
            ev.salary_compatible = False  # type: ignore[misc]
