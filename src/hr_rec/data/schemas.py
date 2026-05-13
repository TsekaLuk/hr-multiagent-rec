"""Pydantic data contracts for resumes, jobs, and matching results.

These are the *boundary types* — anything that enters the system from
disk, network, or LLM tool calls is parsed through these classes. We
fail closed on validation errors rather than silently coercing.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EducationLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"  # 大专
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"


class ExperienceLevel(str, Enum):
    FRESH = "fresh"  # 应届
    Y1_3 = "1-3y"
    Y3_5 = "3-5y"
    Y5_10 = "5-10y"
    Y10P = "10y+"


class Skill(BaseModel):
    """A single skill mention, normalized."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: Annotated[str, Field(min_length=1, max_length=64)]
    esco_uri: str | None = None
    proficiency: float | None = Field(default=None, ge=0.0, le=1.0)
    years: float | None = Field(default=None, ge=0.0, le=80.0)


class EducationEntry(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    school: str
    major: str | None = None
    level: EducationLevel
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def _start_before_end(self) -> EducationEntry:
        if self.start and self.end and self.start > self.end:
            raise ValueError("education.start must be ≤ end")
        return self


class WorkEntry(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    company: str
    title: str
    start: date
    end: date | None = None  # None = current
    description: str = ""

    @model_validator(mode="after")
    def _start_before_end(self) -> WorkEntry:
        if self.end and self.start > self.end:
            raise ValueError("work.start must be ≤ end")
        return self


class SalaryRange(BaseModel):
    """Monthly salary in CNY ¥. min ≤ max."""

    model_config = ConfigDict(frozen=True)

    min_cny: Annotated[int, Field(ge=0, le=10_000_000)]
    max_cny: Annotated[int, Field(ge=0, le=10_000_000)]

    @model_validator(mode="after")
    def _ordered(self) -> SalaryRange:
        if self.min_cny > self.max_cny:
            raise ValueError("salary.min_cny must be ≤ max_cny")
        return self

    def overlaps(self, other: SalaryRange) -> bool:
        return not (self.max_cny < other.min_cny or other.max_cny < self.min_cny)


class Resume(BaseModel):
    """Canonical candidate representation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    resume_id: Annotated[str, Field(min_length=1)]
    summary: str = ""
    location: str  # 城市 e.g. "南京"
    expected_locations: list[str] = Field(default_factory=list)
    expected_salary: SalaryRange | None = None
    education: list[EducationEntry] = Field(default_factory=list)
    experience_level: ExperienceLevel
    work_history: list[WorkEntry] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    raw_text: str = ""  # original unstructured text for embedding

    @field_validator("skills")
    @classmethod
    def _dedup_skills(cls, v: list[Skill]) -> list[Skill]:
        seen: set[str] = set()
        out: list[Skill] = []
        for s in v:
            key = s.name.casefold()
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out


class Job(BaseModel):
    """Canonical job-posting representation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    job_id: Annotated[str, Field(min_length=1)]
    title: str
    company: str = ""
    location: str
    salary: SalaryRange | None = None
    required_education: EducationLevel | None = None
    required_experience: ExperienceLevel | None = None
    required_skills: list[Skill] = Field(default_factory=list)
    preferred_skills: list[Skill] = Field(default_factory=list)
    description: str = ""
    raw_text: str = ""

    @field_validator("required_skills", "preferred_skills")
    @classmethod
    def _dedup_skills(cls, v: list[Skill]) -> list[Skill]:
        seen: set[str] = set()
        out: list[Skill] = []
        for s in v:
            key = s.name.casefold()
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out


class MatchEvidence(BaseModel):
    """Why a candidate matches a job — for explainability."""

    model_config = ConfigDict(frozen=True)

    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    salary_compatible: bool
    location_compatible: bool
    education_satisfied: bool
    experience_satisfied: bool
    rationale: str = ""


class MatchScore(BaseModel):
    """Final ranked match result."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    resume_id: str
    employer_score: Annotated[float, Field(ge=0.0, le=1.0)]
    candidate_score: Annotated[float, Field(ge=0.0, le=1.0)]
    fused_score: Annotated[float, Field(ge=0.0, le=1.0)]
    semantic_similarity: Annotated[float, Field(ge=-1.0, le=1.0)]
    rerank_score: float | None = None
    evidence: MatchEvidence | None = None
