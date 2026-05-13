"""Bidirectional matching scorer.

Two complementary scores fused by a convex combination:

    fused = α · employer_side(j, r) + (1 − α) · candidate_side(j, r)

Design rationale (matches the long-tail test contract):

* **Hard floors are multiplicative.** Failing the education or experience
  requirement *multiplies* the score by < 1.0, so even with perfect skill
  coverage the candidate still drops sharply. This mirrors HR reality
  where a "本科起步" job effectively gates out below-floor applicants.

* **Skill coverage is the dominant signal** on the employer side.
  Preferred-skill bonus is folded into the *base* (not added on top),
  so coverage of preferred skills always strictly beats no-coverage,
  even when required skills are 100% covered.

* **Candidate-side severe salary inversion is a hard-negative signal.**
  We use a negative sub-score (clipped at the end) so a candidate
  asking ¥30k facing a ¥10–15k job lands < 0.5 even with a city match.

All return values live in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

from hr_rec.data.schemas import (
    EducationLevel,
    ExperienceLevel,
    Job,
    MatchEvidence,
    Resume,
)

_EDU_ORDER: dict[EducationLevel, int] = {
    EducationLevel.HIGH_SCHOOL: 0,
    EducationLevel.ASSOCIATE: 1,
    EducationLevel.BACHELOR: 2,
    EducationLevel.MASTER: 3,
    EducationLevel.PHD: 4,
}

_EXP_ORDER: dict[ExperienceLevel, int] = {
    ExperienceLevel.FRESH: 0,
    ExperienceLevel.Y1_3: 1,
    ExperienceLevel.Y3_5: 2,
    ExperienceLevel.Y5_10: 3,
    ExperienceLevel.Y10P: 4,
}

# Hard-floor multiplicative penalties.
_EDU_FAIL_FACTOR = 0.4
_EXP_FAIL_FACTOR = 0.5


@dataclass(frozen=True)
class BidirectionalScore:
    employer: float
    candidate: float
    fused: float
    evidence: MatchEvidence | None


# ---------- shared helpers -------------------------------------------------


def _norm_set(names: list[str]) -> set[str]:
    return {n.strip().casefold() for n in names if n and n.strip()}


def _max_education(r: Resume) -> EducationLevel | None:
    if not r.education:
        return None
    return max(r.education, key=lambda e: _EDU_ORDER[e.level]).level


def _education_satisfied(j: Job, r: Resume) -> bool:
    if j.required_education is None:
        return True
    have = _max_education(r)
    return have is not None and _EDU_ORDER[have] >= _EDU_ORDER[j.required_education]


def _experience_satisfied(j: Job, r: Resume) -> bool:
    if j.required_experience is None:
        return True
    return _EXP_ORDER[r.experience_level] >= _EXP_ORDER[j.required_experience]


# ---------- employer side --------------------------------------------------


def employer_side_score(
    j: Job, r: Resume,
    *,
    required_weight: float = 0.85,
    preferred_weight: float = 0.15,
) -> float:
    req = _norm_set([s.name for s in j.required_skills])
    pref = _norm_set([s.name for s in j.preferred_skills])
    cand = _norm_set([s.name for s in r.skills])

    if req:
        req_cov = len(req & cand) / len(req)
    else:
        req_cov = 0.6  # neutral when no extracted requirements

    pref_cov = (len(pref & cand) / len(pref)) if pref else 0.0

    # Preferred skills are folded into the base so they always strictly differentiate.
    if pref:
        base = required_weight * req_cov + preferred_weight * pref_cov
    else:
        base = req_cov

    # Hard-floor multiplicative penalties.
    if not _education_satisfied(j, r):
        base *= _EDU_FAIL_FACTOR
    if not _experience_satisfied(j, r):
        base *= _EXP_FAIL_FACTOR

    return float(min(1.0, max(0.0, base)))


# ---------- candidate side -------------------------------------------------


_SEVERE_GAP_RATIO = 0.2


def _salary_subscore(j: Job, r: Resume) -> float:
    """Returns a value in [-1, +1]; caller will clip the final score."""
    if r.expected_salary is None or j.salary is None:
        return 0.7  # mild positive when no info — don't punish
    if j.salary.overlaps(r.expected_salary):
        return 1.0
    # Compute gap relative to candidate's expected ceiling
    if r.expected_salary.min_cny > j.salary.max_cny:
        gap = r.expected_salary.min_cny - j.salary.max_cny
    else:
        gap = j.salary.min_cny - r.expected_salary.max_cny
    ratio = gap / max(1, r.expected_salary.max_cny)
    if ratio >= _SEVERE_GAP_RATIO:
        return -1.0  # severe inversion → strong negative
    return 0.3


def candidate_side_score(
    j: Job, r: Resume,
    *,
    location_weight: float = 0.5,
    salary_weight: float = 0.5,
) -> float:
    expected = {c.strip() for c in r.expected_locations if c.strip()}
    loc_score = 1.0 if (j.location in expected or j.location == r.location) else 0.1
    sal_score = _salary_subscore(j, r)

    raw = location_weight * loc_score + salary_weight * sal_score
    return float(min(1.0, max(0.0, raw)))


# ---------- fusion ---------------------------------------------------------


def fuse_scores(employer: float, candidate: float, *, alpha: float = 0.6) -> float:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return float(alpha * employer + (1 - alpha) * candidate)


def bidirectional_score(j: Job, r: Resume, *, alpha: float = 0.6) -> BidirectionalScore:
    emp = employer_side_score(j, r)
    cand = candidate_side_score(j, r)
    fused = fuse_scores(emp, cand, alpha=alpha)

    req_names = {s.name for s in j.required_skills}
    cand_names_cf = {s.name.casefold() for s in r.skills}
    matched = [n for n in req_names if n.casefold() in cand_names_cf]
    missing = [n for n in req_names if n.casefold() not in cand_names_cf]

    expected = {c.strip() for c in r.expected_locations if c.strip()}
    location_compat = j.location in expected or j.location == r.location
    salary_compat = (
        r.expected_salary is None
        or j.salary is None
        or j.salary.overlaps(r.expected_salary)
    )

    evidence = MatchEvidence(
        matched_skills=matched,
        missing_skills=missing,
        salary_compatible=salary_compat,
        location_compatible=location_compat,
        education_satisfied=_education_satisfied(j, r),
        experience_satisfied=_experience_satisfied(j, r),
        rationale=(
            f"技能覆盖 {len(matched)}/{len(req_names) or 1} | "
            f"地点 {'匹配' if location_compat else '不匹配'} | "
            f"薪资 {'匹配' if salary_compat else '不匹配'} | "
            f"学历 {'满足' if _education_satisfied(j, r) else '不满足'} | "
            f"经验 {'满足' if _experience_satisfied(j, r) else '不满足'}"
        ),
    )
    return BidirectionalScore(employer=emp, candidate=cand, fused=fused, evidence=evidence)
