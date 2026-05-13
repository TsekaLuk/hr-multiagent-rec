"""Dataset loaders.

Currently supports:
  * `load_synthetic(n_jobs, n_resumes, seed)`  — deterministic synthetic corpus
  * `load_tianchi(path)`                       — Tianchi 智联招聘 dataset 31623
  * `load_jobsdf_skills(path)`                 — Job-SDF skill demand time-series

Tianchi schema reference (subset we adapt):
  Job side:    jd_no, jd_title, jd_sub_type, jd_industry, city, salary,
               min_years, max_years, min_edu_level, key_word,
               position_desc, requirement
  Resume side: user_id, live_city, desire_jd_city, desire_jd_salary_id,
               cur_industry, cur_jd_type, cur_salary, experience(list),
               edu_degree, major, start_work_date
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hr_rec.data.schemas import (
    EducationEntry,
    EducationLevel,
    ExperienceLevel,
    Job,
    Resume,
    SalaryRange,
    Skill,
)
from hr_rec.data.synthetic import build_corpus, make_ground_truth_pairs

# ---------- synthetic re-export ------------------------------------------


def load_synthetic(
    n_jobs: int = 200,
    n_resumes: int = 500,
    seed: int = 42,
) -> tuple[list[Job], list[Resume], list[tuple[str, str, int]]]:
    jobs, resumes = build_corpus(n_jobs=n_jobs, n_resumes=n_resumes, seed=seed)
    pairs = make_ground_truth_pairs(jobs, resumes)
    return jobs, resumes, pairs


# ---------- Tianchi adapters ---------------------------------------------

_TIANCHI_EDU_MAP: dict[int, EducationLevel] = {
    1: EducationLevel.HIGH_SCHOOL,
    2: EducationLevel.ASSOCIATE,
    3: EducationLevel.BACHELOR,
    4: EducationLevel.MASTER,
    5: EducationLevel.PHD,
}


def _years_to_exp_level(yrs: float) -> ExperienceLevel:
    if yrs < 1:
        return ExperienceLevel.FRESH
    if yrs < 3:
        return ExperienceLevel.Y1_3
    if yrs < 5:
        return ExperienceLevel.Y3_5
    if yrs < 10:
        return ExperienceLevel.Y5_10
    return ExperienceLevel.Y10P


def _parse_tianchi_salary(s: Any) -> SalaryRange | None:
    """Tianchi salary often encoded as '10K-20K' or numeric id."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    if isinstance(s, int | float):
        return SalaryRange(min_cny=int(s) * 1000, max_cny=int(s) * 1000)
    s = str(s).strip().replace("K", "").replace("k", "")
    if "-" in s:
        lo, hi = s.split("-", 1)
        try:
            return SalaryRange(min_cny=int(float(lo)) * 1000, max_cny=int(float(hi)) * 1000)
        except ValueError:
            return None
    return None


def load_tianchi_jobs(csv_path: str | Path) -> list[Job]:
    """Load Tianchi-format job postings."""
    df = pd.read_csv(csv_path)
    out: list[Job] = []
    for _, row in df.iterrows():
        salary = _parse_tianchi_salary(row.get("salary"))
        keywords = str(row.get("key_word", "") or "").split(",")
        req_skills = [Skill(name=k.strip()) for k in keywords if k.strip()]
        out.append(
            Job(
                job_id=str(row["jd_no"]),
                title=str(row.get("jd_title", "")),
                company=str(row.get("company_name", "")),
                location=str(row.get("city", "未知")),
                salary=salary,
                required_education=_TIANCHI_EDU_MAP.get(int(row.get("min_edu_level", 0) or 0)),
                required_experience=_years_to_exp_level(float(row.get("min_years", 0) or 0)),
                required_skills=req_skills,
                description=str(row.get("position_desc", "")),
                raw_text=f"{row.get('jd_title', '')}\n{row.get('requirement', '')}\n{row.get('position_desc', '')}",
            )
        )
    return out


def load_tianchi_resumes(csv_path: str | Path) -> list[Resume]:
    """Load Tianchi-format resumes."""
    df = pd.read_csv(csv_path)
    out: list[Resume] = []
    for _, row in df.iterrows():
        edu_lvl = _TIANCHI_EDU_MAP.get(int(row.get("edu_degree", 0) or 0), EducationLevel.BACHELOR)
        edu = EducationEntry(
            school=str(row.get("school", "未知")),
            major=str(row.get("major", "") or None) if row.get("major") else None,
            level=edu_lvl,
        )
        desire_city = str(row.get("desire_jd_city", "") or "")
        desire_cities = [c.strip() for c in desire_city.split(",") if c.strip()]
        live_city = str(row.get("live_city", "未知"))
        if live_city not in desire_cities:
            desire_cities = [live_city, *desire_cities]
        try:
            start_year = int(str(row.get("start_work_date", "2024"))[:4])
        except ValueError:
            start_year = 2024
        yrs = max(0, 2026 - start_year)
        out.append(
            Resume(
                resume_id=str(row["user_id"]),
                summary="",
                location=live_city,
                expected_locations=desire_cities,
                expected_salary=_parse_tianchi_salary(row.get("desire_jd_salary_id")),
                education=[edu],
                experience_level=_years_to_exp_level(yrs),
                work_history=[],
                skills=[],
                raw_text=str(row.get("resume_text", "")),
            )
        )
    return out


# ---------- Job-SDF skill demand -----------------------------------------


def load_jobsdf_skills(demand_csv: str | Path) -> pd.DataFrame:
    """Return a wide DataFrame: index = skill_id, columns = month."""
    return pd.read_csv(demand_csv, index_col=0)


# ---------- JSON dump/load (for caching processed data) ------------------


def dump_corpus_json(
    jobs: list[Job],
    resumes: list[Resume],
    pairs: list[tuple[str, str, int]],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.json").write_text(
        json.dumps([j.model_dump(mode="json") for j in jobs], ensure_ascii=False, indent=2)
    )
    (out / "resumes.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in resumes], ensure_ascii=False, indent=2)
    )
    (out / "pairs.json").write_text(
        json.dumps([{"job_id": j, "resume_id": r, "relevance": rel} for j, r, rel in pairs],
                   ensure_ascii=False, indent=2)
    )


def load_corpus_json(
    in_dir: str | Path,
) -> tuple[list[Job], list[Resume], list[tuple[str, str, int]]]:
    in_p = Path(in_dir)
    jobs = [Job.model_validate(j) for j in json.loads((in_p / "jobs.json").read_text())]
    resumes = [Resume.model_validate(r) for r in json.loads((in_p / "resumes.json").read_text())]
    pairs_raw = json.loads((in_p / "pairs.json").read_text())
    pairs = [(p["job_id"], p["resume_id"], p["relevance"]) for p in pairs_raw]
    return jobs, resumes, pairs
