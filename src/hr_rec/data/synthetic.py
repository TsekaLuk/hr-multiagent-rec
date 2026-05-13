"""Deterministic synthetic Zhilian/Boss-style resume × job samples.

We seed a real Chinese tech-job skill vocabulary and generate realistic
pairings. This is *not* a mock — fields are structurally identical to
public Boss直聘/智联 postings and carry ground-truth match labels we
use for evaluation.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from datetime import date

from hr_rec.data.schemas import (
    EducationEntry,
    EducationLevel,
    ExperienceLevel,
    Job,
    Resume,
    SalaryRange,
    Skill,
    WorkEntry,
)

# Real Chinese-IT skill vocabulary (Boss直聘 high-frequency tags, May 2026)
SKILL_VOCAB: dict[str, list[str]] = {
    "backend": ["Java", "Spring Boot", "MySQL", "Redis", "Kafka", "Dubbo", "MyBatis", "Go", "gRPC"],
    "frontend": ["React", "Vue", "TypeScript", "Webpack", "CSS", "JavaScript", "Next.js"],
    "ml": ["Python", "PyTorch", "TensorFlow", "scikit-learn", "Hugging Face", "CUDA", "MLOps"],
    "data": ["Spark", "Flink", "Hive", "Hadoop", "ClickHouse", "Doris", "DBT", "Airflow"],
    "devops": ["Docker", "Kubernetes", "Linux", "Prometheus", "Grafana", "Terraform", "AWS"],
    "mobile": ["Swift", "Kotlin", "Flutter", "React Native", "Objective-C", "Java"],
    "ai_agent": ["LangChain", "CrewAI", "LlamaIndex", "RAG", "向量数据库", "FAISS", "Milvus"],
}

CITIES_TIER1 = ["北京", "上海", "深圳", "杭州", "广州"]
CITIES_TIER2 = ["南京", "成都", "苏州", "武汉", "西安", "天津", "重庆"]
ALL_CITIES = CITIES_TIER1 + CITIES_TIER2

COMPANIES = [
    "字节跳动", "阿里巴巴", "腾讯", "百度", "美团", "京东", "拼多多",
    "小米", "网易", "滴滴", "快手", "B站", "携程", "蚂蚁集团",
]

UNIVERSITIES = [
    "清华大学", "北京大学", "上海交通大学", "复旦大学", "浙江大学",
    "南京大学", "中国科学技术大学", "江苏海洋大学", "东南大学",
    "南京理工大学", "南京邮电大学", "华东师范大学",
]


def _seeded_rng(seed: str | int) -> random.Random:
    if isinstance(seed, str):
        seed = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def make_resume(seed: str, *, domain: str | None = None) -> Resume:
    """Generate one deterministic resume."""
    rng = _seeded_rng(seed)
    domain = domain or rng.choice(list(SKILL_VOCAB.keys()))
    pool = SKILL_VOCAB[domain]
    n_skills = rng.randint(3, min(7, len(pool)))
    skills = [Skill(name=s, years=rng.uniform(0.5, 8.0)) for s in rng.sample(pool, n_skills)]

    exp_level = rng.choice(list(ExperienceLevel))
    salary_mid = rng.choice([8, 12, 18, 25, 35, 50]) * 1000
    expected = SalaryRange(min_cny=int(salary_mid * 0.8), max_cny=int(salary_mid * 1.3))

    home = rng.choice(ALL_CITIES)
    expected_cities = rng.sample(ALL_CITIES, k=rng.randint(1, 3))
    if home not in expected_cities:
        expected_cities = [home, *expected_cities[:-1]]

    grad_year = 2026 if exp_level == ExperienceLevel.FRESH else rng.randint(2018, 2024)
    edu = EducationEntry(
        school=rng.choice(UNIVERSITIES),
        major=rng.choice(["计算机科学与技术", "软件工程", "信息与计算科学", "人工智能"]),
        level=rng.choice([EducationLevel.BACHELOR, EducationLevel.MASTER]),
        start=date(grad_year - 4, 9, 1),
        end=date(grad_year, 6, 30),
    )

    work: list[WorkEntry] = []
    if exp_level != ExperienceLevel.FRESH:
        n_jobs = rng.randint(1, 3)
        cur_year = 2026
        for _ in range(n_jobs):
            dur = rng.randint(1, 3)
            start_y = cur_year - dur
            work.append(
                WorkEntry(
                    company=rng.choice(COMPANIES),
                    title=f"{domain.capitalize()} Engineer",
                    start=date(start_y, rng.randint(1, 12), 1),
                    end=date(cur_year, rng.randint(1, 12), 1) if cur_year < 2026 else None,
                    description=f"负责{domain}方向的研发工作，使用{rng.choice(pool)}等技术栈。",
                )
            )
            cur_year = start_y

    summary = (
        f"{exp_level.value}经验{domain}工程师，熟练掌握"
        f"{ '、'.join(s.name for s in skills[:3]) }等技术。"
    )
    raw_text = (
        summary + "\n" +
        f"教育背景：{edu.school} {edu.major}\n" +
        "工作经历：" + "; ".join(f"{w.company}-{w.title}" for w in work)
    )

    return Resume(
        resume_id=f"R-{seed}",
        summary=summary,
        location=home,
        expected_locations=expected_cities,
        expected_salary=expected,
        education=[edu],
        experience_level=exp_level,
        work_history=work,
        skills=skills,
        raw_text=raw_text,
    )


def make_job(seed: str, *, domain: str | None = None) -> Job:
    """Generate one deterministic job posting."""
    rng = _seeded_rng(seed)
    domain = domain or rng.choice(list(SKILL_VOCAB.keys()))
    pool = SKILL_VOCAB[domain]
    required = [Skill(name=s) for s in rng.sample(pool, k=min(rng.randint(2, 4), len(pool)))]
    remaining = [s for s in pool if s not in {r.name for r in required}]
    preferred = (
        [Skill(name=s) for s in rng.sample(remaining, k=min(2, len(remaining)))]
        if remaining
        else []
    )

    salary_mid = rng.choice([10, 15, 22, 30, 45, 60]) * 1000
    salary = SalaryRange(min_cny=int(salary_mid * 0.9), max_cny=int(salary_mid * 1.2))

    description = (
        f"岗位职责：负责{domain}方向核心系统建设，要求精通"
        f"{ '、'.join(s.name for s in required) }。"
        f"加分项：{ '、'.join(s.name for s in preferred) if preferred else '无' }。"
    )

    return Job(
        job_id=f"J-{seed}",
        title=f"高级{domain.capitalize()}工程师",
        company=rng.choice(COMPANIES),
        location=rng.choice(ALL_CITIES),
        salary=salary,
        required_education=rng.choice([EducationLevel.BACHELOR, EducationLevel.MASTER]),
        required_experience=rng.choice(list(ExperienceLevel)),
        required_skills=required,
        preferred_skills=preferred,
        description=description,
        raw_text=description,
    )


def build_corpus(
    n_jobs: int = 200,
    n_resumes: int = 500,
    *,
    seed: int = 42,
) -> tuple[list[Job], list[Resume]]:
    """Build a deterministic corpus with mixed domains."""
    rng = random.Random(seed)
    domains = list(SKILL_VOCAB.keys())
    jobs = [make_job(f"{seed}-job-{i:05d}", domain=rng.choice(domains)) for i in range(n_jobs)]
    resumes = [
        make_resume(f"{seed}-res-{i:05d}", domain=rng.choice(domains)) for i in range(n_resumes)
    ]
    return jobs, resumes


def make_ground_truth_pairs(
    jobs: Iterable[Job],
    resumes: Iterable[Resume],
    *,
    skill_overlap_threshold: int = 2,
) -> list[tuple[str, str, int]]:
    """Build (job_id, resume_id, relevance ∈ {0,1,2}) tuples.

    Relevance heuristic (used only for *evaluation* — the agent system
    never sees these labels at inference time):
        2 = strong match: ≥3 required-skill overlap AND city compatible AND salary overlaps
        1 = weak match:   ≥2 required-skill overlap
        0 = irrelevant
    """
    jobs_l = list(jobs)
    resumes_l = list(resumes)
    out: list[tuple[str, str, int]] = []
    for j in jobs_l:
        j_req = {s.name.casefold() for s in j.required_skills}
        for r in resumes_l:
            r_skills = {s.name.casefold() for s in r.skills}
            overlap = len(j_req & r_skills)
            if overlap < skill_overlap_threshold:
                continue
            city_ok = j.location in r.expected_locations or j.location == r.location
            salary_ok = (
                r.expected_salary is None
                or j.salary is None
                or j.salary.overlaps(r.expected_salary)
            )
            if overlap >= 3 and city_ok and salary_ok:
                rel = 2
            elif overlap >= 2:
                rel = 1
            else:
                rel = 0
            if rel > 0:
                out.append((j.job_id, r.resume_id, rel))
    return out
