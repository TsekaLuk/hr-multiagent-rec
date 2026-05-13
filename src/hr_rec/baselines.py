"""Classical IR baselines: BM25 and TF-IDF.

Used in the ablation grid to anchor the semantic / multi-agent gains.
"""
from __future__ import annotations

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hr_rec.data.schemas import Job, MatchScore, Resume


def _tokenize_zh(text: str) -> list[str]:
    return [t for t in jieba.lcut(text or "") if t.strip()]


def _resume_text(r: Resume) -> str:
    return " ".join(
        [
            r.summary,
            " ".join(s.name for s in r.skills),
            " ".join(f"{w.title} {w.company}" for w in r.work_history),
            r.raw_text,
        ]
    )


def _job_text(j: Job) -> str:
    return " ".join(
        [
            j.title,
            " ".join(s.name for s in j.required_skills),
            " ".join(s.name for s in j.preferred_skills),
            j.description,
            j.raw_text,
        ]
    )


class BM25Baseline:
    """BM25 over jieba-tokenized resume text."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._resumes: dict[str, Resume] = {}

    def index(self, resumes: list[Resume]) -> None:
        self._resumes = {r.resume_id: r for r in resumes}
        self._ids = list(self._resumes.keys())
        corpus = [_tokenize_zh(_resume_text(self._resumes[i])) for i in self._ids]
        self._bm25 = BM25Okapi(corpus)

    def match(self, job: Job, top_k: int = 50) -> list[MatchScore]:
        assert self._bm25 is not None, "call index() first"
        q = _tokenize_zh(_job_text(job))
        scores = self._bm25.get_scores(q)
        order = np.argsort(-scores)[:top_k]
        smax = float(max(scores[order]) or 1.0)
        return [
            MatchScore(
                job_id=job.job_id,
                resume_id=self._ids[i],
                employer_score=0.0,
                candidate_score=0.0,
                fused_score=float(scores[i] / smax),
                semantic_similarity=0.0,
                rerank_score=None,
                evidence=None,
            )
            for i in order
        ]


class TfidfBaseline:
    """TF-IDF cosine similarity baseline."""

    def __init__(self) -> None:
        self._vec: TfidfVectorizer | None = None
        self._mat = None
        self._ids: list[str] = []
        self._resumes: dict[str, Resume] = {}

    def index(self, resumes: list[Resume]) -> None:
        self._resumes = {r.resume_id: r for r in resumes}
        self._ids = list(self._resumes.keys())
        corpus = [" ".join(_tokenize_zh(_resume_text(self._resumes[i]))) for i in self._ids]
        self._vec = TfidfVectorizer(token_pattern=r"\S+")
        self._mat = self._vec.fit_transform(corpus)

    def match(self, job: Job, top_k: int = 50) -> list[MatchScore]:
        assert self._vec is not None and self._mat is not None, "call index() first"
        q = self._vec.transform([" ".join(_tokenize_zh(_job_text(job)))])
        sims = cosine_similarity(q, self._mat).ravel()
        order = np.argsort(-sims)[:top_k]
        return [
            MatchScore(
                job_id=job.job_id,
                resume_id=self._ids[i],
                employer_score=0.0,
                candidate_score=0.0,
                fused_score=float(sims[i]),
                semantic_similarity=float(sims[i]),
                rerank_score=None,
                evidence=None,
            )
            for i in order
        ]
