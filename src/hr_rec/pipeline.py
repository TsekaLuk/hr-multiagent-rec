"""End-to-end matching pipeline. Composes the four layers.

For a single job:
    1.  Encode the job (Qwen3-Embedding).
    2.  FAISS retrieve top-N candidate resumes (vector recall).
    3.  Optional Qwen3-Reranker cross-encoder rerank on those N.
    4.  Compute bidirectional employer / candidate score for each.
    5.  Optional multi-agent coordination on the top-M.
    6.  Return final ranked MatchScore list.

Layers can be turned off — that's how we run the ablation grid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from hr_rec.data.schemas import Job, MatchScore, Resume
from hr_rec.encoding.embedder import Embedder
from hr_rec.encoding.indexer import VectorIndex
from hr_rec.matching.scoring import bidirectional_score

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    top_n_recall: int = 50           # FAISS top-N candidates per job
    top_m_agent: int = 20            # Pass top-M into multi-agent stage
    use_reranker: bool = True
    use_bidirectional: bool = True
    use_multi_agent: bool = True
    alpha: float = 0.6               # employer-side weight in convex fusion
    semantic_weight: float = 0.4     # blend between semantic & bidirectional
    rerank_weight: float = 0.4
    dim: int | None = None           # MRL truncation dim; None = native


class Pipeline:
    """Composable matching pipeline.

    Heavy components (embedder, reranker, orchestrator) are injected so
    we can swap implementations and write ablations cheaply.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        reranker=None,  # type: ignore[no-untyped-def]  hr_rec.matching.reranker.Reranker | None
        orchestrator=None,  # type: ignore[no-untyped-def]  hr_rec.agents.orchestrator.Orchestrator | None
        config: PipelineConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.orchestrator = orchestrator
        self.config = config or PipelineConfig()
        self._index: VectorIndex | None = None
        self._resume_lookup: dict[str, Resume] = {}

    # ---- indexing -------------------------------------------------------

    def index_resumes(self, resumes: list[Resume]) -> None:
        """Encode and FAISS-index all resumes."""
        if not resumes:
            raise ValueError("no resumes to index")
        texts = [_resume_text(r) for r in resumes]
        vecs = self.embedder.encode_batch(texts, dim=self.config.dim)
        dim = vecs.shape[1]
        # Use flat index for ≤5k resumes (exact) else IVF (approximate).
        if len(resumes) <= 5000:
            self._index = VectorIndex(dim=dim, index_type="flat")
        else:
            nlist = max(16, int(np.sqrt(len(resumes))))
            self._index = VectorIndex(
                dim=dim, index_type="ivf", nlist=nlist, nprobe=nlist // 2
            )
        self._index.add(vecs, [r.resume_id for r in resumes])
        self._resume_lookup = {r.resume_id: r for r in resumes}
        logger.info("indexed %d resumes (dim=%d)", len(resumes), dim)

    # ---- matching -------------------------------------------------------

    def match_one(self, job: Job) -> list[MatchScore]:
        if self._index is None:
            raise RuntimeError("call index_resumes() first")
        cfg = self.config

        # 1. Vector recall
        q = self.embedder.encode(_job_text(job), dim=cfg.dim).reshape(1, -1)
        hits = self._index.search(q, top_k=cfg.top_n_recall)[0]
        cand_ids = [h.doc_id for h in hits]
        sim_map = {h.doc_id: h.score for h in hits}

        # 2. Cross-encoder rerank (optional)
        rerank_map: dict[str, float] = {}
        if cfg.use_reranker and self.reranker is not None:
            pairs = [(cand_id, _resume_text(self._resume_lookup[cand_id])) for cand_id in cand_ids]
            ranked = self.reranker.rerank(_job_text(job), pairs)
            rerank_map = dict(ranked)
            # re-order cand_ids by rerank score
            cand_ids = [cid for cid, _ in ranked]

        # 3. Bidirectional scoring (always computed for evidence; weight gated)
        results: list[MatchScore] = []
        for cid in cand_ids:
            r = self._resume_lookup[cid]
            bi = bidirectional_score(job, r, alpha=cfg.alpha)
            sim = sim_map.get(cid, 0.0)
            rs = rerank_map.get(cid)
            # Normalize rerank score (cross-encoder logits) via sigmoid-ish clip.
            rs_norm = _sigmoid(rs) if rs is not None else None

            if cfg.use_bidirectional:
                # Convex blend: semantic + rerank + bidirectional
                ws, wr, wb = (
                    cfg.semantic_weight,
                    cfg.rerank_weight if rs_norm is not None else 0.0,
                    1.0 - cfg.semantic_weight - (cfg.rerank_weight if rs_norm is not None else 0.0),
                )
                wb = max(0.0, wb)
                total_w = ws + wr + wb or 1.0
                blended = (
                    ws * _sim_to_unit(sim)
                    + wr * (rs_norm or 0.0)
                    + wb * bi.fused
                ) / total_w
            else:
                # Pure semantic+rerank (no bidirectional)
                if rs_norm is not None:
                    blended = 0.5 * _sim_to_unit(sim) + 0.5 * rs_norm
                else:
                    blended = _sim_to_unit(sim)

            results.append(
                MatchScore(
                    job_id=job.job_id,
                    resume_id=cid,
                    employer_score=bi.employer,
                    candidate_score=bi.candidate,
                    fused_score=float(min(1.0, max(0.0, blended))),
                    semantic_similarity=float(sim),
                    rerank_score=rs,
                    evidence=bi.evidence,
                )
            )
        results.sort(key=lambda x: x.fused_score, reverse=True)

        # 4. Multi-agent stage (optional)
        if cfg.use_multi_agent and self.orchestrator is not None:
            top_m = results[: cfg.top_m_agent]
            top_resumes = [self._resume_lookup[ms.resume_id] for ms in top_m]
            out = self.orchestrator.run(job, top_resumes, top_m)
            # Splice agent-reordered top-M back in front of the tail.
            tail = results[cfg.top_m_agent :]
            results = list(out.final_ranking) + tail

        return results


# ---- helpers --------------------------------------------------------------


def _resume_text(r: Resume) -> str:
    parts = [
        r.summary,
        " ".join(s.name for s in r.skills),
        " ".join(w.title + " " + w.company for w in r.work_history),
        r.raw_text,
    ]
    return "\n".join(p for p in parts if p)


def _job_text(j: Job) -> str:
    parts = [
        j.title,
        " ".join(s.name for s in j.required_skills),
        " ".join(s.name for s in j.preferred_skills),
        j.description,
        j.raw_text,
    ]
    return "\n".join(p for p in parts if p)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _sim_to_unit(s: float) -> float:
    """Map cosine similarity [-1, 1] → [0, 1]."""
    return float(max(0.0, min(1.0, (s + 1.0) / 2.0)))
