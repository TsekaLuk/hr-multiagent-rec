"""FAISS-CPU vector index — flat or IVF.

NormalizeL2 + Inner-Product == cosine similarity. We assume callers
pass L2-normalized vectors from :class:`Embedder`.
"""
from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import faiss
import numpy as np


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float


IndexType = Literal["flat", "ivf"]


class VectorIndex:
    """A small, typed wrapper around faiss-cpu."""

    def __init__(
        self,
        dim: int,
        *,
        index_type: IndexType = "flat",
        nlist: int = 32,
        nprobe: int | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        # Default nprobe to half of nlist for good recall/latency trade-off.
        self.nprobe = nprobe if nprobe is not None else max(1, nlist // 2)
        self._ids: list[str] = []
        if index_type == "flat":
            self._index: faiss.Index = faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"unknown index_type: {index_type}")

    # ---- mutating -------------------------------------------------------

    def add(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"dimension mismatch: index={self.dim} got={vectors.shape[1]}"
            )
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids count {len(ids)} ≠ vectors count {vectors.shape[0]}"
            )
        vs = np.ascontiguousarray(vectors.astype(np.float32))
        if not self._index.is_trained:
            self._index.train(vs)
        self._index.add(vs)
        if self.index_type == "ivf":
            self._index.nprobe = self.nprobe
        self._ids.extend(ids)

    # ---- query ----------------------------------------------------------

    def search(self, queries: np.ndarray, top_k: int = 10) -> list[list[SearchHit]]:
        if queries.size == 0:
            return []
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"queries shape mismatch (expect (*, {self.dim}))")
        k = min(top_k, len(self._ids))
        qs = np.ascontiguousarray(queries.astype(np.float32))
        # Ensure nprobe is applied right before each search.
        if self.index_type == "ivf":
            faiss.ParameterSpace().set_index_parameter(self._index, "nprobe", self.nprobe)
        scores, idxs = self._index.search(qs, k)
        out: list[list[SearchHit]] = []
        for row_scores, row_idxs in zip(scores, idxs, strict=True):
            row: list[SearchHit] = []
            for s, i in zip(row_scores, row_idxs, strict=True):
                if i < 0 or i >= len(self._ids):
                    continue
                row.append(SearchHit(doc_id=self._ids[i], score=float(s)))
            out.append(row)
        return out

    # ---- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        faiss.write_index(self._index, str(p))
        with open(p.with_suffix(p.suffix + ".meta"), "wb") as f:
            pickle.dump(
                {
                    "ids": self._ids,
                    "dim": self.dim,
                    "index_type": self.index_type,
                    "nlist": self.nlist,
                    "nprobe": self.nprobe,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        p = Path(path)
        with open(p.with_suffix(p.suffix + ".meta"), "rb") as f:
            meta = pickle.load(f)
        inst = cls(
            dim=meta["dim"],
            index_type=meta["index_type"],
            nlist=meta["nlist"],
            nprobe=meta["nprobe"],
        )
        inst._index = faiss.read_index(str(p))
        inst._ids = list(meta["ids"])
        return inst

    def __len__(self) -> int:
        return len(self._ids)
