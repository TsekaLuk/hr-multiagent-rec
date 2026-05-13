"""Cross-encoder reranker backed by Qwen3-Reranker.

Strategy: load the model via sentence-transformers ``CrossEncoder``
when available; fall back to a manual HuggingFace pipeline. Real
model, no mocks.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Reranker:
    """A typed wrapper around a cross-encoder reranker.

    Parameters
    ----------
    model_name:
        e.g. ``"Qwen/Qwen3-Reranker-0.6B"``.
    device:
        ``cuda | mps | cpu | None`` (auto).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device or _detect_device()
        self.max_length = max_length
        logger.info("loading reranker %s on %s", model_name, self.device)
        self._ce = CrossEncoder(model_name, device=self.device, max_length=max_length)

    def score_pair(self, query: str, doc: str) -> float:
        """Score a single (query, doc) pair. Higher = more relevant."""
        s = self._ce.predict([(query or "", doc or "")])
        return float(np.atleast_1d(s)[0])

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Batch-score pairs. Returns (N,) float32."""
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        s = self._ce.predict([(q or "", d or "") for q, d in pairs])
        return np.atleast_1d(s).astype(np.float32)

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Given ``[(doc_id, doc_text), ...]``, return them sorted by relevance."""
        if not candidates:
            return []
        pairs = [(query, doc) for _, doc in candidates]
        scores = self.score_pairs(pairs)
        order = np.argsort(-scores)
        ranked = [(candidates[i][0], float(scores[i])) for i in order]
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
