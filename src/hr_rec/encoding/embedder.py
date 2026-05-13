"""Semantic encoder backed by Qwen3-Embedding (or any Sentence-Transformers model).

Real model, no mocks. Supports MRL dimension truncation as documented
in the Qwen3-Embedding model card.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """A thin, typed wrapper around a Sentence-Transformers model.

    Parameters
    ----------
    model_name:
        HuggingFace model id, e.g. ``"Qwen/Qwen3-Embedding-0.6B"``.
    device:
        ``"cuda" | "mps" | "cpu" | None`` (auto-detect).
    normalize:
        L2-normalize outputs (default True — required for cosine retrieval).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        device: str | None = None,
        normalize: bool = True,
        max_seq_length: int = 512,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # local import keeps cold-start cheap

        self.model_name = model_name
        self.device = device or _detect_device()
        self.normalize = normalize
        logger.info("loading embedding model %s on %s", model_name, self.device)
        self._st = SentenceTransformer(model_name, device=self.device)
        try:
            self._st.max_seq_length = max_seq_length
        except Exception:
            pass
        # Native embedding dimension
        self._dim: int = int(self._st.get_sentence_embedding_dimension() or 0)
        if self._dim <= 0:
            # fall back to a probe
            probe = self._st.encode(["probe"], convert_to_numpy=True)
            self._dim = int(probe.shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    # ---- public API -----------------------------------------------------

    def encode(self, text: str, *, dim: int | None = None) -> np.ndarray:
        """Encode a single string. Returns a (D,) float32 vector."""
        v = self._st.encode(
            [text or ""],  # empty-safe
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )[0].astype(np.float32)
        return self._post(v, dim=dim)

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        dim: int | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Batch encode. Returns a (N, D) float32 matrix."""
        cleaned = [t or "" for t in texts]
        m = self._st.encode(
            cleaned,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)
        if dim is not None:
            m = self._truncate_matrix(m, dim)
        if self.normalize:
            m = _l2_normalize_matrix(m)
        return m

    # ---- internals ------------------------------------------------------

    def _post(self, v: np.ndarray, *, dim: int | None) -> np.ndarray:
        if dim is not None:
            v = v[:dim]
        if self.normalize:
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
        return v.astype(np.float32)

    @staticmethod
    def _truncate_matrix(m: np.ndarray, dim: int) -> np.ndarray:
        return m[:, :dim]


def _l2_normalize_matrix(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)
