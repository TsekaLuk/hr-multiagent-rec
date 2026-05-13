"""Integration tests for the real Qwen3-Embedding encoder. NO MOCKS.

These hit the actual model. Long-tail business cases included:
empty strings, very long Chinese text, emoji, mixed CJK+ASCII,
batch vs single consistency, MRL truncation, semantic monotonicity.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ---------- Basic shape & dtype -------------------------------------------


def test_single_encoding_shape(embedder: object) -> None:
    v = embedder.encode("机器学习工程师")
    assert v.ndim == 1
    assert v.shape[0] >= 256  # Qwen3-Emb-0.6B default ≥ 1024
    assert v.dtype == np.float32 or v.dtype == np.float64


def test_batch_encoding_shape(embedder: object) -> None:
    vs = embedder.encode_batch(["Python 工程师", "数据分析师", "前端开发"])
    assert vs.ndim == 2
    assert vs.shape[0] == 3
    assert vs.shape[1] == embedder.dim


def test_normalized_unit_norm(embedder: object) -> None:
    """Encoder must L2-normalize by default for cosine retrieval."""
    v = embedder.encode("test")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-3


# ---------- Determinism ----------------------------------------------------


def test_determinism_single(embedder: object) -> None:
    v1 = embedder.encode("机器学习")
    v2 = embedder.encode("机器学习")
    np.testing.assert_allclose(v1, v2, atol=1e-5)


def test_batch_equals_single(embedder: object) -> None:
    """Encoding the same texts in batch vs one-by-one must match."""
    texts = ["分布式系统", "深度学习模型", "云原生架构"]
    batch = embedder.encode_batch(texts)
    singles = np.stack([embedder.encode(t) for t in texts])
    np.testing.assert_allclose(batch, singles, atol=1e-4)


# ---------- Semantic monotonicity -----------------------------------------


def test_paraphrase_more_similar_than_unrelated(embedder: object) -> None:
    """Cosine(机器学习, 深度学习) > cosine(机器学习, 红烧肉)."""
    ml = embedder.encode("机器学习算法工程师")
    dl = embedder.encode("深度学习算法工程师")
    food = embedder.encode("红烧肉做法")
    cos_ml_dl = float(ml @ dl)
    cos_ml_food = float(ml @ food)
    assert cos_ml_dl > cos_ml_food + 0.1


def test_self_similarity_is_one(embedder: object) -> None:
    v = embedder.encode("Python 后端工程师")
    assert abs(float(v @ v) - 1.0) < 1e-3


# ---------- MRL (Matryoshka) truncation -----------------------------------


@pytest.mark.parametrize("target_dim", [128, 256, 512, 768])
def test_mrl_truncation_preserves_relative_order(
    embedder: object, target_dim: int
) -> None:
    """Truncated embeddings must keep monotonicity of similarity."""
    if target_dim > embedder.dim:
        pytest.skip(f"target {target_dim} > native {embedder.dim}")
    ml = embedder.encode("机器学习", dim=target_dim)
    dl = embedder.encode("深度学习", dim=target_dim)
    food = embedder.encode("红烧肉", dim=target_dim)
    assert ml.shape[0] == target_dim
    # cosine on truncated+renormalized vectors
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos(ml, dl) > cos(ml, food)


# ---------- Edge cases / long-tail ----------------------------------------


def test_empty_string_does_not_crash(embedder: object) -> None:
    v = embedder.encode("")
    assert v.shape[0] == embedder.dim
    assert np.isfinite(v).all()


def test_whitespace_only(embedder: object) -> None:
    v = embedder.encode("   \n\t   ")
    assert np.isfinite(v).all()


def test_emoji_and_mixed_unicode(embedder: object) -> None:
    v = embedder.encode("精通 Python 🐍 与 分布式系统 🚀")
    assert np.isfinite(v).all()
    assert v.shape[0] == embedder.dim


def test_very_long_chinese_text(embedder: object) -> None:
    """8K-char resume should not raise (gets truncated to model max)."""
    long = ("我是一名经验丰富的软件工程师。" * 500)
    v = embedder.encode(long)
    assert np.isfinite(v).all()


def test_pure_punctuation(embedder: object) -> None:
    v = embedder.encode("。。。！！！？？？")
    assert np.isfinite(v).all()


def test_mixed_chinese_english(embedder: object) -> None:
    a = embedder.encode("Senior Backend Engineer 高级后端工程师")
    b = embedder.encode("高级后端工程师")
    # Bilingual variant should be highly similar to the Chinese-only one
    assert float(a @ b) > 0.6
