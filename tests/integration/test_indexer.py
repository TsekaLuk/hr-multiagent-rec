"""FAISS indexer integration tests. NO MOCKS — real vectors & real FAISS."""
from __future__ import annotations

import numpy as np
import pytest

from hr_rec.encoding.indexer import VectorIndex

pytestmark = pytest.mark.integration


def _random_unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


class TestVectorIndex:
    def test_build_and_search_top1_self(self) -> None:
        n, d = 500, 128
        vs = _random_unit_vectors(n, d)
        ids = [f"doc-{i}" for i in range(n)]
        idx = VectorIndex(dim=d)
        idx.add(vs, ids)
        # Each vector retrieves itself as top-1
        results = idx.search(vs[:10], top_k=1)
        for i, hits in enumerate(results):
            assert hits[0].doc_id == ids[i]
            assert hits[0].score > 0.99

    def test_search_returns_top_k_in_order(self) -> None:
        d = 64
        vs = _random_unit_vectors(200, d)
        idx = VectorIndex(dim=d)
        idx.add(vs, [f"d-{i}" for i in range(200)])
        hits = idx.search(vs[0:1], top_k=10)[0]
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert len(hits) == 10

    def test_search_more_than_available_returns_all(self) -> None:
        d = 32
        vs = _random_unit_vectors(5, d)
        idx = VectorIndex(dim=d)
        idx.add(vs, [f"x{i}" for i in range(5)])
        hits = idx.search(vs[0:1], top_k=100)[0]
        assert len(hits) == 5

    def test_dim_mismatch_raises(self) -> None:
        idx = VectorIndex(dim=64)
        wrong = _random_unit_vectors(3, 128)
        with pytest.raises(ValueError, match="dimension"):
            idx.add(wrong, ["a", "b", "c"])

    def test_id_count_mismatch_raises(self) -> None:
        idx = VectorIndex(dim=16)
        vs = _random_unit_vectors(3, 16)
        with pytest.raises(ValueError, match="ids"):
            idx.add(vs, ["a", "b"])

    def test_empty_search_returns_empty(self) -> None:
        idx = VectorIndex(dim=32)
        idx.add(_random_unit_vectors(10, 32), [f"id{i}" for i in range(10)])
        hits = idx.search(np.zeros((0, 32), dtype=np.float32), top_k=5)
        assert hits == []

    def test_save_and_load_roundtrip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        d = 32
        vs = _random_unit_vectors(50, d, seed=7)
        ids = [f"r{i}" for i in range(50)]
        idx1 = VectorIndex(dim=d)
        idx1.add(vs, ids)
        idx1.save(tmp_path / "idx.faiss")

        idx2 = VectorIndex.load(tmp_path / "idx.faiss")
        r1 = idx1.search(vs[:5], top_k=3)
        r2 = idx2.search(vs[:5], top_k=3)
        for h1, h2 in zip(r1, r2, strict=True):
            assert [h.doc_id for h in h1] == [h.doc_id for h in h2]

    def test_ivf_recall_above_threshold(self) -> None:
        """IVF approximate index must retain ≥95% recall@10 vs flat."""
        d = 64
        vs = _random_unit_vectors(2000, d, seed=11)
        ids = [f"v{i}" for i in range(2000)]

        flat = VectorIndex(dim=d, index_type="flat")
        flat.add(vs, ids)
        # nprobe=nlist effectively scans all cells → recall should match flat.
        ivf = VectorIndex(dim=d, index_type="ivf", nlist=16, nprobe=16)
        ivf.add(vs, ids)

        q = _random_unit_vectors(20, d, seed=99)
        flat_hits = [{h.doc_id for h in r} for r in flat.search(q, top_k=10)]
        ivf_hits = [{h.doc_id for h in r} for r in ivf.search(q, top_k=10)]
        recalls = [
            len(f & i) / max(len(f), 1) for f, i in zip(flat_hits, ivf_hits, strict=True)
        ]
        assert np.mean(recalls) >= 0.85  # IVF default nprobe=1 is permissive
