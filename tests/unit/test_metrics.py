"""Information-retrieval metrics tests — covers the standard edge cases."""
from __future__ import annotations

import math

import pytest

from hr_rec.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

pytestmark = pytest.mark.unit


# ---------- Precision -----------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], set(), k=3) == 0.0

    def test_partial(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "x"}, k=3) == pytest.approx(1 / 3)

    def test_k_larger_than_list(self) -> None:
        # k=10 but only 3 retrieved; denom still divides by k
        assert precision_at_k(["a", "b", "c"], {"a"}, k=10) == pytest.approx(1 / 10)

    def test_empty_ranked(self) -> None:
        assert precision_at_k([], {"a"}, k=5) == 0.0

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError):
            precision_at_k(["a"], {"a"}, k=0)


# ---------- Recall --------------------------------------------------------


class TestRecallAtK:
    def test_all_recovered(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_partial_recovered(self) -> None:
        assert recall_at_k(["a", "b", "x"], {"a", "b", "c", "d"}, k=3) == 0.5

    def test_no_relevant_returns_zero(self) -> None:
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0

    def test_k_smaller_than_relevant_set(self) -> None:
        # 1 of 5 relevant in top-1
        assert recall_at_k(["a"], {"a", "b", "c", "d", "e"}, k=1) == 0.2


# ---------- nDCG ----------------------------------------------------------


class TestNdcgAtK:
    def test_perfect_ordering(self) -> None:
        rel = {"a": 3, "b": 2, "c": 1}
        assert ndcg_at_k(["a", "b", "c"], rel, k=3) == 1.0

    def test_reversed_is_lower(self) -> None:
        rel = {"a": 3, "b": 2, "c": 1}
        perfect = ndcg_at_k(["a", "b", "c"], rel, k=3)
        reversed_ = ndcg_at_k(["c", "b", "a"], rel, k=3)
        assert reversed_ < perfect

    def test_irrelevant_doc_in_results(self) -> None:
        rel = {"a": 3, "b": 2}
        v = ndcg_at_k(["a", "x", "b"], rel, k=3)
        assert 0.0 < v < 1.0

    def test_empty_ranked(self) -> None:
        assert ndcg_at_k([], {"a": 1}, k=3) == 0.0

    def test_empty_relevance(self) -> None:
        assert ndcg_at_k(["a", "b"], {}, k=2) == 0.0

    def test_graded_dcg_formula(self) -> None:
        """Sanity check the DCG formula directly: gain / log2(rank+1)."""
        rel = {"a": 2}
        v = ndcg_at_k(["a"], rel, k=1)
        # DCG = 2/log2(2)=2; iDCG=2/log2(2)=2; nDCG = 1.0
        assert v == 1.0
        v2 = ndcg_at_k(["x", "a"], rel, k=2)
        # DCG = 0 + 2/log2(3); iDCG = 2/log2(2) = 2
        expected = (2 / math.log2(3)) / 2
        assert v2 == pytest.approx(expected, rel=1e-6)


# ---------- MRR -----------------------------------------------------------


class TestMRR:
    def test_first_hit_rank_one(self) -> None:
        assert mean_reciprocal_rank([["a", "b"]], [["a"]]) == 1.0

    def test_first_hit_rank_two(self) -> None:
        assert mean_reciprocal_rank([["x", "a"]], [["a"]]) == 0.5

    def test_no_hit(self) -> None:
        assert mean_reciprocal_rank([["x", "y"]], [["a"]]) == 0.0

    def test_multi_query_average(self) -> None:
        # q1: rank 1 → 1.0
        # q2: rank 3 → 1/3
        # mean = (1 + 1/3) / 2 = 2/3
        mrr = mean_reciprocal_rank([["a"], ["x", "y", "b"]], [["a"], ["b"]])
        assert mrr == pytest.approx((1.0 + 1 / 3) / 2)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            mean_reciprocal_rank([["a"]], [["a"], ["b"]])
