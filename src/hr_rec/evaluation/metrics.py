"""Information-retrieval ranking metrics.

* **Precision@K** — fraction of retrieved top-K items that are relevant.
* **Recall@K** — fraction of all relevant items retrieved in top-K.
* **nDCG@K** — normalized discounted cumulative gain, supports graded relevance.
* **MRR** — mean reciprocal rank of the first relevant item.

All functions take ``ranked_ids`` (ordered list of doc IDs) and either
``relevant_ids`` (set) or ``relevance_map`` (doc_id → grade ≥ 0).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ranked_ids:
        return 0.0
    rel = set(relevant_ids)
    top = ranked_ids[:k]
    return sum(1 for d in top if d in rel) / k


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & rel) / len(rel)


def _dcg(grades: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance_map: Mapping[str, float],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ranked_ids or not relevance_map:
        return 0.0
    grades = [float(relevance_map.get(d, 0.0)) for d in ranked_ids[:k]]
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    if not ideal or _dcg(ideal) == 0:
        return 0.0
    return _dcg(grades) / _dcg(ideal)


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]],
    relevant_per_query: Sequence[Iterable[str]],
) -> float:
    if len(rankings) != len(relevant_per_query):
        raise ValueError("rankings and relevant_per_query length mismatch")
    if not rankings:
        return 0.0
    rr_sum = 0.0
    for ranked, rels in zip(rankings, relevant_per_query, strict=True):
        rel_set = set(rels)
        for i, d in enumerate(ranked):
            if d in rel_set:
                rr_sum += 1.0 / (i + 1)
                break
    return rr_sum / len(rankings)
