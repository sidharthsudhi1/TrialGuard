"""Reciprocal rank fusion (RRF) over multiple ranked lists."""

from __future__ import annotations


def rrf(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Fuse ranked lists with RRF.

    score(d) = sum(1 / (k + rank_i(d))) across all lists.
    rank is 1-based.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (nct_id, _) in enumerate(ranking, start=1):
            scores[nct_id] = scores.get(nct_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]
