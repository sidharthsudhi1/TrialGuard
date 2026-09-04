"""Reciprocal rank fusion (RRF) over multiple ranked lists."""

from __future__ import annotations

import os


def keyword_decay_enabled() -> bool:
    """Weight each keyword's lists by its importance rank. On.

    The extraction prompt asks the model to order keywords most-to-least
    important, and fusion used to discard that: all 24 lists (12 keywords x 2
    backends) counted equally. Weighting by 1/i is TrialGPT's published
    aggregation (Jin et al., Nat Commun 2024) and is measured significant here on
    both cohorts of record at the depths that matter:

        TREC 2021  recall@100 0.3374 -> 0.3546 (p=0.0386)
                   recall@200 0.4951 -> 0.5250 (p=0.0014)
        TREC 2022  recall@100 0.3920 -> 0.4153 (p=0.0025)
                   recall@200 0.5588 -> 0.5826 (p=0.0063)

    R3 tested this and did not adopt it, having measured only recall@50 — where
    the effect is weakest (p=0.2301 on TREC 2021) and where, before H1, the
    cohort pool was assumed to stop. H1 then showed the agent converts a deep
    pool at a flat rate, which moved the metric that matters to recall@100-200.
    Same lever, different depth, opposite verdict. See
    data/reports/keyword_decay_findings.md.

    Set TG_KEYWORD_DECAY=0 to reproduce any ranking committed before this.
    """
    return os.environ.get("TG_KEYWORD_DECAY", "1") == "1"


def importance_weights(n_lists: int, lists_per_query: int = 2) -> list[float] | None:
    """1/i weight per ranked list, i being its keyword's 1-based importance rank.

    Lists arrive grouped by keyword — (dense, lexical) for keyword 1, then for
    keyword 2, and so on — so the keyword index is the list index floor-divided
    by the number of backends. Returns None when weighting is off or there is
    only one query, where every scheme is identical and the extra work would
    only risk changing a committed number.
    """
    if not keyword_decay_enabled() or n_lists <= lists_per_query:
        return None
    return [1.0 / (i // lists_per_query + 1) for i in range(n_lists)]


def rrf(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 20,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists with RRF.

    score(d) = sum(w_i / (k + rank_i(d))) across all lists, w_i = 1 without
    weights. rank is 1-based.
    """
    scores: dict[str, float] = {}
    for idx, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[idx]
        for rank, (nct_id, _) in enumerate(ranking, start=1):
            scores[nct_id] = scores.get(nct_id, 0.0) + w / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]
