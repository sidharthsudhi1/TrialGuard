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


DEEP_TOP_K = 100


def list_pool(top_k: int) -> int:
    """Depth of each per-keyword ranked list fed to RRF: 200 for deep pools, else 50.

    With 50-deep lists a trial scores only by reaching the top 50 of some single
    keyword's list, so a trial ranked moderately for many keywords -- the shape
    of an eligible one -- scores nothing. 200-deep lists recover those, which
    moves ranks 100-500 but not the head (R1/R1b, $0, cached keywords):

        TREC 2021  @100 +0.018  @200 +0.073  @500 +0.113   (p_bh < 0.001)
        TREC 2022  @100 +0.022  @200 +0.067  @500 +0.088   (p_bh < 0.03)
        @5/@10/@25 on TREC 2021, TREC 2022 and SIGIR: no gain, none significant,
        SIGIR @25 -0.032 (CI excludes 0, p_bh 0.19)

    So depth follows the request: the demo's 5-25 results keep 50-deep lists and
    anything asking for 100+ gets 200. It pays only because of keyword decay;
    before 1/i weighting deep lists diluted recall (structural_recall_plan §5).
    See data/reports/r1b_findings.md.

    TG_LIST_POOL=<n> pins every list to n; TG_LIST_POOL=50 reproduces any ranking
    committed before 2026-09-30.
    """
    pinned = os.environ.get("TG_LIST_POOL")
    if pinned:
        return int(pinned)
    return 200 if top_k >= DEEP_TOP_K else 50


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
