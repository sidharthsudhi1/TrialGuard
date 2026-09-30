"""R1 learning-to-rank: fusion parity, recall, and a ranker that can learn."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from trialguard.eval.ltr import _fuse, _ranks, recall_at, score_fold, train  # noqa: E402
from trialguard.retrieval.fusion import rrf  # noqa: E402


def test_ranks_are_one_based_descending():
    assert _ranks(np.array([0.1, 0.9, 0.5])).tolist() == [3, 1, 2]


def test_fuse_matches_rrf_scores_on_untied_lists():
    rng = np.random.default_rng(0)
    a, b = rng.random(30), rng.random(30)
    ids = [f"t{i}" for i in range(30)]
    ra, rb = _ranks(a), _ranks(b)
    mine = _fuse([ra, rb], pool=10, weights=[1.0, 0.5])
    lists = [
        [(ids[i], 0.0) for i in np.argsort(-a)[:10]],
        [(ids[i], 0.0) for i in np.argsort(-b)[:10]],
    ]
    ref = dict(rrf(lists, top_k=30, weights=[1.0, 0.5]))
    for i, n in enumerate(ids):
        assert mine[i] == pytest.approx(ref.get(n, 0.0))


def test_recall_at_counts_gold_in_prefix():
    assert recall_at(["a", "b", "c", "d"], ["b", "d"], 2) == 0.5


def _synthetic(n_patients, seed):
    """Feature 1 carries the label; the served order ignores it."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        n = 60
        y = (rng.random(n) < 0.15).astype(int) * 2
        X = np.column_stack([rng.random(n), y + rng.normal(0, 0.3, n)]).astype(np.float32)
        cands = [f"p{p}t{i}" for i in range(n)]
        rows.append({
            "candidates": cands,
            "served": cands,  # served order is arbitrary w.r.t. the label
            "gold_eligible": [c for c, lab in zip(cands, y, strict=True) if lab == 2] or [cands[0]],
            "X": X,
            "y": y,
        })
    return rows


def test_ranker_learns_a_signal_the_served_order_lacks():
    model = train(_synthetic(40, 1), [0.0, 1.0, 3.0])
    per = score_fold(model, _synthetic(20, 2), depths=(10,))
    assert np.mean(per["ltr@10"]) > np.mean(per["served@10"]) + 0.2
