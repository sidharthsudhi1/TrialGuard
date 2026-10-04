"""Topic-level inference: bootstrap CIs, paired Wilcoxon, BH, power."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from trialguard.eval.significance import (
    bh_adjust,
    bootstrap_ci,
    compare_family,
    min_detectable_effect,
    paired_bootstrap_ci,
    ratio_of_sums,
    required_n,
    wilcoxon_paired,
)


def test_ratio_of_sums_pools_rather_than_averages():
    # 1/1 and 0/9: mean of rates is 0.5, pooled rate is 0.1
    stat = ratio_of_sums([1, 0], [1, 9])
    assert stat(np.arange(2)) == pytest.approx(0.1)


def test_bootstrap_ci_brackets_the_point_and_is_seeded():
    rng = np.random.default_rng(1)
    num = rng.integers(0, 10, 40)
    den = num + rng.integers(1, 10, 40)
    a = bootstrap_ci(ratio_of_sums(num, den), 40, seed=7)
    b = bootstrap_ci(ratio_of_sums(num, den), 40, seed=7)
    assert a == b
    assert a["ci"][0] <= a["point"] <= a["ci"][1]


def test_undefined_statistic_reports_none_not_nan():
    out = bootstrap_ci(ratio_of_sums([0, 0, 0], [0, 0, 0]), 3)
    assert out["point"] is None
    assert out["ci"] == [None, None]
    assert bootstrap_ci(ratio_of_sums([], []), 0)["point"] is None


def test_paired_ci_excludes_zero_for_a_consistent_gain():
    a = np.linspace(0.1, 0.5, 30)
    out = paired_bootstrap_ci(a, a + 0.05)
    assert out["ci"][0] > 0


def test_wilcoxon_counts_direction_and_all_ties_is_not_an_error():
    out = wilcoxon_paired([1, 2, 3, 4], [2, 2, 1, 5])
    assert (out["better"], out["worse"], out["ties"]) == (2, 1, 1)
    assert wilcoxon_paired([1, 1], [1, 1])["p"] == 1.0


def test_bh_matches_scipy():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5, "e": 0.002}
    ours = bh_adjust(p)
    names = list(p)
    ref = false_discovery_control([p[n] for n in names], method="bh")
    for n, r in zip(names, ref, strict=True):
        assert ours[n] == pytest.approx(r, abs=1e-4)


def test_family_adjustment_can_revoke_a_nominal_win():
    rng = np.random.default_rng(3)
    base = rng.normal(0.3, 0.1, 25)
    family = {f"arm{i}": (base, base + rng.normal(0, 0.05, 25)) for i in range(10)}
    out = compare_family(family)
    assert all(r["p_bh"] >= r["p"] for r in out.values())
    assert all(len(r["delta_ci"]) == 2 for r in out.values())


def test_power_helpers_round_trip():
    deltas = np.random.default_rng(5).normal(0.02, 0.1, 20)
    mde = min_detectable_effect(deltas)
    assert mde > 0
    assert required_n(float(np.std(deltas, ddof=1)), mde) == pytest.approx(20, abs=1)
