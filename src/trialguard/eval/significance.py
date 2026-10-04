"""Reproducible significance for the faithfulness A/B.

Phase 3 reported Fisher p-values that were computed off-code; this module puts
that computation inside the harness so every reported number is regenerated from
the run, never hand-typed.

The test is run over the MATCHED trial set — trials both arms actually completed
— because the verified arm can stop early on the free-tier daily cap, so the two
arms' trial sets are not identical unless we intersect them. Reporting over the
intersection removes the selection skew that unequal truncation would introduce.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import fisher_exact, norm, wilcoxon


def matched_ab(baseline_per_trial: dict, verified_per_trial: dict) -> dict:
    """Fisher exact on the unsupported-vs-grounded 2x2 over trials both arms ran.

    Each per_trial value is {"decisive": int, "unsupported": int}. Pools criteria
    within the intersection of trial ids. Returns the 2x2, odds ratio, p-value,
    and the two arms' unsupported rates on the matched set.
    """
    matched = sorted(set(baseline_per_trial) & set(verified_per_trial))
    b_dec = b_uns = v_dec = v_uns = 0
    for nct in matched:
        b_dec += baseline_per_trial[nct]["decisive"]
        b_uns += baseline_per_trial[nct]["unsupported"]
        v_dec += verified_per_trial[nct]["decisive"]
        v_uns += verified_per_trial[nct]["unsupported"]
    b_grd = b_dec - b_uns
    v_grd = v_dec - v_uns
    table = [[b_uns, b_grd], [v_uns, v_grd]]
    odds, p = fisher_exact(table)
    return {
        "matched_trials": len(matched),
        "table": {
            "baseline": {"unsupported": b_uns, "grounded": b_grd, "decisive": b_dec},
            "verified": {"unsupported": v_uns, "grounded": v_grd, "decisive": v_dec},
        },
        "baseline_unsupported_rate": round(b_uns / b_dec, 4) if b_dec else 0.0,
        "verified_unsupported_rate": round(v_uns / v_dec, 4) if v_dec else 0.0,
        "relative_change": round((v_uns / v_dec - b_uns / b_dec) / (b_uns / b_dec), 4)
        if b_dec and v_dec and b_uns else 0.0,
        "odds_ratio": round(float(odds), 4),
        "fisher_p": round(float(p), 4),
        "significant_05": bool(p < 0.05),
    }


# --- Topic-level inference (V1) ---------------------------------------------
#
# Every headline outside the faithfulness A/B is a per-patient metric averaged or
# pooled over 20-75 patients, and until now it was reported as a point estimate.
# These helpers put an interval on it and correct for the number of comparisons
# run against the same cohorts. Wilcoxon + Benjamini-Hochberg is the procedure
# Otero, Parapar & Barreiro (ECIR 2025, arXiv:2501.03930) found holds Type I error
# at TREC topic counts with the best power of the options they simulated.

_N_RESAMPLES = 10_000


def _r4(x: float) -> float | None:
    return None if np.isnan(x) else round(float(x), 4)


def bootstrap_ci(
    stat, n_units: int, n_resamples: int = _N_RESAMPLES, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Percentile bootstrap over units (patients). `stat(idx)` scores a resample.

    Resampling patients rather than trials keeps each patient's correlated trials
    together, which is what makes the interval honest for pooled rates. An
    undefined statistic (zero denominator) reports None, never NaN, so the JSON
    report stays valid.
    """
    point = float(stat(np.arange(n_units)))
    if n_units < 2:
        return {"point": _r4(point), "ci": [_r4(point), _r4(point)], "n": n_units}
    rng = np.random.default_rng(seed)
    draws = np.fromiter(
        (stat(rng.integers(0, n_units, n_units)) for _ in range(n_resamples)),
        dtype=float,
        count=n_resamples,
    )
    draws = draws[~np.isnan(draws)]
    if not len(draws):
        return {"point": _r4(point), "ci": [None, None], "n": n_units}
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"point": _r4(point), "ci": [_r4(lo), _r4(hi)], "n": n_units}


def ratio_of_sums(num, den):
    """Statistic for pooled rates: sum(num) / sum(den) over the resampled units."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)

    def stat(idx):
        d = den[idx].sum()
        return num[idx].sum() / d if d else np.nan

    return stat


def paired_bootstrap_ci(a, b, **kw) -> dict:
    """CI on mean(b - a) over paired units."""
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return bootstrap_ci(lambda idx: diff[idx].mean(), len(diff), **kw)


def wilcoxon_paired(a, b) -> dict:
    """Two-sided Wilcoxon signed-rank on paired per-unit scores, b vs a.

    Ties (zero differences) are dropped, the scipy default. When every pair ties
    there is nothing to test and p is 1.0 rather than an exception.
    """
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    better, worse = int((diff > 0).sum()), int((diff < 0).sum())
    p = float(wilcoxon(diff).pvalue) if better + worse else 1.0
    return {
        "n": len(diff),
        "mean_delta": round(float(diff.mean()), 4) if len(diff) else 0.0,
        "better": better,
        "worse": worse,
        "ties": len(diff) - better - worse,
        "p": round(p, 4),
    }


def bh_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg adjusted p-values, keyed like the input."""
    names = sorted(pvalues, key=lambda name: pvalues[name])
    m = len(names)
    adjusted, running = {}, 1.0
    for rank in range(m, 0, -1):
        name = names[rank - 1]
        running = min(running, pvalues[name] * m / rank)
        adjusted[name] = round(min(1.0, running), 4)
    return adjusted


def compare_family(family: dict[str, tuple], alpha: float = 0.05, seed: int = 0) -> dict:
    """Wilcoxon + BH across one experiment family, with a bootstrap CI per delta.

    `family` maps comparison name -> (baseline_scores, candidate_scores), paired by
    unit. A family is every comparison an experiment reports against one cohort;
    adjusting within it is what keeps the tenth arm from winning by chance.
    """
    tests = {name: wilcoxon_paired(a, b) for name, (a, b) in family.items()}
    adj = bh_adjust({name: t["p"] for name, t in tests.items()})
    return {
        name: {
            **t,
            "delta_ci": paired_bootstrap_ci(*family[name], seed=seed)["ci"],
            "p_bh": adj[name],
            "significant_bh": adj[name] < alpha,
        }
        for name, t in tests.items()
    }


def min_detectable_effect(deltas, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest mean paired delta detectable at this n, from pilot per-unit deltas.

    Normal approximation to the paired t-test. Wilcoxon's asymptotic efficiency is
    ~0.955 of t under normality, so read this as a slight underestimate for it.
    Use it before a run: if the effect you hope for is below it, add patients.
    """
    d = np.asarray(deltas, dtype=float)
    if len(d) < 2:
        return float("nan")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return round(float(z * d.std(ddof=1) / np.sqrt(len(d))), 4)


def required_n(sd: float, effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Units needed to detect a mean paired delta `effect` given per-unit delta sd."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil((z * sd / effect) ** 2))
