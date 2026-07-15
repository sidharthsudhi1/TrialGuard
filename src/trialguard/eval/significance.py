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

from scipy.stats import fisher_exact


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
