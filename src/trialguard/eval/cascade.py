"""R4 budget sizing. The formulas are the 2026-11-16 amendment, not a result."""

from __future__ import annotations

TAIL_BUDGET_USD = 2.70
TAIL_SAFETY = 1.5
FULL_TAIL = 400
GAIN_FLOOR_USD = 0.25
DROP_LIMIT = 0.05


def tail_depth(
    call_usd: float, n_patients: int, budget: float = TAIL_BUDGET_USD
) -> int:
    """Equal tail depth that fits in `budget` at TAIL_SAFETY times one call."""
    if call_usd <= 0 or n_patients <= 0:
        return 1
    depth = int(budget / (call_usd * TAIL_SAFETY * n_patients))
    return max(1, min(FULL_TAIL, depth))


def seventy_b_count(
    remaining_usd: float, call_usd: float, n_patients: int, depth: int
) -> int:
    """Equal 70B assessments per patient. Zero when the floor or the depth is gone."""
    if remaining_usd < GAIN_FLOOR_USD or call_usd <= 0 or n_patients <= 0 or depth <= 0:
        return 0
    count = int((remaining_usd - 0.05) / (call_usd * 1.3 * n_patients))
    return max(0, min(depth, count))


def drop_rate(rows: list[dict]) -> dict:
    """Share of assessed gold-eligible tail trials the small model tiered excluded."""
    eligible = [r for r in rows if r.get("gold") == "eligible"]
    dropped = [r for r in eligible if r.get("tier") == "excluded"]
    return {
        "eligible": len(eligible),
        "excluded": len(dropped),
        "rate": None if not eligible else round(len(dropped) / len(eligible), 4),
    }


def drop_kills(rate: float | None) -> bool:
    return rate is not None and rate > DROP_LIMIT
