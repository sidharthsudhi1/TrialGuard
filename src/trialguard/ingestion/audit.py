"""The audit step of the refresh: gates a pull must pass before it is published.

Write-Audit-Publish. The pull is complete by construction (ctgov.pull_trials
raises otherwise), so these gates are about whether it is *sane*: a CT.gov field
rename would otherwise publish a corpus of empty criteria, and an embedding
config leaking into the refresh environment would mix vector spaces.

Two classes:

- hard gates fail the run whatever the mode. They catch states that are never
  legitimate: malformed ids, out-of-scope statuses, a mismatched embedding
  config, or a content field missing from a large share of the pull.
- calibrated gates have thresholds that need real baselines. They only log
  until TG_REFRESH_GATES=enforce, so two weeks of refresh_runs rows can show
  where the thresholds belong before they are allowed to block.

`--allow NAME` overrides one named gate for one run, recorded in the ledger.
embed_tag_match cannot be overridden: a model or config change is a full
re-index, not a refresh.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass

NCT_RE = re.compile(r"^NCT\d{8}$")

UNOVERRIDABLE = frozenset({"embed_tag_match"})

# A field missing from over half the pull is a schema change, not sparse data.
HARD_MISSING_SHARE = 0.5


@dataclass
class Gate:
    name: str
    value: float | int | str | None
    threshold: str
    passed: bool
    hard: bool = False
    overridden: bool = False

    @property
    def blocks(self) -> bool:
        if self.passed or self.overridden:
            return False
        return self.hard or enforcing()


def enforcing() -> bool:
    return os.environ.get("TG_REFRESH_GATES", "log").lower() == "enforce"


@dataclass
class AuditInput:
    fetched: int
    total_count: int
    active: int
    to_embed: int
    to_expire: int
    field_missing: dict[str, int]
    statuses: list[str]
    requested_statuses: list[str]
    ids: list[str]
    mean_criteria: float
    current_tag: str
    corpus_tag: str | None
    bootstrap: bool = False
    previous: dict | None = None


def _share(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def audit(a: AuditInput, allow: frozenset[str] = frozenset()) -> list[Gate]:
    gates: list[Gate] = []

    def add(name, value, threshold, passed, hard=False):
        gates.append(Gate(name, value, threshold, bool(passed), hard))

    add("completeness", a.fetched, f"== {a.total_count}", a.fetched == a.total_count, hard=True)

    bad_ids = sum(1 for n in a.ids if not NCT_RE.match(n))
    add("id_format", bad_ids, "== 0", bad_ids == 0, hard=True)

    requested = set(a.requested_statuses)
    off_scope = sum(1 for s in a.statuses if s not in requested)
    add("status_in_scope", off_scope, "== 0", off_scope == 0, hard=True)

    # 'unknown' rows are the migration's way of saying "re-embed me", so they do
    # not vote; an empty corpus has nothing to disagree with.
    add(
        "embed_tag_match",
        a.corpus_tag,
        f"== {a.current_tag}",
        a.corpus_tag in (None, a.current_tag),
        hard=True,
    )

    for field, n in sorted(a.field_missing.items()):
        share = _share(n, a.fetched)
        add(f"missing_{field}_hard", share, f"<= {HARD_MISSING_SHARE}",
            share <= HARD_MISSING_SHARE, hard=True)

    # --- calibrated -----------------------------------------------------------
    # 0 of 26,107 live trials had an empty eligibilityCriteria on 2026-09-23.
    add("empty_eligibility", _share(a.field_missing.get("eligibility_raw", 0), a.fetched),
        "<= 0.01", _share(a.field_missing.get("eligibility_raw", 0), a.fetched) <= 0.01)
    add("empty_last_updated", _share(a.field_missing.get("last_updated", 0), a.fetched),
        "<= 0.005", _share(a.field_missing.get("last_updated", 0), a.fetched) <= 0.005)

    if not a.bootstrap and a.active:
        add("min_size", _share(a.fetched, a.active), ">= 0.90",
            a.fetched >= 0.90 * a.active)
        expire_cap = max(300, int(0.03 * a.active))
        add("expiry_churn", a.to_expire, f"<= {expire_cap}", a.to_expire <= expire_cap)
        # The first post-deploy cycle measured 2,982 / 25,965 = 11.5%.
        embed_cap = max(1000, int(0.15 * a.active))
        add("reembed_churn", a.to_embed, f"<= {embed_cap}", a.to_embed <= embed_cap)

    prev = a.previous or {}
    prev_mean = prev.get("mean_criteria")
    if prev_mean:
        drift = round(abs(a.mean_criteria - prev_mean) / prev_mean, 4)
        add("mean_criteria", a.mean_criteria, f"within 20% of {prev_mean}", drift <= 0.20)
    for field, share_before in (prev.get("missing_share") or {}).items():
        share = _share(a.field_missing.get(field, 0), a.fetched)
        add(f"missing_{field}_drift", share, f"<= {share_before} + 0.05",
            share <= share_before + 0.05)

    for g in gates:
        if g.name in allow and g.name not in UNOVERRIDABLE:
            g.overridden = True
    return gates


def blocking(gates: list[Gate]) -> list[Gate]:
    return [g for g in gates if g.blocks]


def failed(gates: list[Gate]) -> list[Gate]:
    return [g for g in gates if not g.passed]


def stats(a: AuditInput) -> dict:
    """What the next run compares itself against."""
    return {
        "mean_criteria": round(a.mean_criteria, 3),
        "missing_share": {f: _share(n, a.fetched) for f, n in a.field_missing.items()},
    }


def to_json(gates: list[Gate], a: AuditInput) -> dict:
    return {
        "mode": "enforce" if enforcing() else "log",
        "gates": [asdict(g) for g in gates],
        "stats": stats(a),
    }
