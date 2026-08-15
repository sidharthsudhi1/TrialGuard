"""Regenerate README faithfulness numbers from committed JSON artifacts.

Prints the verifier stress counts and Phase 8 / Phase 9v4 headline metrics so
README claims can be cross-checked against data/reports/*.json. Does not rewrite
README automatically — paste deliberately after review.

  python -m trialguard.scripts.sync_readme_metrics
"""

from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path("data/reports")


def _load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def main() -> None:
    stress = _load("verifier_stress.json")
    print("=== verifier_stress.json ===")
    print(
        f"catch {stress['n_corrupted']}/{stress['n_corrupted']} "
        f"(genuine={stress['n_genuine']}, "
        f"false_rejection={stress['verifier_false_rejection_rate']})"
    )

    for name in (
        "phase8di_agent_sigir.json",
        "phase8v2_agent_sigir.json",
        "phase9v4_agent_sigir.json",
        "phase9v4_agent_trec_2021.json",
    ):
        path = REPORTS / name
        if not path.exists():
            print(f"\n=== {name} (missing) ===")
            continue
        r = _load(name)
        v = r["verified"]
        s = r.get("significance", {})
        print(f"\n=== {name} ===")
        print(
            f"trial_accuracy={v.get('trial_accuracy')}  "
            f"coverage={v.get('coverage')}  "
            f"citation_precision={v.get('citation_precision')}  "
            f"abstention={v.get('abstention_rate')}"
        )
        for kind, c in (v.get("by_kind") or {}).items():
            print(
                f"  {kind}: n={c.get('n_criteria')}  "
                f"unsupported={c.get('unsupported_verdict_rate')}  "
                f"precision={c.get('citation_precision')}"
            )
        if s:
            print(
                f"matched={s.get('matched_trials')}  "
                f"rel_change={s.get('relative_change')}  "
                f"fisher_p={s.get('fisher_p')}"
            )


if __name__ == "__main__":
    main()
