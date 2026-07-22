"""Phase 5 CI regression gate — fails the build when faithfulness regresses.

Runs on committed artifacts only, so it works from a clean checkout with no Groq
calls and no analyst cache (data/cache is gitignored): a deterministic grounding
stress test over a checked-in golden fixture, plus threshold checks against a
committed A/B report. Thresholds live in data/reports/baselines.json.

CLI: `python -m trialguard.eval.regression_gate` — prints a pass/fail table and
exits non-zero if any gate trips. Wired into .github/workflows/ci.yml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from trialguard.verify.grounding import is_grounded, normalize

FIXTURE = Path("tests/fixtures/grounding_golden.json")
BASELINES = Path("data/reports/baselines.json")

# short function/stop words carry no clinical signal; corrupting them would be a
# weak test, and some (e.g. "of") are ubiquitous enough to reappear by chance.
_STOP = {"the", "and", "for", "with", "must", "have", "per", "was", "are", "who", "any"}


def _content_tokens(quote: str) -> list[str]:
    return [t for t in normalize(quote).split() if len(t) >= 3 and t not in _STOP]


def _sentinel(source_norm: str, i: int) -> str:
    """A token guaranteed absent from the source, so swapping it in must break
    verbatim grounding."""
    tok = f"zzq{i}xq"
    while tok in source_norm:
        tok += "q"
    return tok


def stress_test(cases: list[dict]) -> dict:
    """Catch-rate over corrupted quotes + false-rejection-rate over genuine ones.

    For every grounded case: assert the untouched quote still grounds (no false
    rejection), then corrupt each clinically meaningful token one at a time with a
    sentinel absent from the source and assert every corruption is rejected. This
    is the deterministic, sample-size-independent faithfulness proof, run in CI.
    """
    grounded_cases = [c for c in cases if c["grounded"]]
    corrupted_total = corrupted_rejected = 0
    genuine_total = genuine_grounded = 0

    for c in grounded_cases:
        src, quote = c["source"], c["quote"]
        genuine_total += 1
        if is_grounded(quote, src):
            genuine_grounded += 1
        src_norm = normalize(src)
        toks = _content_tokens(quote)
        for i, tok in enumerate(toks):
            corrupted = " ".join(
                _sentinel(src_norm, i) if t == tok else t
                for t in normalize(quote).split()
            )
            corrupted_total += 1
            if not is_grounded(corrupted, src):
                corrupted_rejected += 1

    return {
        "verifier_catch_rate": corrupted_rejected / corrupted_total if corrupted_total else 1.0,
        "verifier_false_rejection_rate": (genuine_total - genuine_grounded) / genuine_total
        if genuine_total
        else 0.0,
        "n_corrupted": corrupted_total,
        "n_genuine": genuine_total,
    }


def _resolve(metric: str, report: dict, stress: dict) -> float:
    """Resolve a dotted metric path against the stress result or the report dict."""
    if metric in stress:
        return float(stress[metric])
    node: object = report
    for part in metric.split("."):
        node = node[part]  # type: ignore[index]
    return float(node)  # type: ignore[arg-type]


def _passes(value: float, op: str, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    raise ValueError(f"unknown op {op!r}")


def evaluate(baselines_path: Path = BASELINES, fixture_path: Path = FIXTURE) -> dict:
    baselines = json.loads(baselines_path.read_text())
    cases = json.loads(fixture_path.read_text())["cases"]
    report = json.loads((Path(baselines["report"])).read_text())
    stress = stress_test(cases)

    results = []
    for gate in baselines["gates"]:
        value = _resolve(gate["metric"], report, stress)
        ok = _passes(value, gate["op"], gate["threshold"])
        results.append(
            {
                "metric": gate["metric"],
                "op": gate["op"],
                "threshold": gate["threshold"],
                "value": round(value, 4),
                "passed": ok,
                "desc": gate.get("desc", ""),
            }
        )
    return {"passed": all(r["passed"] for r in results), "results": results, "stress": stress}


def main() -> None:
    outcome = evaluate()
    print(f"Regression gate — report: {json.loads(BASELINES.read_text())['report']}")
    print(
        f"Stress test: catch_rate={outcome['stress']['verifier_catch_rate']:.4f} "
        f"({outcome['stress']['n_corrupted']} corruptions), "
        f"false_rejection={outcome['stress']['verifier_false_rejection_rate']:.4f} "
        f"({outcome['stress']['n_genuine']} genuine)"
    )
    print(f"{'':2} {'metric':40} {'value':>9}  {'op':2} {'threshold':>9}")
    for r in outcome["results"]:
        mark = "OK" if r["passed"] else "XX"
        print(f"{mark:2} {r['metric']:40} {r['value']:>9}  {r['op']:2} {r['threshold']:>9}")
    if outcome["passed"]:
        print("\nPASS — no regression.")
        sys.exit(0)
    failed = [r["metric"] for r in outcome["results"] if not r["passed"]]
    print(f"\nFAIL — regressed gates: {', '.join(failed)}")
    sys.exit(1)


if __name__ == "__main__":
    main()
