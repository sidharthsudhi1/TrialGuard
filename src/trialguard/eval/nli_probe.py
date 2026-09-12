"""Can an off-the-shelf NLI model close the entailment gap? (P2 feasibility.)

WS-5b put the gap at 36%: a verbatim quote can be real and still not establish
the verdict it supports. AD-3 rules out an LLM verifier on correlated-error
grounds -- a judge from the analyst's own family fails on the criteria the
analyst fails on. An NLI model is the standing counter-proposal, because it is a
different architecture trained on a different task, and `production_readiness.md`
carries it at P2 as "a model to evaluate, not a check to add".

This evaluates it, against the 50 hand-adjudicated items in
`ws5b_entailment_sample.json`. That file is a labelled gold set: each item is a
grounded decisive verdict with a human judgment of whether the quote establishes
it. So the question is answerable without spending anything or shipping
anything.

The bar is not accuracy. A verifier that is 70% accurate is not usable here,
because the product claim is that a GROUNDED badge means something. What matters
is whether it can be run at a threshold where it almost never rejects a sound
citation, and still catches enough unsound ones to be worth the latency. That is
precision on the "does not entail" class at high confidence, and it is what the
report below leads with.

  python -m trialguard.eval.nli_probe --model microsoft/deberta-large-mnli
"""

from __future__ import annotations

import json
from pathlib import Path

REPORT_DIR = Path("data/reports")
GOLD = REPORT_DIR / "ws5b_entailment_sample.json"

# The verdict is the claim; the quote is the evidence offered for it. NLI asks
# whether the premise entails the hypothesis, so the quote is the premise and the
# claim has to be written out as a sentence.
_CLAIM = {
    ("inclusion", "met"): "The patient satisfies this requirement: {criterion}",
    ("inclusion", "not_met"): "The patient does not satisfy this requirement: {criterion}",
    ("exclusion", "met"): "The patient matches this exclusion: {criterion}",
    ("exclusion", "not_met"): "The patient does not match this exclusion: {criterion}",
}


def _hypothesis(item: dict) -> str:
    kind = item.get("kind") if item.get("kind") in ("inclusion", "exclusion") else "inclusion"
    template = _CLAIM[(kind, item["verdict"])]
    return template.format(criterion=item["criterion"].rstrip(". "))


def load_gold() -> list[dict]:
    data = json.loads(GOLD.read_text())
    return [item for c in data["cohorts"].values() for item in c["items"]]


def score(items: list[dict], model_name: str, batch_size: int = 8) -> list[dict]:
    """Run NLI over (quote, claim) pairs. Returns per-item probabilities."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    # Label order differs between checkpoints, so it is read off the config
    # rather than assumed. Assuming it is how an NLI probe silently reports the
    # contradiction probability as entailment.
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    entail_idx = labels.get("entailment")
    if entail_idx is None:
        raise ValueError(f"{model_name} has no entailment label: {model.config.id2label}")

    out = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        enc = tok(
            [b["quote"] for b in batch],
            [_hypothesis(b) for b in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1)
        for b, p in zip(batch, probs):
            out.append({**b, "p_entail": round(float(p[entail_idx]), 4)})
    return out


def evaluate(scored: list[dict], thresholds: tuple[float, ...]) -> dict:
    """At each threshold, what the model would reject and what that costs.

    "Reject" means p(entail) below the threshold, i.e. the model declines to
    confirm the citation supports the verdict.
    """
    gold_bad = [s for s in scored if not s["entails"]]
    gold_ok = [s for s in scored if s["entails"]]
    rows = []
    for t in thresholds:
        rejected = [s for s in scored if s["p_entail"] < t]
        caught = [s for s in rejected if not s["entails"]]
        wrongly = [s for s in rejected if s["entails"]]
        rows.append(
            {
                "threshold": t,
                "rejected": len(rejected),
                # Of what it rejects, how much was really unsound. Below 1.0 the
                # verifier is destroying sound citations to catch unsound ones.
                "precision": round(len(caught) / len(rejected), 4) if rejected else None,
                # Of the unsound citations, how many it catches.
                "recall": round(len(caught) / len(gold_bad), 4) if gold_bad else None,
                "sound_citations_destroyed": len(wrongly),
            }
        )
    # Ranking quality, independent of calibration. A verifier can be unusable at
    # every threshold and still rank correctly, which is the difference between
    # "wrong model" and "wrong threshold".
    wins = ties = 0
    for g in gold_ok:
        for b in gold_bad:
            if g["p_entail"] > b["p_entail"]:
                wins += 1
            elif g["p_entail"] == b["p_entail"]:
                ties += 1
    pairs = len(gold_ok) * len(gold_bad)
    return {
        "n": len(scored),
        "n_non_entailing": len(gold_bad),
        "n_entailing": len(gold_ok),
        "auc": round((wins + 0.5 * ties) / pairs, 4) if pairs else None,
        "base_rate": round(len(gold_bad) / len(scored), 4) if scored else None,
        "operating_points": rows,
    }


def main() -> None:
    import argparse

    from rich.console import Console
    from rich.table import Table

    ap = argparse.ArgumentParser(description="NLI feasibility against the WS-5b gold set")
    ap.add_argument("--model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    console = Console()
    items = load_gold()
    console.print(f"[bold]NLI probe[/bold] {args.model} · {len(items)} adjudicated items")
    scored = score(items, args.model)
    # Not 0.5. The model is badly miscalibrated on this task: the median
    # p(entail) is 0.027 for citations a human judged sound, so any threshold
    # near the natural decision boundary rejects everything and reports the base
    # rate as precision. Ranking still separates the classes (AUC 0.79), so the
    # operating points that matter are two orders of magnitude lower.
    thresholds = (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05)
    result = evaluate(scored, thresholds)

    t = Table("p(entail) <", "rejects", "precision", "recall", "sound destroyed")
    for r in result["operating_points"]:
        t.add_row(
            str(r["threshold"]), str(r["rejected"]), str(r["precision"]),
            str(r["recall"]), str(r["sound_citations_destroyed"]),
        )
    console.print(t)
    console.print(
        f"gold: {result['n_non_entailing']} non-entailing, {result['n_entailing']} entailing"
        f" · base rate {result['base_rate']} · AUC {result['auc']}"
    )

    out = Path(args.out) if args.out else REPORT_DIR / "nli_probe.json"
    out.write_text(json.dumps({"model": args.model, "result": result,
                               "scored": scored}, indent=2))
    console.print(f"\nReport: [cyan]{out}[/cyan]")


if __name__ == "__main__":
    main()
