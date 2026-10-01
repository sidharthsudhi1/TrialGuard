"""R3: synthetic training groups for the R2 cross-encoder.

R2 cleared deep fusion on every cell but missed its +0.05 bar, which is the
pre-registered trigger for this. IELAB (TREC CT 2023, arXiv:2401.01566) closed the
same gap by having an LLM write patients for trials the qrels never cover. Every
note is synthetic by construction: written by the model for a real trial's
criteria, with no identifiers, and dropped if detect_phi flags anything.

No source trial and no negative is ever in a test cohort's candidate pool, so a
synthetic group cannot teach the model anything about a trial it is later scored
on. Pre-registered in docs/weakpoints_fix_plan.md §4. Runs on EC2, never on the
Mac (it loads the TREC FileIndex).

    python -m trialguard.eval.r3_synthetic pick
    python -m trialguard.eval.r3_synthetic generate
    python -m trialguard.eval.r3_synthetic export
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

R3_DIR = Path("data/cache/r3")
R2_DIR = Path("data/cache/r2")
N_TRIALS = 2000
SEED = 0
NEG_DEPTH = 30
N_NEG = 7

_PROMPT = """\
You write synthetic patient vignettes for testing a clinical-trial matching system.
Given a trial's eligibility criteria, write ONE vignette for a fictional patient who
meets every inclusion criterion and none of the exclusion criteria.

Style: an admission or clinic note summary, 5 to 10 sentences, third person, as a
physician would write it. State age and sex, the presenting condition, relevant
history, prior treatments, and key findings or labs, with plausible values.
Do not quote the criteria, do not mention the trial, and do not say the patient is
eligible. Include no names, dates, places, record numbers or any identifier.
Output the vignette text only."""


def _pools() -> set[str]:
    out: set[str] = set()
    for c in ("trec_2021", "trec_2022"):
        with gzip.open(R2_DIR / f"{c}.json.gz", "rt") as f:
            out |= set(json.load(f)["docs"])
    return out


def _corpus() -> dict[str, dict]:
    from trialguard.eval.file_index import EVAL_DIR
    from trialguard.ingestion.normalise import normalise_trial

    out = {}
    with open(EVAL_DIR / "trec_2021" / "trec_2021_corpus.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            out[obj["_id"]] = normalise_trial({
                "nct_id": obj["_id"], "title": obj.get("title", ""),
                "eligibility_raw": obj.get("text", ""),
            })
    return out


def pick() -> Path:
    pools = _pools()
    corpus = _corpus()
    ok = sorted(
        n for n, t in corpus.items()
        if n not in pools
        and len(t.get("inclusion_criteria", [])) >= 3
        and len(t.get("exclusion_criteria", [])) >= 1
    )
    chosen = random.Random(SEED).sample(ok, min(N_TRIALS, len(ok)))  # noqa: S311
    R3_DIR.mkdir(parents=True, exist_ok=True)
    path = R3_DIR / "picked.json"
    path.write_text(json.dumps({"eligible_for_sampling": len(ok), "pool_trials": len(pools),
                                "trials": chosen}))
    print(f"{len(ok)} trials outside both pools; picked {len(chosen)}")
    return path


def _criteria_text(t: dict) -> str:
    inc = "\n".join(f"- {c}" for c in t.get("inclusion_criteria", []))
    exc = "\n".join(f"- {c}" for c in t.get("exclusion_criteria", []))
    return f"Inclusion criteria:\n{inc}\n\nExclusion criteria:\n{exc}"


def _write_note(nct: str, trial: dict) -> dict:
    from trialguard.agent.sanitize import detect_phi
    from trialguard.llm.cost import active_ledger
    from trialguard.llm.provider import active_model, active_provider, extract_usage, get_chat_model
    from trialguard.retrieval.query_transform import generate_keywords

    path = R3_DIR / "notes" / f"{nct}.json"
    if path.exists():
        return json.loads(path.read_text())
    resp = get_chat_model("analyst").invoke([
        SystemMessage(content=_PROMPT),
        HumanMessage(content=_criteria_text(trial)),
    ])
    active_ledger().record(extract_usage(resp), active_provider(), active_model())
    note = str(resp.content).strip()
    phi = detect_phi(note)
    rec = {"nct_id": nct, "note": note, "phi": phi,
           "keywords": [] if phi else generate_keywords(note)}
    path.write_text(json.dumps(rec))
    return rec


def generate(workers: int = 10) -> None:
    picked = json.loads((R3_DIR / "picked.json").read_text())["trials"]
    corpus = _corpus()
    (R3_DIR / "notes").mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_write_note, n, corpus[n]) for n in picked]
        for f in futs:
            try:
                f.result()
                done += 1
            except Exception as e:  # one provider error must not end the batch
                print(f"skip: {type(e).__name__}", flush=True)
            if done % 100 == 0:
                print(f"{done}/{len(picked)}", flush=True)


def export() -> Path:
    """Groups in R2's shape: query, then positive doc text followed by 7 negatives."""
    from trialguard.eval.file_index import get_index
    from trialguard.eval.r2_crossencoder import _doc

    pools = _pools()
    corpus = _corpus()
    idx = get_index("trec_2021")
    rng = random.Random(SEED)  # noqa: S311
    groups, dropped = [], {"phi": 0, "no_keywords": 0, "few_negatives": 0}
    for path in sorted((R3_DIR / "notes").glob("*.json")):
        rec = json.loads(path.read_text())
        if rec["phi"]:
            dropped["phi"] += 1
            continue
        kws = rec["keywords"]
        if not kws or kws == [rec["note"]]:
            dropped["no_keywords"] += 1
            continue
        hits = [n for n, _ in idx.search(rec["note"], top_k=NEG_DEPTH, use_keywords=True)]
        cands = [n for n in hits if n != rec["nct_id"] and n not in pools and n in corpus]
        if len(cands) < N_NEG:
            dropped["few_negatives"] += 1
            continue
        negs = rng.sample(cands, N_NEG)
        groups.append({"query": "; ".join(kws),
                       "docs": [_doc(corpus[n]) for n in [rec["nct_id"], *negs]]})
    out = R3_DIR / "synthetic.json.gz"
    with gzip.open(out, "wt") as f:
        json.dump({"groups": groups, "dropped": dropped}, f)
    print(f"{len(groups)} synthetic groups; dropped {dropped}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "generate", "export"])
    args = ap.parse_args()
    {"pick": pick, "generate": generate, "export": export}[args.cmd]()


if __name__ == "__main__":
    main()
