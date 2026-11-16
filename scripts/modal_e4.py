"""E4 on a Modal L4. Scores only. The kill bar stays in eval/e4.py.

    modal run scripts/modal_e4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

GOLD = Path("data/reports/e7_combined_gold.jsonl")
OUT = Path("data/reports/e4_probe.json")
LETTUCE = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
MINICHECK = "lytang/MiniCheck-Flan-T5-Large"
# Yes/No pieces in the current Flan-T5 vocab, tried only if the published ids
# fail the README check. Order is the published pair first.
LABEL_PAIRS = ((3, 209), (465, 2163), (150, 4273), (4168, 19739))
README_DOC = (
    "A group of students gather in the school library to study "
    "for their upcoming final exams."
)
README_YES = "The students are preparing for an examination."
README_NO = "The students are on vacation."

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0",
        "transformers==5.12.1",
        "lettucedetect",
        "sentencepiece",
        "protobuf",
        "accelerate",
    )
    .add_local_python_source("trialguard")
)
app = modal.App("trialguard-e4", image=image)


def _items() -> list[dict]:
    from trialguard.eval.cohorts import load_patients
    from trialguard.eval.e4 import sentence_containing
    from trialguard.eval.nli_probe import _hypothesis, load_gold

    notes: dict[tuple[str, str], str] = {}
    for cohort in ("sigir", "trec_2021", "trec_2022"):
        for patient in load_patients(cohort):
            notes[(cohort, patient["patient_id"])] = patient["description"]
    rows = []
    for item in load_gold(GOLD):
        note = notes.get((item["cohort"], item["patient_id"]), "")
        sentence = sentence_containing(note, item["quote"])
        rows.append(
            {
                "label_source": item["label_source"],
                "entails": bool(item["entails"]),
                "hypothesis": _hypothesis(item),
                "quote": item["quote"],
                "sentence": sentence if sentence is not None else item["quote"],
                "sentence_fallback": sentence is None,
            }
        )
    return rows


@app.function(gpu="L4", cpu=4, memory=16384, timeout=3600, scaledown_window=5)
def score(items: list[dict]) -> dict:
    from lettucedetect.models.inference import HallucinationDetector
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from trialguard.eval.e4 import entail_from_tokens

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MINICHECK)
    model = AutoModelForSeq2SeqLM.from_pretrained(MINICHECK).to(device).eval()
    no_id, yes_id, readme = _choose_ids(tok, model)
    claims = [row["hypothesis"] for row in items]
    minicheck = {
        "quote": _minicheck(tok, model, [row["quote"] for row in items], claims, no_id, yes_id),
        "sentence": _minicheck(
            tok, model, [row["sentence"] for row in items], claims, no_id, yes_id
        ),
    }
    del model

    detector = HallucinationDetector(method="transformer", model_path=LETTUCE)
    lettuce = {"quote": [], "sentence": []}
    sample = None
    for row in items:
        for arm, premise in (("quote", row["quote"]), ("sentence", row["sentence"])):
            tokens = detector.predict(
                context=[premise],
                question="Is this claim supported?",
                answer=row["hypothesis"],
                output_format="tokens",
            )
            if sample is None and tokens:
                sample = tokens[0]
            lettuce[arm].append(entail_from_tokens(tokens))
    return {
        "minicheck_ids": {"no": no_id, "yes": yes_id, "readme": readme},
        "lettuce_sample_keys": sorted(sample) if isinstance(sample, dict) else None,
        "minicheck": minicheck,
        "lettuce": lettuce,
    }


def _choose_ids(tok, model) -> tuple[int, int, list[float]]:
    """Published ids, unless they fail the README pair. Gold is not involved."""
    for no_id, yes_id in LABEL_PAIRS:
        probs = _minicheck(
            tok, model, [README_DOC, README_DOC], [README_YES, README_NO], no_id, yes_id
        )
        if probs[0] > 0.8 and probs[1] < 0.2:
            return no_id, yes_id, probs
    raise RuntimeError("no MiniCheck label ids separate the README pair")


def _minicheck(
    tok, model, docs: list[str], claims: list[str], no_id: int, yes_id: int
) -> list[float]:
    import torch

    device = next(model.parameters()).device
    eos = tok.eos_token or ""
    out: list[float] = []
    batch = 8
    for start in range(0, len(docs), batch):
        chunk_docs = docs[start : start + batch]
        chunk_claims = claims[start : start + batch]
        text = [
            "predict: " + doc + eos + claim
            for doc, claim in zip(chunk_docs, chunk_claims, strict=True)
        ]
        enc = tok(
            text, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(device)
        decoder = torch.zeros((enc["input_ids"].size(0), 1), dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder,
            ).logits.squeeze(1)
        pair = torch.softmax(logits[:, [no_id, yes_id]], dim=-1)
        out.extend(round(float(p), 4) for p in pair[:, 1].cpu())
    return out


@app.local_entrypoint()
def main() -> None:
    from trialguard.eval.e4 import THRESHOLDS, clears_kill
    from trialguard.eval.nli_probe import evaluate

    items = _items()
    remote = score.remote(items)
    report: dict = {
        "n": len(items),
        "sentence_fallbacks": sum(1 for row in items if row["sentence_fallback"]),
        "thresholds": list(THRESHOLDS),
        "minicheck_ids": remote["minicheck_ids"],
        "lettuce_sample_keys": remote["lettuce_sample_keys"],
        "models": {},
    }
    for name, key in ((MINICHECK, "minicheck"), (LETTUCE, "lettuce")):
        arms = {}
        for arm in ("quote", "sentence"):
            scored = [
                {**row, "p_entail": remote[key][arm][i]}
                for i, row in enumerate(items)
            ]
            human = [row for row in scored if row["label_source"] == "human"]
            model = [row for row in scored if row["label_source"] != "human"]
            human_eval = evaluate(human, THRESHOLDS)
            arms[arm] = {
                "human": human_eval,
                "model": evaluate(model, THRESHOLDS),
                "clears_kill": clears_kill(human_eval["operating_points"]),
            }
        report["models"][name] = arms
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}")
    for name, arms in report["models"].items():
        for arm, body in arms.items():
            print(f"{name} {arm} clears={body['clears_kill']}")
