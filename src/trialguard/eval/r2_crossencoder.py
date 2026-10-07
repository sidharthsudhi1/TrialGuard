"""R2: a cross-encoder fine-tuned on TREC qrels, trained on one cohort, tested on the other.

R1 re-weighted signals retrieval had already computed and added nothing. This
brings new evidence: a model that reads the patient's keywords against the trial
text. It starts from ncbi/MedCPT-Cross-Encoder, which lost zero-shot (R4, the
2026-08-31 report), and is fine-tuned on the task's own labels, which is what
2024-2026 trial-matching systems do (IELAB at TREC CT 2023, TrialMatchAI, LLM-Match).

Pre-registered in docs/weakpoints_fix_plan.md §4 before any data was scored.

Three steps, so the GPU box needs only torch + transformers and never this repo:

    python -m trialguard.eval.r2_crossencoder export            # local, $0
    python r2_crossencoder.py train-score --train trec_2021 --test trec_2022   # GPU
    python r2_crossencoder.py train-score --train trec_2022 --test trec_2021   # GPU
    python -m trialguard.eval.r2_crossencoder evaluate          # local, $0
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

EXPORT_DIR = Path("data/cache/r2")
MODEL = "ncbi/MedCPT-Cross-Encoder"
GROUP = 8  # 1 positive + 7 negatives
HARD = 3  # at most this many negatives are label-1 "excluded" trials
MAX_LEN = 512
LR = 2e-5
SEED = 0
RRF_K = 60


# ---- export (local) --------------------------------------------------------

def _doc(t: dict) -> str:
    # Shared with serving, so the text the model reads cannot drift from training.
    from trialguard.retrieval.ce_rerank import doc_text

    return doc_text(t)


def export() -> None:
    """Queries, candidate pools, labels and trial texts, from R1's feature caches."""
    from trialguard.eval.cohorts import load_patients
    from trialguard.eval.ltr import _load, _trial_meta
    from trialguard.retrieval.query_transform import generate_keywords

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for cohort in ("trec_2021", "trec_2022"):
        rows = _load(cohort)
        notes = {p["patient_id"]: p["description"] for p in load_patients(cohort)}
        meta = _trial_meta(cohort, {n for r in rows for n in r["candidates"]})
        out = {
            "patients": [
                {
                    "patient_id": r["patient_id"],
                    "query": "; ".join(generate_keywords(notes[r["patient_id"]])),
                    "candidates": r["candidates"],
                    "labels": r["y"].tolist(),
                    "gold_eligible": r["gold_eligible"],
                    "served": r["served"],
                }
                for r in rows
            ],
            "docs": {n: _doc(t) for n, t in meta.items()},
        }
        path = EXPORT_DIR / f"{cohort}.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(out, f)
        print(f"{path}: {len(out['patients'])} patients, {len(out['docs'])} trials")


# ---- train and score (GPU; torch + transformers only) ----------------------

def _groups(data: dict, rng: random.Random) -> list[list[tuple[str, str, int]]]:
    """One group per eligible candidate: the positive first, then 7 pool negatives."""
    groups = []
    for p in data["patients"]:
        pos = [c for c, y in zip(p["candidates"], p["labels"], strict=True) if y == 2]
        hard = [c for c, y in zip(p["candidates"], p["labels"], strict=True) if y == 1]
        easy = [c for c, y in zip(p["candidates"], p["labels"], strict=True) if y == 0]
        for c in pos:
            negs = rng.sample(hard, min(HARD, len(hard)))
            negs += rng.sample(easy, min(GROUP - 1 - len(negs), len(easy)))
            if len(negs) < GROUP - 1:
                continue
            groups.append([(p["query"], data["docs"][n], i) for i, n in enumerate([c, *negs])])
    rng.shuffle(groups)
    return groups


def _train(train_paths: list[str], batch_groups: int, extra_path: str | None = None):
    """Fine-tune from MODEL on every train file. Returns (tok, model, dev, amp, n_groups)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)  # noqa: S311 -- sampling, not security
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # No GPU quota on this account: CPU runs on Sapphire Rapids (c7i), whose AMX
    # units make bf16 autocast practical. fp16 + GradScaler stays for CUDA.
    amp = torch.float16 if dev == "cuda" else torch.bfloat16
    if dev == "cpu":
        import os

        # Physical cores: hyperthreads contend in the matmuls (measured 2.61 vs
        # 2.92 s/step at 16 vs 32 threads on c7i.8xlarge).
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Attention dropout forces the unfused attention path on CPU: 14.9 s/step
    # with it, 4.3 s/step without, hidden dropout kept in both.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, attention_probs_dropout_prob=0.0
    ).to(dev)

    groups = []
    for path in train_paths:
        with gzip.open(path, "rt") as f:
            groups += _groups(json.load(f), rng)
    if len(train_paths) > 1:
        rng.shuffle(groups)
    if extra_path:  # R3: synthetic groups, already (query, positive-first docs)
        with gzip.open(extra_path, "rt") as f:
            extra = json.load(f)["groups"]
        groups += [[(g["query"], d, i) for i, d in enumerate(g["docs"])] for g in extra]
        rng.shuffle(groups)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = (len(groups) + batch_groups - 1) // batch_groups
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps)
    )
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    model.train()
    for step in range(steps):
        batch = groups[step * batch_groups:(step + 1) * batch_groups]
        q = [x[0] for g in batch for x in g]
        d = [x[1] for g in batch for x in g]
        enc = tok(q, d, truncation="only_second", max_length=MAX_LEN, padding=True,
                  return_tensors="pt").to(dev)
        with torch.autocast(dev, dtype=amp):
            logits = model(**enc).logits.view(len(batch), GROUP).float()
        # Localized contrastive: the positive sits at index 0 of every group.
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long,
                                                                     device=dev))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 50 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f}", flush=True)
    return tok, model, dev, amp, len(groups)


def train_score(train_path: str, test_path: str, out_path: str, batch_groups: int = 4,
                extra_path: str | None = None) -> None:
    import torch

    tok, model, dev, amp, n_groups = _train([train_path], batch_groups, extra_path)
    with gzip.open(test_path, "rt") as f:
        test = json.load(f)
    model.eval()
    scores = {}
    with torch.no_grad():
        for p in test["patients"]:
            out = []
            for i in range(0, len(p["candidates"]), 128):
                chunk = p["candidates"][i:i + 128]
                enc = tok([p["query"]] * len(chunk), [test["docs"][n] for n in chunk],
                          truncation="only_second", max_length=MAX_LEN, padding=True,
                          return_tensors="pt").to(dev)
                with torch.autocast(dev, dtype=amp):
                    out += model(**enc).logits.view(-1).float().tolist()
            scores[p["patient_id"]] = out
    Path(out_path).write_text(json.dumps({"train": train_path, "test": test_path,
                                          "n_groups": n_groups, "scores": scores}))
    print(f"wrote {out_path}")


def train_final(out_dir: str, batch_groups: int = 4) -> None:
    """The serving model: same recipe, trained on both TREC cohorts, saved for
    settings.retrieval_ce_model. Cross-cohort runs measured it; this one is not
    evaluable on TREC, since it has seen every TREC topic."""
    paths = [str(EXPORT_DIR / f"{c}.json.gz") for c in ("trec_2021", "trec_2022")]
    tok, model, *_ = _train(paths, batch_groups)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"saved {out_dir}")


# ---- evaluate (local) ------------------------------------------------------

def _recall(order: list[str], gold: list[str], k: int) -> float:
    return len(set(gold) & set(order[:k])) / len(gold)


def _orders(p: dict, s) -> dict[str, list[str]]:
    import numpy as np

    cand = p["candidates"]
    ce = [cand[i] for i in np.argsort(-np.array(s), kind="stable")]
    ce_rank = {n: r for r, n in enumerate(ce, start=1)}
    deep_rank = {n: r for r, n in enumerate(cand, start=1)}
    fused = sorted(cand, key=lambda n: -(1 / (RRF_K + ce_rank[n]) + 1 / (RRF_K + deep_rank[n])))
    return {"deep": cand, "served50": p["served"], "ce": ce, "ce_rrf_deep": fused}


def evaluate(tag: str = "") -> dict:
    """Arms vs deep fusion. With a tag (R3), also each arm vs the same R2 arm."""
    import numpy as np

    from trialguard.eval.significance import compare_family

    report: dict = {"tag": tag, "folds": {}}
    vs_deep, vs_r2, depths = {}, {}, (100, 200)
    for train_c, test_c in (("trec_2021", "trec_2022"), ("trec_2022", "trec_2021")):
        with gzip.open(EXPORT_DIR / f"{test_c}.json.gz", "rt") as f:
            test = json.load(f)
        fold = f"{train_c}->{test_c}"
        runs = {tag or "r2": f"scores_{tag + '_' if tag else ''}{train_c}_to_{test_c}.json"}
        if tag:
            runs["r2"] = f"scores_{train_c}_to_{test_c}.json"
        per: dict[str, list[float]] = {}
        for run, fname in runs.items():
            scores = json.loads((EXPORT_DIR / fname).read_text())["scores"]
            for p in test["patients"]:
                for name, order in _orders(p, scores[p["patient_id"]]).items():
                    key = name if name in ("deep", "served50") else f"{run}:{name}"
                    for k in depths:
                        per.setdefault(f"{key}@{k}", []).append(
                            _recall(order, p["gold_eligible"], k))
        # Baselines were appended once per run; keep one copy.
        n = len(test["patients"])
        per = {m: v[:n] for m, v in per.items()}
        report["folds"][fold] = {"n_test": n,
                                 "mean": {m: round(float(np.mean(v)), 4) for m, v in per.items()}}
        run = tag or "r2"
        for arm in ("ce", "ce_rrf_deep"):
            for k in depths:
                vs_deep[f"{run}:{arm}:{fold}@{k}"] = (per[f"deep@{k}"], per[f"{run}:{arm}@{k}"])
                if tag:
                    vs_r2[f"{arm}:{fold}@{k}"] = (per[f"r2:{arm}@{k}"], per[f"{run}:{arm}@{k}"])
    report["vs_deep"] = compare_family(vs_deep)
    if tag:
        report["vs_r2"] = compare_family(vs_r2)
    name = f"r2_crossencoder{'_' + tag if tag else ''}.json"
    Path("data/reports", name).write_text(json.dumps(report, indent=2, sort_keys=True))
    for fold, f in report["folds"].items():
        print(fold, f["mean"])
    for fam in ("vs_deep", "vs_r2"):
        for key, v in report.get(fam, {}).items():
            print(f"{fam} {key}: {v['mean_delta']:+.4f} {v['delta_ci']} "
                  f"{v['better']}/{v['worse']} p_bh {v['p_bh']}")
    return report


def write_orders(tag: str = "") -> None:
    """Per-patient ce_rrf_deep rankings for each test cohort, for end_to_end --order-file."""
    for train_c, test_c in (("trec_2021", "trec_2022"), ("trec_2022", "trec_2021")):
        with gzip.open(EXPORT_DIR / f"{test_c}.json.gz", "rt") as f:
            test = json.load(f)
        prefix = f"{tag}_" if tag else ""
        scores = json.loads((EXPORT_DIR / f"scores_{prefix}{train_c}_to_{test_c}.json").read_text())
        orders = {p["patient_id"]: _orders(p, scores["scores"][p["patient_id"]])["ce_rrf_deep"]
                  for p in test["patients"]}
        path = EXPORT_DIR / f"order_ce_rrf_deep_{prefix}{test_c}.json"
        path.write_text(json.dumps(orders))
        print(f"{path}: {len(orders)} patients")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "train-score", "train-final", "evaluate", "orders"])
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--dir", default=str(EXPORT_DIR))
    ap.add_argument("--extra", help="R3 synthetic groups (.json.gz)")
    ap.add_argument("--tag", default="", help="prefix for score files and report")
    ap.add_argument("--out", default="data/models/r2_ce", help="train-final output dir")
    args = ap.parse_args()
    if args.cmd == "export":
        export()
    elif args.cmd == "train-score":
        d = Path(args.dir)
        prefix = f"{args.tag}_" if args.tag else ""
        train_score(str(d / f"{args.train}.json.gz"), str(d / f"{args.test}.json.gz"),
                    str(d / f"scores_{prefix}{args.train}_to_{args.test}.json"),
                    extra_path=args.extra)
    elif args.cmd == "train-final":
        train_final(args.out)
    elif args.cmd == "orders":
        write_orders(args.tag)
    else:
        evaluate(args.tag)


if __name__ == "__main__":
    main()
