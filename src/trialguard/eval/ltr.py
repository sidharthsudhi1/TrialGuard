"""R1: learning-to-rank over signals retrieval already computes, trained cross-cohort.

Gold sits at median rank 130-180 and recall@500 is ~0.77 (structural_recall_plan
§0): the misses are misordered, not missing. Every reorderer tried so far was
zero-shot (R4 cross-encoders, H2 listwise LLM) and none beat the fused order.
This one is supervised on the task's own qrels and trained on one TREC cohort,
tested on the other, so no test topic is ever seen in training.

Features are cheap and mostly already computed per query: per-keyword dense and
lexical ranks and scores, the note-level dense score, demographic gate margins,
criteria counts, and keyword overlap with inclusion vs exclusion text.

    python -m trialguard.eval.ltr --build trec_2021      # features, cached
    TG_INDEX_EXCLUSION=0 python -m trialguard.eval.ltr --build trec_2022
    python -m trialguard.eval.ltr --evaluate
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

CACHE_DIR = Path("data/cache/ltr")
REPORT = Path("data/reports/r1_ltr.json")

DEEP_POOL = 1000  # per keyword list, as in the gold-rank depth diagnostic
CANDIDATES = 500  # fused deep pool the ranker reorders
SERVED_POOL = 50  # FileIndex.search default, what the served eval uses
RRF_K = 60
GAIN = {"eligible": 2, "excluded": 1}

FEATURES = [
    "rrf_served", "rrf_deep", "log_rank_served", "log_rank_deep",
    "log_min_dense_rank", "log_min_lex_rank",
    "w_recip_dense", "w_recip_lex",
    "n_kw_dense_top100", "n_kw_lex_top100",
    "max_dense_cos", "mean_dense_cos", "max_lex_norm", "mean_lex_norm",
    "kw1_dense_cos", "kw1_lex_norm", "note_dense_cos",
    "demo_gate_fail", "age_over_min", "age_under_max", "sex_restricted",
    "n_inclusion", "n_exclusion", "log_text_len",
    "kw_overlap_incl", "kw_overlap_excl", "kw_overlap_excl_minus_incl",
]

_STOP = set("a an the of and or with in on for to by at from as is are was were be".split())


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 2}


def _trial_meta(cohort: str, keep: set[str]) -> dict[str, dict]:
    """Normalised trials with the age/sex metadata `_load_trec_trials` drops."""
    from trialguard.eval.file_index import EVAL_DIR
    from trialguard.ingestion.normalise import normalise_trial

    out = {}
    with open(EVAL_DIR / cohort / f"{cohort}_corpus.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            nct = obj.get("_id", "")
            if nct not in keep:
                continue
            meta = obj.get("metadata", {})
            t = normalise_trial({
                "nct_id": nct,
                "title": obj.get("title", ""),
                "eligibility_raw": obj.get("text", ""),
            })
            t.update(min_age=meta.get("minimum_age", ""), max_age=meta.get("maximum_age", ""),
                     sex=meta.get("gender", ""))
            out[nct] = t
    return out


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every row under descending score."""
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def _fuse(rank_lists: list[np.ndarray], pool: int, weights: list[float]) -> np.ndarray:
    """Weighted RRF over full rank arrays, each truncated to `pool`; matches fusion.rrf."""
    score = np.zeros(len(rank_lists[0]))
    for r, w in zip(rank_lists, weights, strict=True):
        mask = r <= pool
        score[mask] += w / (RRF_K + r[mask])
    return score


def rank_lists(idx, note: str) -> dict:
    """Full per-keyword dense and lexical rank arrays, in fusion's list order."""
    from trialguard.ingestion.embed import embed_text
    from trialguard.retrieval.fusion import importance_weights
    from trialguard.retrieval.query_transform import generate_keywords

    if len(idx._chunk_owners) != len(idx._nct_ids):
        raise ValueError("R1 expects an unchunked index (one row per trial)")
    keywords = generate_keywords(note)
    kvec = np.array([embed_text(k, is_query=True) for k in keywords], dtype=np.float32)
    dense = idx._matrix @ kvec.T  # rows x K
    lex = np.stack([idx._bm25.get_scores(_tok(k)) for k in keywords], axis=1)
    d_ranks = [_ranks(dense[:, i]) for i in range(len(keywords))]
    l_ranks = [_ranks(lex[:, i]) for i in range(len(keywords))]
    lists = [r for pair in zip(d_ranks, l_ranks, strict=True) for r in pair]
    return {
        "keywords": keywords, "dense": dense, "lex": lex, "d_ranks": d_ranks,
        "l_ranks": l_ranks, "lists": lists,
        "weights": importance_weights(len(lists)) or [1.0] * len(lists),
    }


def patient_pool(idx, note: str) -> dict:
    """Deep candidate pool, served baseline order, and per-candidate retrieval signals."""
    from trialguard.ingestion.embed import embed_text

    ids = idx._nct_ids
    rl = rank_lists(idx, note)
    keywords, dense, lex = rl["keywords"], rl["dense"], rl["lex"]
    d_ranks, l_ranks, lists, w = rl["d_ranks"], rl["l_ranks"], rl["lists"], rl["weights"]
    note_cos = idx._matrix @ np.array(embed_text(note, is_query=True), dtype=np.float32)

    served = _fuse(lists, SERVED_POOL, w)
    deep = _fuse(lists, DEEP_POOL, w)
    # The baseline is FileIndex.search itself, not a re-derivation: equal RRF
    # scores tie-break by dict insertion order there, which a re-derivation
    # reproduces only up to ties.
    row = {n: i for i, n in enumerate(ids)}
    served_order = np.array(
        [row[n] for n, _ in idx.search(note, top_k=CANDIDATES, use_keywords=True)], dtype=np.int64
    )
    deep_order = np.argsort(-deep, kind="stable")
    cand = deep_order[:CANDIDATES]
    served_rank = np.full(len(ids), 10_000)
    served_rank[served_order] = np.arange(1, len(served_order) + 1)
    deep_rank = np.empty(len(ids), dtype=np.int64)
    deep_rank[deep_order] = np.arange(1, len(ids) + 1)
    lex_max = lex.max(axis=0)
    lex_max[lex_max == 0] = 1.0
    return {
        "keywords": keywords,
        "candidates": [ids[r] for r in cand],
        "served": [ids[r] for r in served_order],
        "served_score": served[cand], "deep_score": deep[cand],
        "served_rank": served_rank[cand], "deep_rank": deep_rank[cand],
        "D": np.stack([r[cand] for r in d_ranks], axis=1),
        "L": np.stack([r[cand] for r in l_ranks], axis=1),
        "dense": dense[cand], "lex_n": lex[cand] / lex_max, "note_cos": note_cos[cand],
    }


def patient_features(pool: dict, note: str, labels: dict[str, str], meta: dict) -> dict:
    """Feature matrix and graded labels over one patient's candidate pool."""
    from trialguard.retrieval.demographics import _limit_years, exclusion_reason, parse_patient

    kw = pool["keywords"]
    kw_w = np.array([1.0 / (i + 1) for i in range(len(kw))])
    kw_tokens = set().union(*(_content_tokens(k) for k in kw)) or {""}
    patient = parse_patient(note)
    age = patient["age"]
    D, L, dense, lex_n = pool["D"], pool["L"], pool["dense"], pool["lex_n"]
    rows = []
    for j, nct in enumerate(pool["candidates"]):
        t = meta.get(nct, {})
        incl = " ".join(t.get("inclusion_criteria", []))
        excl = " ".join(t.get("exclusion_criteria", []))
        lo, hi = _limit_years(t.get("min_age")), _limit_years(t.get("max_age"))
        o_in = len(kw_tokens & _content_tokens(incl)) / len(kw_tokens)
        o_ex = len(kw_tokens & _content_tokens(excl)) / len(kw_tokens)
        rows.append([
            pool["served_score"][j], pool["deep_score"][j],
            math.log(pool["served_rank"][j]), math.log(pool["deep_rank"][j]),
            math.log(D[j].min()), math.log(L[j].min()),
            float((kw_w / (RRF_K + D[j])).sum()), float((kw_w / (RRF_K + L[j])).sum()),
            int((D[j] <= 100).sum()), int((L[j] <= 100).sum()),
            float(dense[j].max()), float(dense[j].mean()),
            float(lex_n[j].max()), float(lex_n[j].mean()),
            float(dense[j, 0]), float(lex_n[j, 0]), float(pool["note_cos"][j]),
            1.0 if t and exclusion_reason(t, patient) else 0.0,
            (age - lo) if age is not None and lo is not None else 0.0,
            (hi - age) if age is not None and hi is not None else 0.0,
            1.0 if (t.get("sex") or "").upper() in ("MALE", "FEMALE") else 0.0,
            len(t.get("inclusion_criteria", [])), len(t.get("exclusion_criteria", [])),
            math.log1p(len(incl) + len(excl)),
            o_in, o_ex, o_ex - o_in,
        ])
    return {
        "candidates": pool["candidates"],
        "served": pool["served"],
        "X": np.array(rows, dtype=np.float32),
        "y": np.array([GAIN.get(labels.get(n, ""), 0) for n in pool["candidates"]],
                      dtype=np.int32),
    }


def _tok(text: str) -> list[str]:
    from trialguard.eval.file_index import _tokenize

    return _tokenize(text)


def build(cohort: str) -> Path:
    """Features for every scored patient in a cohort, cached to disk."""
    from trialguard.eval.cohorts import load_patients
    from trialguard.eval.end_to_end import _gold_by_patient
    from trialguard.eval.file_index import get_index
    from trialguard.ingestion.embed import embed_tag

    idx = get_index(cohort)
    gold = _gold_by_patient(cohort)
    corpus = set(idx._nct_ids)
    patients = []
    for p in load_patients(cohort):
        lab = gold.get(p["patient_id"], {})
        elig = sorted(n for n, g in lab.items() if g == "eligible" and n in corpus)
        if elig:
            patients.append((p, lab, elig))

    pools = [patient_pool(idx, p["description"]) for p, _, _ in patients]
    meta = _trial_meta(cohort, {n for pl in pools for n in pl["candidates"]})
    out = []
    for (p, lab, elig), pl in zip(patients, pools, strict=True):
        f = patient_features(pl, p["description"], lab, meta)
        out.append({"patient_id": p["patient_id"], "gold_eligible": elig, **f})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cohort}_{embed_tag()}.json"
    path.write_text(json.dumps([
        {**o, "X": o["X"].tolist(), "y": o["y"].tolist()} for o in out
    ]))
    return path


def _load(cohort: str) -> list[dict]:
    paths = sorted(CACHE_DIR.glob(f"{cohort}_*.json"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one feature cache for {cohort}, found {paths}")
    rows = json.loads(paths[0].read_text())
    for r in rows:
        r["X"], r["y"] = np.array(r["X"], dtype=np.float32), np.array(r["y"])
    return rows


def recall_at(order: list[str], gold: list[str], k: int) -> float:
    g = set(gold)
    return len(g & set(order[:k])) / len(g)


# Fixed before any test fold was scored. No tuning against a test cohort.
PARAMS = dict(
    objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15,
    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    lambdarank_truncation_level=200, random_state=0, verbose=-1,
)


def train(rows: list[dict], label_gain: list[float]):
    import lightgbm as lgb

    model = lgb.LGBMRanker(**PARAMS, label_gain=label_gain)
    X = np.concatenate([r["X"] for r in rows])
    y = np.concatenate([r["y"] for r in rows])
    if len(label_gain) == 2:
        y = (y == 2).astype(int)
    model.fit(X, y, group=[len(r["y"]) for r in rows])
    return model


def score_fold(model, rows: list[dict], depths=(100, 200)) -> dict:
    per = {f"served@{k}": [] for k in depths} | {f"deep@{k}": [] for k in depths} | {
        f"ltr@{k}": [] for k in depths
    }
    for r in rows:
        pred = model.predict(r["X"])
        ltr = [r["candidates"][i] for i in np.argsort(-pred, kind="stable")]
        for k in depths:
            per[f"served@{k}"].append(recall_at(r["served"], r["gold_eligible"], k))
            per[f"deep@{k}"].append(recall_at(r["candidates"], r["gold_eligible"], k))
            per[f"ltr@{k}"].append(recall_at(ltr, r["gold_eligible"], k))
    return per


def evaluate() -> dict:
    from trialguard.eval.significance import compare_family

    cohorts = {"trec_2021": _load("trec_2021"), "trec_2022": _load("trec_2022")}
    result: dict = {"params": PARAMS, "features": FEATURES, "folds": {}}
    family, ltr_vs_deep, deep_vs_served = {}, {}, {}
    for scheme, gain in (("graded", [0.0, 1.0, 3.0]), ("eligible_only", [0.0, 1.0])):
        for train_c, test_c in (("trec_2021", "trec_2022"), ("trec_2022", "trec_2021")):
            model = train(cohorts[train_c], gain)
            per = score_fold(model, cohorts[test_c])
            name = f"{scheme}:{train_c}->{test_c}"
            result["folds"][name] = {
                "n_test": len(cohorts[test_c]),
                "mean": {m: round(float(np.mean(v)), 4) for m, v in per.items()},
                "importance_gain": dict(sorted(
                    zip(FEATURES, model.booster_.feature_importance("gain").round(1).tolist(),
                        strict=True),
                    key=lambda kv: -kv[1],
                )[:10]),
            }
            for k in (100, 200):
                family[f"{name}@{k}"] = (per[f"served@{k}"], per[f"ltr@{k}"])
                ltr_vs_deep[f"{name}@{k}"] = (per[f"deep@{k}"], per[f"ltr@{k}"])
                if scheme == "graded":  # deep order does not depend on the scheme
                    deep_vs_served[f"{test_c}@{k}"] = (per[f"served@{k}"], per[f"deep@{k}"])
    result["vs_served"] = compare_family(family)
    result["ltr_vs_deep"] = compare_family(ltr_vs_deep)
    result["deep_vs_served"] = compare_family(deep_vs_served)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", choices=["trec_2021", "trec_2022"])
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    if args.build:
        print(f"Features: {build(args.build)}")
    if args.evaluate:
        res = evaluate()
        for name, f in res["folds"].items():
            print(name, f["mean"])
        for fam in ("vs_served", "ltr_vs_deep", "deep_vs_served"):
            print(f"-- {fam}")
            for name, v in res[fam].items():
                print(f"{name}: delta {v['mean_delta']:+.4f} CI {v['delta_ci']} "
                      f"{v['better']}/{v['worse']} p_bh {v['p_bh']}")
        print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
