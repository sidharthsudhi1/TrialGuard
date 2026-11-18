"""R2S Stage A: cross-encoder wall time and peak RSS on the serving machine.

Runs inside the API image on a one-off Fly machine (no traffic):
    python scripts/r2s_speed.py <variant>      # F512 | Q512 | Q256 | Q256-300

Prints one JSON line per variant. Bars live in data/reports/r2s_registration.md.
"""

from __future__ import annotations

import json
import resource
import statistics
import sys
import time

VARIANTS = {
    "F512": {"quant": False, "max_len": 512, "n": 500, "queries": 1},
    "Q512": {"quant": True, "max_len": 512, "n": 500, "queries": 3},
    "Q256": {"quant": True, "max_len": 256, "n": 500, "queries": 3},
    "Q256-300": {"quant": True, "max_len": 256, "n": 300, "queries": 3},
}
QUERIES = [
    ["metastatic lung adenocarcinoma", "EGFR exon 19 deletion", "ECOG 1",
     "prior osimertinib", "brain metastases", "age 62", "female"],
    ["triple negative breast cancer", "BRCA1 mutation", "neoadjuvant chemotherapy",
     "residual disease", "age 44", "premenopausal"],
    ["relapsed diffuse large B-cell lymphoma", "two prior lines", "CAR-T naive",
     "ECOG 2", "creatinine clearance 50", "age 71", "male"],
]


def _rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


def _docs(n: int) -> list[str]:
    from trialguard.db.schema import get_conn
    from trialguard.retrieval.ce_rerank import doc_text

    sql = (
        "SELECT title, inclusion_criteria, exclusion_criteria FROM trials "
        "WHERE source = 'ctgov_live' AND expired_at IS NULL "
        "ORDER BY md5(nct_id || 'r2s') LIMIT 500"
    )
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    docs = [doc_text({"title": t, "inclusion_criteria": i, "exclusion_criteria": e})
            for t, i, e in rows]
    return docs[:n]


def main(name: str) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from trialguard.config import settings
    from trialguard.ingestion.embed import embed_text
    from trialguard.retrieval import ce_rerank

    cfg = VARIANTS[name]
    embed_text("warmup", is_query=True)
    docs = _docs(cfg["n"])
    base = _rss_mb()
    path = settings.retrieval_ce_model or "/app/models/r2_ce"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    if cfg["quant"]:
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    lens = [len(tok(d, truncation=True, max_length=cfg["max_len"])["input_ids"]) for d in docs]
    times = []
    for kw in QUERIES[: cfg["queries"]]:
        q = "; ".join(kw)
        t = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(docs), ce_rerank.BATCH):
                chunk = docs[i:i + ce_rerank.BATCH]
                enc = tok([q] * len(chunk), chunk, truncation="only_second",
                          max_length=cfg["max_len"], padding=True, return_tensors="pt")
                model(**enc)
        times.append(time.perf_counter() - t)
    print(json.dumps({
        "variant": name, **cfg, "threads": torch.get_num_threads(),
        "docs": len(docs), "doc_tokens_median": statistics.median(lens),
        "doc_at_cap": sum(n >= cfg["max_len"] for n in lens),
        "secs": [round(x, 1) for x in times], "secs_median": round(statistics.median(times), 1),
        "base_rss_mb": round(base), "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
