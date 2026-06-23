"""Phase 1 ingestion CLI.

Usage:
    python -m trialguard.scripts.ingest --trials 3000
    python -m trialguard.scripts.ingest --skip-trials  # only eval cohorts
    python -m trialguard.scripts.ingest --skip-eval    # only trials
"""

from __future__ import annotations

import argparse

from rich.console import Console

console = Console()


def run(max_trials: int, skip_trials: bool, skip_eval: bool) -> None:
    from trialguard.db.schema import init_schema
    from trialguard.eval.cohorts import download_cohorts, load_labels, load_patients
    from trialguard.ingestion.ctgov import fetch_oncology_trials
    from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_batch
    from trialguard.ingestion.loader import upsert_trials
    from trialguard.ingestion.normalise import normalise_trial
    from trialguard.tracing import flush

    console.print("[bold]TrialGuard Phase 1 — Ingestion[/bold]")

    console.print("Initialising schema...")
    init_schema()

    if not skip_trials:
        console.print(f"Pulling up to {max_trials} oncology trials from CT.gov...")
        batch: list[dict] = []
        total = 0

        for trial in fetch_oncology_trials(max_trials=max_trials):
            trial = normalise_trial(trial)
            batch.append(trial)

            if len(batch) == 200:
                texts = [eligibility_text_for_embedding(t) for t in batch]
                embeddings = embed_batch(texts)
                for t, emb in zip(batch, embeddings):
                    t["embedding"] = emb
                upserted = upsert_trials(batch)
                total += upserted
                console.print(f"  Upserted {total} trials so far...")
                batch = []

        if batch:
            texts = [eligibility_text_for_embedding(t) for t in batch]
            embeddings = embed_batch(texts)
            for t, emb in zip(batch, embeddings):
                t["embedding"] = emb
            total += upsert_trials(batch)

        console.print(f"[green]Trials done: {total} upserted.[/green]")

    if not skip_eval:
        console.print("Downloading eval cohorts...")
        download_cohorts()

        from trialguard.db.schema import get_conn
        import psycopg2.extras, json

        with get_conn() as conn, conn.cursor() as cur:
            for cohort in ("sigir", "trec_2021", "trec_2022"):
                patients = load_patients(cohort)
                psycopg2.extras.execute_batch(
                    cur,
                    """
                    INSERT INTO eval_patients (patient_id, cohort, description, raw)
                    VALUES (%(patient_id)s, %(cohort)s, %(description)s, %(raw)s::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        {**p, "raw": json.dumps(p["raw"])}
                        for p in patients
                    ],
                )

                labels = load_labels(cohort)
                psycopg2.extras.execute_batch(
                    cur,
                    """
                    INSERT INTO eval_labels (patient_id, nct_id, cohort, label)
                    VALUES (%(patient_id)s, %(nct_id)s, %(cohort)s, %(label)s)
                    ON CONFLICT DO NOTHING
                    """,
                    labels,
                )
                console.print(
                    f"  {cohort}: {len(patients)} patients, {len(labels)} labels"
                )

            conn.commit()

        console.print("[green]Eval cohorts loaded.[/green]")

    flush()
    console.print("[bold green]Phase 1 complete.[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="TrialGuard Phase 1 ingestion")
    parser.add_argument("--trials", type=int, default=3000, dest="max_trials")
    parser.add_argument("--skip-trials", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    run(args.max_trials, args.skip_trials, args.skip_eval)


if __name__ == "__main__":
    main()
