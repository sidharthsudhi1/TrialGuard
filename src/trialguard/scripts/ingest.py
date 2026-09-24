"""Phase 1 ingestion CLI.

Usage:
    python -m trialguard.scripts.ingest                # schema, trials, eval cohorts
    python -m trialguard.scripts.ingest --skip-trials  # only eval cohorts
    python -m trialguard.scripts.ingest --skip-eval    # only trials

Trials load through scripts/refresh.py, so a first load gets the same
completeness proof, audit gates and ledger row as every refresh after it.
"""

from __future__ import annotations

import argparse

from rich.console import Console

console = Console()


def run(skip_trials: bool, skip_eval: bool) -> None:
    from trialguard.db.schema import init_schema
    from trialguard.eval.cohorts import download_cohorts, load_labels, load_patients
    from trialguard.tracing import flush

    console.print("[bold]TrialGuard Phase 1 — Ingestion[/bold]")

    console.print("Initialising schema...")
    init_schema()

    if not skip_trials:
        # The corpus load is the refresh: an empty corpus bootstraps (chunked
        # commits, churn gates off, content gates on), and a partial one resumes
        # as an ordinary diff, re-embedding nothing it already holds.
        from trialguard.scripts.refresh import EXIT_OK
        from trialguard.scripts.refresh import run as refresh_run

        code, report = refresh_run()
        if code != EXIT_OK:
            raise SystemExit(f"Trial load did not publish ({report.get('outcome')}).")
        console.print(f"[green]Trials: {report.get('summary') or report}[/green]")

    if not skip_eval:
        console.print("Downloading eval cohorts...")
        download_cohorts()

        import json

        import psycopg2.extras

        from trialguard.db.schema import get_conn

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
    parser.add_argument("--skip-trials", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    run(args.skip_trials, args.skip_eval)


if __name__ == "__main__":
    main()
