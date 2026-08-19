"""Faithfulness-forward Gradio demo (Phase 6 + production corpus wiring).

Shows the thesis rather than asserting it: every criterion verdict is rendered with
its verbatim citation and a grounded / unverifiable badge, and "cannot determine"
is a first-class outcome on screen.

Two retrieval backends, selected by Settings.demo_source / TG_DEMO_SOURCE:
- sigir (default): self-contained FileIndex — $0 on HF Spaces, no database.
- ctgov_live: production retrieve() over pgvector + Postgres FTS (25,965 trials).

The pipeline seams (retrieve_trials, assess_note) are module-level so the UI logic
is testable without a live Groq call. gradio is imported lazily in build_ui/launch
so importing this module (and the tests) does not require it.
"""

from __future__ import annotations

import functools
import os

from trialguard.config import settings

DEMO_COHORT = "sigir"
TOP_K = 3
MAX_CRITERIA = 24  # aligned with agent.schema.MAX_CRITERIA; truncation is surfaced

_BADGE = {
    "eligible": "🟢 Eligible",
    "excluded": "🔴 Excluded",
    "cannot_determine": "🟡 Cannot determine",
}
_VERDICT = {
    "met": "✅ met",
    "not_met": "❌ not met",
    "cannot_determine": "🟡 cannot determine",
    "unverifiable": "⚠️ unverifiable",
}
# CT.gov enrollment status → display. Empty for eval trials (SIGIR carries no
# status), so the badge only appears on the live ctgov_live corpus.
_STATUS = {
    "RECRUITING": "🟢 recruiting",
    "NOT_YET_RECRUITING": "🟡 opens soon",
    "ENROLLING_BY_INVITATION": "🔵 by invitation",
}


def demo_source() -> str:
    """Active retrieval backend. Env TG_DEMO_SOURCE overrides settings."""
    return os.environ.get("TG_DEMO_SOURCE", settings.demo_source).lower()


def _cap_top_k(top_k: int) -> int:
    return max(1, min(int(top_k), settings.demo_max_top_k))


@functools.lru_cache(maxsize=1)
def _load_sigir():
    from trialguard.eval.file_index import _load_sigir_trials, get_index
    from trialguard.ingestion.normalise import normalise_trial

    idx = get_index(DEMO_COHORT)
    by_id = {t["nct_id"]: normalise_trial(t) for t in _load_sigir_trials()}
    return idx, by_id


@functools.lru_cache(maxsize=1)
def presets(n: int = 4) -> dict[str, str]:
    """A few synthetic patient notes to seed the demo (SIGIR synthetic cohort).

    Reads queries.jsonl only. Must not call get_index() — retrieved.json is
    gitignored, so CI and a fresh clone have no FileIndex corpus.
    """
    from trialguard.eval.cohorts import load_patients

    items = [(p["patient_id"], p["description"]) for p in load_patients(DEMO_COHORT)[:n]]
    return {f"Synthetic patient {pid}": note for pid, note in items}


def warm_models() -> None:
    """Load MedCPT into memory so the first user request is not a cold start."""
    from trialguard.ingestion.embed import embed_text

    embed_text("warmup query", is_query=True)


def retrieve_trials(note: str, top_k: int = TOP_K) -> list[dict]:
    from trialguard.agent.schema import build_typed_criteria

    top_k = _cap_top_k(top_k)
    source = demo_source()
    if source == "ctgov_live":
        from trialguard.db.queries import get_trials
        from trialguard.retrieval.pipeline import retrieve

        hits, _lat = retrieve(note, top_k=top_k, source="ctgov_live", use_keywords=True)
        rows = get_trials([nct for nct, _ in hits], source="ctgov_live")
        out = []
        for nct, score in hits:
            t = rows.get(nct)
            if not t:
                continue
            criteria, truncated = build_typed_criteria(t, max_total=MAX_CRITERIA)
            if not criteria:
                continue
            out.append(
                {
                    "nct_id": nct,
                    "score": round(float(score), 4),
                    "criteria": criteria,
                    "criteria_truncated": truncated,
                    "source_text": t.get("eligibility_raw") or "",
                    "status": t.get("status"),
                    "title": t.get("title"),
                }
            )
        return out

    # Default / fallback: SIGIR FileIndex ($0, no database).
    idx, by_id = _load_sigir()
    hits = idx.search(note, top_k=top_k, use_keywords=True)
    out = []
    for nct, score in hits:
        t = by_id.get(nct)
        if not t:
            continue
        criteria, truncated = build_typed_criteria(t, max_total=MAX_CRITERIA)
        if not criteria:
            continue
        out.append(
            {
                "nct_id": nct,
                "score": round(float(score), 4),
                "criteria": criteria,
                "criteria_truncated": truncated,
                "source_text": t.get("eligibility_raw", ""),
                "status": t.get("status"),
                "title": t.get("title"),
            }
        )
    return out


def assess_note(note: str, top_k: int = TOP_K, skip_cache_write: bool = False) -> dict:
    """Retrieve candidate trials and run the self-verifying agent on each."""
    from trialguard.agent.graph import assess

    trials = retrieve_trials(note, top_k)
    results = []
    for tr in trials:
        state = assess(
            note,
            tr["nct_id"],
            tr["criteria"],
            tr["source_text"],
            max_retries=2,
            criteria_truncated=tr.get("criteria_truncated", False),
            skip_cache_write=skip_cache_write,
        )
        results.append(
            {
                "nct_id": tr["nct_id"],
                "score": tr["score"],
                "status": tr.get("status"),
                "title": tr.get("title"),
                "trial_verdict": state.get("trial_verdict", "cannot_determine"),
                "criteria_truncated": tr.get("criteria_truncated", False),
                "assessments": state.get("assessments", []),
            }
        )
    return {"note": note, "results": results}


def render(result: dict) -> str:
    """Render an assessment result as faithfulness-forward markdown."""
    results = result.get("results", [])
    if not results:
        return "_No candidate trials retrieved for this note._"

    lines = [f"### {len(results)} candidate trials assessed\n"]
    for r in results:
        badge = _BADGE.get(r["trial_verdict"], r["trial_verdict"])
        status = _STATUS.get((r.get("status") or "").upper(), "")
        title = (r.get("title") or "").strip()
        header = f"#### {r['nct_id']} — {badge}"
        if status:
            header += f"  <sub>{status}</sub>"
        lines.append(header)
        if title:
            lines.append(f"*{title}*\n")
        lines.append(f"<sub>retrieval score {r['score']}</sub>\n")
        if r.get("criteria_truncated"):
            lines.append(
                "<sub>⚠️ criteria list truncated at cap — roll-up may be incomplete</sub>\n"
            )
        for a in r["assessments"]:
            verdict = _VERDICT.get(a.get("verdict", ""), a.get("verdict", ""))
            kind = a.get("kind", "inclusion")
            crit = a.get("criterion", "")
            quote = (a.get("quote") or "").strip()
            grounded = a.get("grounded")
            lines.append(f"- **{verdict}** <sub>[{kind}]</sub> — {crit}")
            if a.get("grounding_failure"):
                lines.append(
                    "  <br>⚠️ _quote not verbatim in source — downgraded to unverifiable, "
                    "never forced_"
                )
            elif quote and grounded:
                lines.append(f'  <br>🟢 grounded citation: _"{quote}"_')
        lines.append("")
    lines.append(
        "---\n<sub>Every decisive verdict is backed by a verbatim citation checked "
        "deterministically against the trial text; ungrounded claims are forced to "
        "*unverifiable*, never passed through.</sub>"
    )
    return "\n".join(lines)


def run(note: str, top_k: int = TOP_K) -> str:
    """UI entry: assess and render, with graceful degradation on the Groq cap."""
    from trialguard.agent.ratelimit import BudgetExhausted
    from trialguard.agent.sanitize import detect_injection

    if not note or not note.strip():
        return "_Enter or pick a synthetic patient note to begin._"
    if detect_injection(note):
        return (
            "⚠️ This note looks like a prompt-injection attempt and was rejected. "
            "Paste a synthetic clinical narrative, or pick a preset."
        )

    # Cache writes only for known presets — free-text notes must not grow the
    # analyst cache unboundedly on an ephemeral Space filesystem.
    preset_notes = set(presets().values())
    skip_write = note.strip() not in preset_notes
    try:
        return render(assess_note(note, top_k, skip_cache_write=skip_write))
    except BudgetExhausted:
        return (
            "⚠️ The free Groq daily token budget is spent. Try a preset (its result "
            "is cached) or come back tomorrow — the $0 constraint is real."
        )
    except Exception as e:  # noqa: BLE001 — surface any backend error to the demo user
        if "rate_limit" in str(e) or "429" in str(e):
            return "⚠️ Groq rate limit hit — wait a moment and retry, or use a preset."
        raise


def build_ui():
    import gradio as gr

    preset_map = presets()
    source = demo_source()
    corpus_blurb = (
        "Searching the live recruiting-oncology corpus (pgvector + Postgres FTS)."
        if source == "ctgov_live"
        else "Searching the SIGIR eval FileIndex ($0, no database)."
    )

    with gr.Blocks(title="TrialGuard — self-verifying trial eligibility") as demo:
        gr.Markdown(
            "# TrialGuard\n"
            "**Self-verifying clinical-trial eligibility.** Every verdict is backed "
            "by a verbatim citation from the trial, or flagged *unverifiable* — never "
            f"forced. All patient notes here are synthetic.\n\n_{corpus_blurb}_"
        )
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=list(preset_map), label="Synthetic patient preset", value=None
                )
                note = gr.Textbox(label="Patient note", lines=8, placeholder="Synthetic note…")
                go = gr.Button("Assess eligibility", variant="primary")
            with gr.Column(scale=2):
                out = gr.Markdown()
        with gr.Accordion("How this works", open=False):
            gr.Markdown(
                "1. **Retrieve** candidate trials (MedCPT dense + BM25/FTS, RRF).\n"
                "2. **Analyst** drafts a per-criterion verdict with a verbatim quote "
                "(inclusion and exclusion).\n"
                "3. **Deterministic grounding** checks each quote is really in the source; "
                "ungrounded claims are downgraded to *unverifiable* and retried (max 2).\n"
                "4. **Roll-up**: excluded if any inclusion not met or any exclusion met; "
                "eligible only if all inclusion met and all exclusion not met; else "
                "cannot determine."
            )
        preset.change(lambda p: preset_map.get(p, ""), inputs=preset, outputs=note)
        # Show an immediate status before the (multi-second) Groq round trips, so the
        # button never looks frozen; then run and replace it with the result.
        go.click(
            lambda: "⏳ Retrieving trials and assessing criteria… "
            "(first query also loads the retrieval model; this can take 10–30s)",
            outputs=out,
        ).then(run, inputs=note, outputs=out)
    return demo


def launch(**kwargs):
    if demo_source() == "ctgov_live":
        warm_models()
    build_ui().launch(**kwargs)
