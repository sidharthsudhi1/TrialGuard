"""Faithfulness-forward Gradio demo (Phase 6).

Shows the thesis rather than asserting it: every criterion verdict is rendered with
its verbatim citation and a grounded / unverifiable badge, and "cannot determine"
is a first-class outcome on screen. Retrieval uses the self-contained FileIndex
(numpy + cached MedCPT embeddings), so the demo needs no pgvector DB and runs at $0
on an HF Spaces CPU.

The pipeline seams (retrieve_trials, assess_note) are module-level so the UI logic
is testable without a live Groq call. gradio is imported lazily in build_ui/launch
so importing this module (and the tests) does not require it.
"""

from __future__ import annotations

import functools

DEMO_COHORT = "sigir"
TOP_K = 5
MAX_CRITERIA = 12

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


@functools.lru_cache(maxsize=1)
def _load():
    from trialguard.eval.cohorts import load_patients
    from trialguard.eval.file_index import _load_sigir_trials, get_index
    from trialguard.ingestion.normalise import normalise_trial

    idx = get_index(DEMO_COHORT)
    by_id = {t["nct_id"]: normalise_trial(t) for t in _load_sigir_trials()}
    patients = {p["patient_id"]: p["description"] for p in load_patients(DEMO_COHORT)}
    return idx, by_id, patients


def presets(n: int = 4) -> dict[str, str]:
    """A few synthetic patient notes to seed the demo (SIGIR synthetic cohort)."""
    _, _, patients = _load()
    items = list(patients.items())[:n]
    return {f"Synthetic patient {pid}": note for pid, note in items}


def retrieve_trials(note: str, top_k: int = TOP_K) -> list[dict]:
    idx, by_id, _ = _load()
    hits = idx.search(note, top_k=top_k, use_keywords=True)
    out = []
    for nct, score in hits:
        t = by_id.get(nct)
        if not t:
            continue
        criteria = t.get("inclusion_criteria", [])[:MAX_CRITERIA]
        if not criteria:
            continue
        out.append(
            {
                "nct_id": nct,
                "score": round(float(score), 4),
                "criteria": criteria,
                "source_text": t.get("eligibility_raw", ""),
            }
        )
    return out


def assess_note(note: str, top_k: int = TOP_K) -> dict:
    """Retrieve candidate trials and run the self-verifying agent on each."""
    from trialguard.agent.graph import assess

    trials = retrieve_trials(note, top_k)
    results = []
    for tr in trials:
        state = assess(
            note, tr["nct_id"], tr["criteria"], tr["source_text"], max_retries=2
        )
        results.append(
            {
                "nct_id": tr["nct_id"],
                "score": tr["score"],
                "trial_verdict": state.get("trial_verdict", "cannot_determine"),
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
        lines.append(f"#### {r['nct_id']} — {badge}")
        lines.append(f"<sub>retrieval score {r['score']}</sub>\n")
        for a in r["assessments"]:
            verdict = _VERDICT.get(a.get("verdict", ""), a.get("verdict", ""))
            crit = a.get("criterion", "")
            quote = (a.get("quote") or "").strip()
            grounded = a.get("grounded")
            lines.append(f"- **{verdict}** — {crit}")
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

    if not note or not note.strip():
        return "_Enter or pick a synthetic patient note to begin._"
    try:
        return render(assess_note(note, top_k))
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

    with gr.Blocks(title="TrialGuard — self-verifying trial eligibility") as demo:
        gr.Markdown(
            "# TrialGuard\n"
            "**Self-verifying clinical-trial eligibility.** Every verdict is backed "
            "by a verbatim citation from the trial, or flagged *unverifiable* — never "
            "forced. All patient notes here are synthetic.\n"
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
                "1. **Retrieve** candidate trials (MedCPT dense + BM25, RRF).\n"
                "2. **Analyst** drafts a per-criterion verdict with a verbatim quote.\n"
                "3. **Deterministic grounding** checks each quote is really in the source; "
                "ungrounded claims are downgraded to *unverifiable* and retried (max 2).\n"
                "4. **Roll-up**: excluded if any criterion not met, eligible only if all met, "
                "else cannot determine."
            )
        preset.change(lambda p: preset_map.get(p, ""), inputs=preset, outputs=note)
        go.click(run, inputs=note, outputs=out)
    return demo


def launch(**kwargs):
    build_ui().launch(**kwargs)
