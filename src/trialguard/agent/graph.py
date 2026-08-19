"""LangGraph eligibility graph: Analyst -> Grounding -> (bounded retry) -> Reporter.

The verifier is deterministic (verify/grounding.py): it re-reads the source and
rejects any verdict whose quote is not verbatim-present. On rejection the graph
routes back to the Analyst, hard-capped at max_retries, after which unresolved
criteria are marked "unverifiable" — never forced to a verdict.

Two arms share this graph:
  - single-pass baseline: max_retries=0 (analyst + grounding, no loop)
  - verified:             max_retries=2 (the thesis configuration)
"""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from trialguard.agent.analyst import CACHE_DIR as ANALYST_CACHE
from trialguard.agent.analyst import _cache_key, analyze_trial
from trialguard.agent.schema import attach_kinds, normalize_criteria, rollup_trial_verdict
from trialguard.verify.grounding import ground_assessments


class State(TypedDict, total=False):
    patient_note: str
    nct_id: str
    criteria: list  # list[str] or list[{text, kind}]
    source_text: str
    max_retries: int
    handler: object
    retries: int
    assessments: list[dict]
    trial_verdict: str
    criteria_truncated: bool
    skip_cache_write: bool


def _analyst_node(state: State) -> State:
    attempt = state.get("retries", 0)
    note = state["patient_note"]
    typed = normalize_criteria(state["criteria"])
    # Retrieval-aware retry: instead of a generic "copy verbatim" nudge, hand the
    # analyst the exact trial source span it must quote from, plus the specific
    # criteria whose quotes failed grounding last attempt. The generic nudge only
    # recovered paraphrase failures (SIGIR); pointing at the source span gives the
    # model the characters to copy, the intended fix for TREC's verbatim misses.
    if attempt > 0:
        prior = state.get("assessments", [])
        failed = [a.get("criterion", "") for a in prior if a.get("grounding_failure")]
        crit_list = "\n".join(f"- {c}" for c in failed)
        span = state["source_text"].strip()
        note = (
            f"{note}\n\n[Retry {attempt}] These criteria need a verbatim quote that "
            f"was not found in the source last time:\n{crit_list}\n\nCopy quotes "
            f"character-for-character from this exact trial source text:\n"
            f'"""\n{span}\n"""'
        )
        # In cached-only mode a cold retry cache must not trigger a fresh Groq call.
        # Keep the first-attempt assessments; the bounded loop then exhausts to
        # "unverifiable" without spending quota. Lets all cohorts regenerate the
        # significance + curve from cache alone.
        if os.environ.get("TG_CACHED_ONLY") == "1":
            key = _cache_key(note, state["nct_id"])
            if not (ANALYST_CACHE / f"{key}.json").exists():
                return {"assessments": prior}
    raw = analyze_trial(
        note,
        state["nct_id"],
        typed,
        handler=state.get("handler"),
        skip_cache_write=state.get("skip_cache_write", False),
    )
    # A citation is grounded if it is a verbatim span of ANY provided source:
    # the trial's eligibility text or the patient note. "met"/"not_met" verdicts
    # legitimately cite patient facts ("58-year-old woman") as well as trial text.
    combined_source = state["patient_note"] + "\n" + state["source_text"]
    # Kinds are attached before grounding: an exclusion "not_met" asserts absence
    # and is verified against the note rather than by a verbatim span, so the
    # verifier has to know the criterion's kind to pick the right check.
    typed_raw = attach_kinds(raw, typed)
    grounded = ground_assessments(
        typed_raw, combined_source, patient_text=state["patient_note"]
    )
    return {"assessments": grounded}


def _needs_retry(state: State) -> str:
    failures = any(a.get("grounding_failure") for a in state["assessments"])
    if failures and state.get("retries", 0) < state.get("max_retries", 0):
        return "retry"
    return "report"


def _retry_node(state: State) -> State:
    return {"retries": state.get("retries", 0) + 1}


def _report_node(state: State) -> State:
    """Trial roll-up with inverted exclusion semantics (see rollup_trial_verdict)."""
    return {"trial_verdict": rollup_trial_verdict(state["assessments"])}


def build_graph():
    g = StateGraph(State)
    g.add_node("analyst", _analyst_node)
    g.add_node("retry", _retry_node)
    g.add_node("report", _report_node)
    g.set_entry_point("analyst")
    g.add_conditional_edges("analyst", _needs_retry, {"retry": "retry", "report": "report"})
    g.add_edge("retry", "analyst")
    g.add_edge("report", END)
    return g.compile()


_GRAPH = None


def assess(
    patient_note: str,
    nct_id: str,
    criteria: list,
    source_text: str,
    max_retries: int = 2,
    handler=None,
    criteria_truncated: bool = False,
    skip_cache_write: bool = False,
) -> dict:
    """Run the graph for one (patient, trial). Returns final State dict."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    from trialguard.tracing import trace_config

    config = trace_config(handler, nct_id=nct_id, max_retries=max_retries)
    return _GRAPH.invoke(
        {
            "patient_note": patient_note,
            "nct_id": nct_id,
            "criteria": criteria,
            "source_text": source_text,
            "max_retries": max_retries,
            "handler": handler,
            "retries": 0,
            "criteria_truncated": criteria_truncated,
            "skip_cache_write": skip_cache_write,
        },
        config=config,
    )
