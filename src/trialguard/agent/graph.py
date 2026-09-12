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
from trialguard.agent.schema import attach_kinds, normalize_criteria, rollup_trial
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
    trial_tier: str
    n_unknown: int
    unknown_criteria: list[str]
    disqualifying_criteria: list[str]
    criteria_truncated: bool
    truncated_block: bool
    skip_cache_write: bool
    on_criterion: object


def _retry_failed_only() -> bool:
    """L4: on retry, re-ask only the criteria whose quotes failed grounding.

    Off by default. Turning it on changes the retry prompt, which changes the
    analyst cache key for every retry entry — the committed v1–v4 results would
    silently become cache misses. Same additive discipline as prompt versions:
    the flag exists so the A/B can run without touching what is already
    measured.
    """
    return os.environ.get("TG_RETRY_FAILED_ONLY") == "1"


def _retry_missing() -> bool:
    """P1: re-ask for criteria the analyst never answered at all.

    Measured on TREC 2021: the analyst returns ~9-13% fewer assessment objects
    than it was handed. Not truncation -- max_tokens is 4096 against a
    1,732-token longest response. It is invisible by construction: a criterion
    with no assessment cannot fail grounding, so it never reaches the retry edge,
    and it vanishes into needs_review while rollup_trial still calls the trial
    eligible over the subset it did receive, which CLAUDE.md names unsound.

    Two prompt fixes were measured and rejected first -- v5's numbering works but
    costs 26% of TREC surfaced recall (AD-16), and v6's instruction alone does
    nothing (AD-17). This is the mechanism fix: ask again for what is missing,
    reusing the edge that already re-asks for named criteria.

    **On by default**, unlike the flag above, because the behaviour it replaces
    is not merely lower-recall but unsound: rollup_trial calls a trial
    `eligible` when every criterion it *received* was met, and CLAUDE.md already
    names that unsound over a truncated list. The recall this costs was never
    earned -- it came from not looking.

    Measured on TREC 2021, 20 patients, top-10: criteria never answered fall
    229 -> 71 (13.0% -> 4.0%), grounded rises 840 -> 916, and `eligible`
    precision rises 0.625 -> 0.800 while surfaced recall falls 0.0336 -> 0.0289.
    On SIGIR, where only 12 of 1,814 criteria were being skipped, it is
    neutral-to-positive: surfaced recall unchanged, precision 0.339 -> 0.351.

    `TG_RETRY_MISSING=0` restores the previous behaviour and is what reproduces
    retry numbers committed before 2026-09-10, the same way TG_KEYWORD_DECAY=0
    reproduces pre-2026-09-04 rankings. It has to exist: this changes the retry
    prompt and therefore the cache key of every retry entry.
    """
    return os.environ.get("TG_RETRY_MISSING", "1") != "0"


def _missing_criteria(assessments: list[dict], typed: list[dict]) -> list[dict]:
    """Criteria that came back with no assessment at all.

    Aligned rather than matched on exact text. The model routinely answers a
    criterion while echoing it differently, and re-asking for something it
    already answered spends a call to be told the same thing.
    """
    from trialguard.agent.schema import align_assessments

    slots, _ = align_assessments(assessments, typed)
    return [c for c, answer in zip(typed, slots) if answer is None]


def _merge_retry(
    prior: list[dict], retried: list[dict], recoverable: set[str] | None = None
) -> list[dict]:
    """Overlay retried assessments onto the criteria they were re-asked for.

    Only entries that failed grounding are eligible for replacement, so a
    passing verdict from attempt one can never be overwritten by a retry that
    wandered onto a different criterion. Matching is by criterion text with a
    positional fallback over the failed subset, mirroring attach_kinds: the
    model does not always echo the criterion verbatim, and the alternative to a
    fallback is dropping a recovered quote.
    """
    unconsumed = list(retried)
    by_text: dict[str, dict] = {}
    for a in retried:
        text = a.get("criterion", "")
        if text and text not in by_text:
            by_text[text] = a

    out = []
    for a in prior:
        if not a.get("grounding_failure"):
            out.append(a)
            continue
        replacement = by_text.pop(a.get("criterion", ""), None)
        if replacement is None and unconsumed:
            # Echo did not match, so fall back to the next answer the model gave
            # in order. Retried entries are asked in failed-criteria order.
            replacement = unconsumed[0]
        if replacement is not None and replacement in unconsumed:
            unconsumed.remove(replacement)
        out.append(replacement if replacement is not None else a)

    # A criterion that was missing from attempt one is not in `prior`, so the
    # loop above cannot place it and the recovery would be silently thrown away
    # -- the exact defect this retry exists to fix. Appended here, and only when
    # the text matches a criterion that really was missing, so a model that
    # invents a criterion on retry still cannot add one.
    if recoverable:
        from trialguard.verify.grounding import normalize

        placed = {normalize(str(a.get("criterion", ""))) for a in out}
        for a in unconsumed:
            key = normalize(str(a.get("criterion", "")))
            if key in recoverable and key not in placed:
                placed.add(key)
                out.append(a)
    return out


def _backfill(
    retried: list[dict], prior: list[dict], typed: list[dict]
) -> list[dict]:
    """Retried answers, plus any criterion only attempt one answered.

    Resolved against the real criteria list rather than between the two
    attempts. Deduplicating on the model's own echoed text is not enough: if the
    retry rephrases a criterion, its text does not match attempt one's and the
    criterion ends up answered twice. Measured on SIGIR, that produced 32 more
    assessments than there were criteria -- and two entries for one criterion
    with opposing verdicts can flip a trial to excluded on a disqualifier that
    does not exist.

    So exactly one entry per criterion, the retry's where it answered. Retried
    entries matching no criterion pass through unchanged, as they always have;
    attach_kinds marks those "unknown" and the roll-up treats them as unresolved.
    """
    from trialguard.verify.grounding import normalize

    def _index(rows: list[dict]) -> dict[str, dict]:
        by: dict[str, dict] = {}
        for a in rows:
            key = normalize(str(a.get("criterion", "")))
            if key and key not in by:
                by[key] = a
        return by

    ret_by, prior_by = _index(retried), _index(prior)
    known = {normalize(c["text"]) for c in typed}

    out = [a for a in retried if normalize(str(a.get("criterion", ""))) not in known]
    for c in typed:
        key = normalize(c["text"])
        answer = ret_by.get(key) or prior_by.get(key)
        if answer is not None:
            out.append(answer)
    return out


def _analyst_node(state: State) -> State:
    attempt = state.get("retries", 0)
    note = state["patient_note"]
    typed = normalize_criteria(state["criteria"])
    # Retrieval-aware retry: instead of a generic "copy verbatim" nudge, hand the
    # analyst the exact trial source span it must quote from, plus the specific
    # criteria whose quotes failed grounding last attempt. The generic nudge only
    # recovered paraphrase failures (SIGIR); pointing at the source span gives the
    # model the characters to copy, the intended fix for TREC's verbatim misses.
    prior: list[dict] = []
    missing: list[dict] = []
    asked_subset = False
    if attempt > 0:
        prior = state.get("assessments", [])
        failed = [a.get("criterion", "") for a in prior if a.get("grounding_failure")]
        crit_list = "\n".join(f"- {c}" for c in failed)
        # Criteria that came back with no assessment at all. Asked for
        # separately, because the instruction they need is the opposite one: a
        # failed criterion is told its quote was not verbatim, while a missing
        # one was never answered and telling it to "copy the quote more
        # carefully" would be nonsense.
        missing = _missing_criteria(prior, typed) if _retry_missing() else []
        # L4: re-ask only what failed. A median trial has 6 criteria and few
        # fail, so the retry call shrinks to a fraction of the original instead
        # of re-deciding verdicts that already grounded.
        if _retry_failed_only():
            failed_set = {c for c in failed if c}
            missing_set = {c["text"] for c in missing}
            subset = [c for c in typed if c["text"] in failed_set | missing_set]
            if subset:
                typed = subset
                asked_subset = True
        # Narrowing a missing-only retry to just the skipped criteria was the
        # obvious next move and was measured: it left 144 of 1,761 TREC criteria
        # unanswered against 71 for the full-list re-ask, and dropped `eligible`
        # precision 0.800 -> 0.571. Re-asking everything lets the model produce a
        # fresh complete answer that replaces attempt one wholesale; a narrowed
        # ask produces a handful of entries that have to be merged back by text
        # match, and what the merge cannot place is lost. Not done, deliberately.

        # Built from parts: a retry driven only by omissions must not open with
        # "these criteria need a verbatim quote", followed by nothing.
        blocks = []
        if failed:
            blocks.append(
                "These criteria need a verbatim quote that was not found in the "
                f"source last time:\n{crit_list}"
            )
        if missing:
            listed = "\n".join(f"- {c['text']}" for c in missing)
            blocks.append(
                "These criteria were not answered at all last time. Answer each "
                f"one now, and do not omit any:\n{listed}"
            )
        span = state["source_text"].strip()
        blocks.append(
            "Copy quotes character-for-character from this exact trial source "
            f'text:\n"""\n{span}\n"""'
        )
        note = f"{note}\n\n[Retry {attempt}] " + "\n\n".join(blocks)
        # In cached-only mode a cold retry cache must not trigger a fresh Groq call.
        # Keep the first-attempt assessments; the bounded loop then exhausts to
        # "unverifiable" without spending quota. Lets all cohorts regenerate the
        # significance + curve from cache alone.
        if os.environ.get("TG_CACHED_ONLY") == "1":
            key = _cache_key(note, state["nct_id"])
            if not (ANALYST_CACHE / f"{key}.json").exists():
                return {"assessments": prior}
    # on_criterion is passed only when a caller actually wants progress events,
    # so the default path's call shape is unchanged.
    extra = {}
    if state.get("on_criterion") is not None:
        extra["on_criterion"] = state["on_criterion"]
    raw = analyze_trial(
        note,
        state["nct_id"],
        typed,
        handler=state.get("handler"),
        skip_cache_write=state.get("skip_cache_write", False),
        **extra,
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
        typed_raw,
        combined_source,
        patient_text=state["patient_note"],
        trial_text=state["source_text"],
    )
    # A retry must never lose a criterion attempt one answered. Without this the
    # full-list retry replaces attempt one wholesale, so a retry that happens to
    # return a shorter list makes coverage *worse*: measured at 71, 144 and 269
    # criteria left unanswered across three runs of the same configuration on
    # TREC, against 229 with no retry at all. That spread is not the prompt, it
    # is the model's output length varying and the result being taken whole.
    #
    # Backfilled rather than preferred: the retry's answer wins for every
    # criterion it did answer, and attempt one fills the gaps. A restored entry
    # can be a grounding failure, which is the honest outcome -- unverifiable
    # beats a criterion that silently vanished.
    if attempt > 0 and prior and not asked_subset:
        grounded = _backfill(grounded, prior, typed)

    # A partial retry answered only the criteria it was re-asked, so its result
    # is an overlay on attempt one rather than the whole trial.
    #
    # Gated on having actually narrowed the list, not on the retry returning
    # fewer entries than attempt one. That heuristic held only while a retry
    # could not add anything: recovering a criterion that was never answered
    # makes the retry *longer* than the prior list, and the comparison then
    # skipped the merge and threw attempt one away.
    if attempt > 0 and prior and asked_subset:
        from trialguard.verify.grounding import normalize

        grounded = _merge_retry(
            prior, grounded, {normalize(c["text"]) for c in missing}
        )
    return {"assessments": grounded}


def _needs_retry(state: State) -> str:
    if state.get("retries", 0) >= state.get("max_retries", 0):
        return "report"
    if any(a.get("grounding_failure") for a in state["assessments"]):
        return "retry"
    # A criterion that was never answered produces no assessment, so it cannot
    # fail grounding and this edge never saw it. That is the whole reason the
    # shortfall stayed invisible.
    if _retry_missing():
        typed = normalize_criteria(state["criteria"])
        if _missing_criteria(state["assessments"], typed):
            return "retry"
    return "report"


def _retry_node(state: State) -> State:
    return {"retries": state.get("retries", 0) + 1}


def _report_node(state: State) -> State:
    """Trial roll-up with inverted exclusion semantics (see rollup_trial)."""
    roll = rollup_trial(
        state["assessments"], truncated=state.get("criteria_truncated", False)
    )
    return {
        "trial_verdict": roll["verdict"],
        "trial_tier": roll["tier"],
        "n_unknown": roll["n_unknown"],
        "unknown_criteria": roll["unknown"],
        "disqualifying_criteria": roll["disqualifying"],
        "truncated_block": roll["truncated_block"],
    }


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
    on_criterion=None,
) -> dict:
    """Run the graph for one (patient, trial). Returns final State dict.

    `on_criterion` receives each assessment as the model emits it (L6). Progress
    only: the objects are pre-grounding, and a retry can supersede them, so the
    returned state stays the sole authority on what the system concluded.
    """
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
            "on_criterion": on_criterion,
        },
        config=config,
    )
