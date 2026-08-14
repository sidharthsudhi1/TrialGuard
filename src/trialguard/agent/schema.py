"""Structured agent I/O. Criterion-level JSON only — no free-text verdicts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["met", "not_met", "cannot_determine", "unverifiable"]
CriterionKind = Literal["inclusion", "exclusion"]

# Cap on criteria passed to the analyst per trial. "Eligible only if all met"
# over a silently truncated list is unsound, so callers must surface truncation.
MAX_CRITERIA = 24


class CriterionAssessment(BaseModel):
    criterion: str
    verdict: Verdict
    kind: CriterionKind = "inclusion"
    quote: str = Field(
        default="", description="Verbatim span from trial text supporting the verdict."
    )
    rationale: str = ""
    grounded: bool = False
    grounding_failure: bool = False


class TrialAssessment(BaseModel):
    nct_id: str
    assessments: list[CriterionAssessment]
    # eligible only if all inclusion met AND all exclusion not_met; excluded if
    # any inclusion not_met OR any exclusion met; else cannot_determine.
    trial_verdict: Literal["eligible", "excluded", "cannot_determine"] = "cannot_determine"


_ALLOWED_VERDICTS = {"met", "not_met", "cannot_determine", "unverifiable"}


def normalize_criteria(criteria: list) -> list[dict]:
    """Accept list[str] (legacy = inclusion) or list[{text, kind}] → typed list."""
    out: list[dict] = []
    for c in criteria:
        if isinstance(c, str):
            out.append({"text": c, "kind": "inclusion"})
        elif isinstance(c, dict) and c.get("text"):
            kind = c.get("kind", "inclusion")
            if kind not in ("inclusion", "exclusion"):
                kind = "inclusion"
            out.append({"text": str(c["text"]), "kind": kind})
    return out


def build_typed_criteria(
    trial: dict, max_total: int = MAX_CRITERIA
) -> tuple[list[dict], bool]:
    """Inclusion then exclusion, capped. Returns (criteria, truncated)."""
    items = [{"text": t, "kind": "inclusion"} for t in (trial.get("inclusion_criteria") or [])]
    items += [{"text": t, "kind": "exclusion"} for t in (trial.get("exclusion_criteria") or [])]
    truncated = len(items) > max_total
    return items[:max_total], truncated


def attach_kinds(assessments: list[dict], typed: list[dict]) -> list[dict]:
    """Stamp each assessment with its criterion kind (text match, then index)."""
    by_text = {c["text"]: c["kind"] for c in typed}
    out = []
    for i, a in enumerate(assessments):
        crit = a.get("criterion", "")
        # Compromised/mocked analysts may echo a typed criterion dict as the field.
        if isinstance(crit, dict):
            crit = str(crit.get("text", ""))
            a = {**a, "criterion": crit}
        else:
            crit = str(crit)
        kind = by_text.get(crit)
        if kind is None:
            # Model may echo "[exclusion] text" — strip a leading kind tag.
            for prefix in ("[exclusion] ", "[inclusion] "):
                if crit.lower().startswith(prefix):
                    kind = by_text.get(crit[len(prefix) :]) or prefix.strip("[] ").lower()
                    break
        if kind is None and i < len(typed):
            kind = typed[i]["kind"]
        out.append({**a, "kind": kind or "inclusion"})
    return out


def rollup_trial_verdict(assessments: list[dict]) -> str:
    """Trial roll-up with inverted exclusion semantics.

    Inclusion not_met → excluded. Exclusion met → excluded (patient matches a
    disqualifier). Eligible only when every inclusion is met and every exclusion
    is not_met. Any abstention/unverifiable without a hard exclude → cannot_determine.
    """
    if not assessments:
        return "cannot_determine"
    excluded = False
    unresolved = False
    for a in assessments:
        kind = a.get("kind", "inclusion")
        v = a.get("verdict")
        if kind == "exclusion":
            if v == "met":
                excluded = True
            elif v != "not_met":
                unresolved = True
        else:
            if v == "not_met":
                excluded = True
            elif v != "met":
                unresolved = True
    if excluded:
        return "excluded"
    if unresolved:
        return "cannot_determine"
    return "eligible"


def validate_assessments(raw: object) -> list[dict]:
    """Coerce untrusted analyst JSON into safe criterion dicts (OWASP LLM05).

    Runs on the live model-output boundary before caching. Drops anything that is
    not a dict, forces the verdict to a known enum (unknown -> cannot_determine so
    a malformed verdict can never be treated as decisive), coerces quote/criterion
    to strings, and keeps only the fields the pipeline reads — so a model response
    with extra or wrongly typed keys cannot inject unexpected behavior downstream.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        verdict = item.get("verdict")
        if verdict not in _ALLOWED_VERDICTS:
            verdict = "cannot_determine"
        out.append(
            {
                "criterion": str(item.get("criterion", "")),
                "verdict": verdict,
                "quote": str(item.get("quote", "") or ""),
                "rationale": str(item.get("rationale", "")),
            }
        )
    return out
