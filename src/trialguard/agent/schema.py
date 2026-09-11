"""Structured agent I/O. Criterion-level JSON only — no free-text verdicts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["met", "not_met", "cannot_determine", "unverifiable"]
# "unknown" is emitted by attach_kinds when no anchor identifies the criterion.
# It is not a third semantics: rollup_trial treats it as unresolved.
CriterionKind = Literal["inclusion", "exclusion", "unknown"]

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


def align_assessments(
    assessments: list[dict], typed: list[dict]
) -> tuple[list[dict | None], list[dict]]:
    """Pair each asked criterion with the assessment answering it.

    Returns (per-criterion answers, assessments matching no criterion).

    Exact normalized text first, then containment. Containment is needed because
    the model routinely answers a criterion while echoing it differently: asked
    "General: Age equal to or greater than 18.", it returns "Age equal to or
    greater than 18". Measured over the eval cohorts, exact matching alone
    called 7.2% of TREC criteria unanswered that had in fact been answered.

    One-to-one, longest criterion first. A short echo must not claim a long
    criterion that a longer echo answers, and neither may be consumed twice --
    two entries against one criterion is how a trial gets excluded on a
    disqualifier that does not exist.
    """
    from trialguard.verify.grounding import normalize

    keys = [normalize(c["text"]) for c in typed]
    slots: list[dict | None] = [None] * len(typed)
    free = list(range(len(assessments)))
    echo = [normalize(str(a.get("criterion", ""))) for a in assessments]

    for i, key in enumerate(keys):
        for j in list(free):
            if echo[j] and echo[j] == key:
                slots[i] = assessments[j]
                free.remove(j)
                break

    # Longest first: the most specific criterion gets first claim on an echo
    # that several could contain.
    for i in sorted(range(len(typed)), key=lambda i: -len(keys[i])):
        if slots[i] is not None or not keys[i]:
            continue
        for j in list(free):
            e = keys[i]
            if echo[j] and (echo[j] in e or e in echo[j]):
                slots[i] = assessments[j]
                free.remove(j)
                break

    return slots, [assessments[j] for j in free]


def attach_kinds(assessments: list[dict], typed: list[dict]) -> list[dict]:
    """Stamp each assessment with its criterion kind.

    Anchors in order of strength: exact text, an echoed [kind] tag, normalized
    text, then position — and position only when the response and the criteria
    list are the same length. Anything left over is "unknown" rather than a
    guess; rollup_trial treats that as unresolved.
    """
    from trialguard.verify.grounding import normalize

    by_text = {c["text"]: c["kind"] for c in typed}
    by_norm = {normalize(c["text"]): c["kind"] for c in typed}
    aligned = len(assessments) == len(typed)
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
        if kind is None:
            # Whitespace, casing and punctuation drift are not a lost anchor.
            # Measured on SIGIR: this recovers 7.3% of assessments that would
            # otherwise have been guessed at by position.
            kind = by_norm.get(normalize(crit))
        if kind is None:
            # Position is only evidence when the model returned exactly the
            # criteria it was asked about. Once the counts disagree, index i is
            # not typed[i], and a wrong guess on a trial carrying both kinds
            # inverts the criterion's meaning: "patient does not have
            # <disqualifier>" flips from a pass to a failed requirement. Guessing
            # was affordable when the alternative was a wrong verdict either way;
            # with the tiered roll-up an unresolved criterion costs a
            # needs_review row instead, which is the honest answer.
            kind = typed[i]["kind"] if aligned and i < len(typed) else "unknown"
        out.append({**a, "kind": kind})
    return out


def rollup_trial(assessments: list[dict]) -> dict:
    """Tiered trial roll-up: the verdict, and what stands between it and eligible.

    Inclusion not_met → excluded. Exclusion met → excluded (patient matches a
    disqualifier). Eligible only when every inclusion is met and every exclusion
    is not_met. Anything else is unresolved.

    The tier exists because a single verdict cannot carry this system's shape.
    Measured on SIGIR (A1/A5): the "eligible" tier is 100% precise but recovers
    0.0208 of gold, while admitting every trial that merely lacks a disqualifier
    reaches 0.1319 at 28.8% precision and surfaces 68% of the trials the cohort
    marks *excluded*. Neither is the product on its own. A trial blocked only by
    facts the note never stated is a different object from one the patient is
    disqualified from, and the clinician is the right party to judge it — so the
    count of unresolved criteria is reported rather than collapsed into a
    verdict. `needs_review` rows are meant to be ranked by `n_unknown` ascending.

    `verdict` is unchanged from the original three-way roll-up so the regression
    gate and the committed faithfulness floors keep measuring what they measured.
    """
    disqualifying: list[str] = []
    unknown: list[str] = []
    for a in assessments:
        kind = a.get("kind", "inclusion")
        v = a.get("verdict")
        text = str(a.get("criterion", ""))
        if kind == "unknown":
            # Neither semantics can be applied, so no claim can be made.
            unknown.append(text)
        elif kind == "exclusion":
            if v == "met":
                disqualifying.append(text)
            elif v != "not_met":
                unknown.append(text)
        else:
            if v == "not_met":
                disqualifying.append(text)
            elif v != "met":
                unknown.append(text)

    if not assessments:
        verdict, tier = "cannot_determine", "needs_review"
    elif disqualifying:
        verdict, tier = "excluded", "excluded"
    elif unknown:
        verdict, tier = "cannot_determine", "needs_review"
    else:
        verdict, tier = "eligible", "eligible"

    return {
        "verdict": verdict,
        "tier": tier,
        "n_criteria": len(assessments),
        "n_unknown": len(unknown),
        "n_disqualifying": len(disqualifying),
        "unknown": unknown,
        "disqualifying": disqualifying,
    }


def rollup_trial_verdict(assessments: list[dict]) -> str:
    """Three-way verdict. Thin wrapper on rollup_trial for existing callers."""
    return rollup_trial(assessments)["verdict"]


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
