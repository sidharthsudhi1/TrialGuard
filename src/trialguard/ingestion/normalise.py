"""Normalise raw eligibility text from CT.gov into structured inclusion/exclusion lists."""

from __future__ import annotations

import re


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip()


# The SIGIR corpus has the words "exclusion criteria" stripped out of its
# headers, leaving a line holding nothing but ":" as the only boundary — 2,666 of
# its 2,991 trials. Missing it files every exclusion criterion as an inclusion,
# and the roll-up then reads a correct "patient does not have <disqualifier>"
# as a failed requirement and excludes a trial the patient is eligible for.
# CT.gov spells the header out, so this never fires on the production corpus
# (measured: 0 of 25,965 ctgov_live trials carry such a line).
_BARE_HEADER = re.compile(r"^[ \t]*:[ \t]*$", re.M)

# A criterion that ends by announcing a list owns the lines nested under it. Split
# into siblings they are silently ANDed, which is wrong twice over: an inclusion
# "any of the following" then demands every alternative at once, and an "except:"
# carve-out is promoted into a disqualifier of its own. Both push a trial toward
# excluded, which is the expensive direction. Measured on ctgov_live: 1,354 trials
# (5.2%) carry an inclusion disjunction and 205 (0.8%) a carve-out.
#
# Exclusion disjunctions are deliberately not listed as a problem — "excluded if
# any exclusion is met" already is the disjunction, so flattening those 2,416
# trials happens to produce the right answer.
_GROUP_HEADER = re.compile(
    r"(?:any (?:one )?of the following|one or more of the following"
    r"|at least one of the following|except|with the exception of|unless)"
    r"[^.\n]{0,40}:\s*$",
    re.IGNORECASE,
)

# Children must be indented at least this much deeper than their parent. CT.gov
# nests at 2, 3 or 5 spaces; the eval corpora were flattened to a uniform single
# leading space before they were packaged, so a 1-space delta is formatting noise
# and must not be read as structure. Where the source carries no indentation the
# extent of a list is unknowable and nothing is grouped — 76.8% of affected
# ctgov_live trials have it, and the rest are left exactly as they parse today.
_NEST = 2

# A merged group replaces N prompt lines with one, so an unbounded merge can hand
# the analyst a single 9k-character criterion and crowd out the other 23. Above
# this budget the group is left flat: the disjunction stays mis-parsed, but a
# criterion nobody can read is not an improvement on one that is merely wrong.
# p95 of a parsed criterion is 365 characters, so this fires rarely.
_GROUP_MAX_CHARS = 1500


def _split_criteria(raw: str) -> tuple[list[str], list[str]]:
    """Split raw eligibility text into inclusion and exclusion criterion lists."""
    if not raw:
        return [], []

    raw = _strip_markdown(raw)

    inc_pattern = re.compile(r"inclusion criteria[:\s]*", re.IGNORECASE)
    exc_pattern = re.compile(r"exclusion criteria[:\s]*", re.IGNORECASE)

    inc_match = inc_pattern.search(raw)
    exc_match = exc_pattern.search(raw)

    def _parse_block(text: str) -> list[str]:
        rows: list[tuple[int, str]] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            indent = len(raw_line) - len(raw_line.lstrip())
            line = re.sub(r"^[\d\.\-\*\•]+\s*", "", stripped).strip()
            if len(line) > 10:
                rows.append((indent, line))

        criteria: list[str] = []
        i = 0
        while i < len(rows):
            indent, line = rows[i]
            if _GROUP_HEADER.search(line):
                j = i + 1
                while j < len(rows) and rows[j][0] >= indent + _NEST:
                    j += 1
                if j > i + 1:
                    merged = line + " " + "; ".join(t for _, t in rows[i + 1 : j])
                    if len(merged) <= _GROUP_MAX_CHARS:
                        criteria.append(merged)
                        i = j
                        continue
            criteria.append(line)
            i += 1
        return criteria

    if inc_match and exc_match:
        if inc_match.start() < exc_match.start():
            inclusion_text = raw[inc_match.end():exc_match.start()]
            exclusion_text = raw[exc_match.end():]
        else:
            exclusion_text = raw[exc_match.end():inc_match.start()]
            inclusion_text = raw[inc_match.end():]
        return _parse_block(inclusion_text), _parse_block(exclusion_text)

    if inc_match:
        body = raw[inc_match.end():]
        # Only trusted after a real inclusion header: a bare ":" with nothing
        # before it says which side is which is not a boundary we can read.
        bare = _BARE_HEADER.search(body)
        if bare:
            return _parse_block(body[: bare.start()]), _parse_block(body[bare.end():])
        return _parse_block(body), []

    if exc_match:
        return [], _parse_block(raw[exc_match.end():])

    return _parse_block(raw), []


def normalise_trial(trial: dict) -> dict:
    """Add inclusion_criteria and exclusion_criteria lists to a trial dict."""
    inclusion, exclusion = _split_criteria(trial.get("eligibility_raw", ""))
    return {
        **trial,
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
    }
