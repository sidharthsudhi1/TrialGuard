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
        lines = [l.strip() for l in text.splitlines()]
        criteria = []
        for line in lines:
            line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line).strip()
            if len(line) > 10:
                criteria.append(line)
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
        return _parse_block(raw[inc_match.end():]), []

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
