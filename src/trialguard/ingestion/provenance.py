"""Per-row provenance: what a stored trial was built from, and by what.

Every row records the hash of the exact text that was embedded, a hash of the
CT.gov fields it stores, the embedding config and the parser version. The
refresh diffs on these instead of CT.gov's lastUpdatePostDate, which is
day-granular (a second revision posted the same day is invisible to it) and
says nothing about a parser or embedding change on our side.
"""

from __future__ import annotations

import hashlib
import json

from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_tag
from trialguard.ingestion.normalise import PARSER_VERSION, strict_criteria

# Every CT.gov field the trials table stores. A change to any of them without a
# change to doc_hash is a metadata-only update: rewrite the row, skip the model.
CONTENT_FIELDS = (
    "title",
    "status",
    "phase",
    "conditions",
    "interventions",
    "eligibility_raw",
    "min_age",
    "max_age",
    "sex",
    "healthy_volunteers",
    "last_updated",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def doc_hash(trial: dict) -> str:
    """Hash of the string the embedding encodes. Equal hash, equal vector."""
    return _sha(eligibility_text_for_embedding(trial))


def content_hash(trial: dict) -> str:
    canonical = {f: trial.get(f) for f in CONTENT_FIELDS}
    return _sha(json.dumps(canonical, sort_keys=True, default=str))


def parser_version() -> str:
    """PARSER_VERSION, discriminated by TG_STRICT_CRITERIA since that changes the parse."""
    return PARSER_VERSION if strict_criteria() else f"{PARSER_VERSION}+lax"


def stamp(trial: dict) -> dict:
    """The provenance columns for a trial that is about to be written."""
    return {
        "doc_hash": doc_hash(trial),
        "content_hash": content_hash(trial),
        "embed_tag": embed_tag(),
        "parser_version": parser_version(),
    }
