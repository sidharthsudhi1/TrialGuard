"""Plan, confirm and publish a refresh (pipeline 1 hardening, WS-5).

The diff is on content, not on CT.gov's lastUpdatePostDate: a trial is
re-embedded exactly when the text its embedding encodes (doc_hash) moved, or it
was embedded under a different config. Everything else that changed is a
metadata-only rewrite that never touches the model.

A trial missing from a complete crawl is not assumed gone. Its current status
is looked up by id; it expires only if CT.gov says it left the requested
statuses (or no longer returns it), or if it has stayed out of the condition
query for SCOPE_MISSING_RUNS consecutive complete crawls. Expiry is soft
(expired_at), so a trial that returns with the same text comes back without
an embedding call.

publish() writes the whole run in one transaction: readers see the corpus
before or after, never partway, and a failure leaves nothing to clean up
because the embedding (the only expensive work) happened before it began.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import psycopg2.extras

from trialguard.db.schema import get_conn
from trialguard.ingestion import ledger
from trialguard.ingestion.loader import update_metadata, upsert_rows
from trialguard.ingestion.provenance import content_hash, doc_hash

SCOPE_MISSING_RUNS = 3
PUBLISH_CHUNK = 500
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Existing:
    doc_hash: str | None
    content_hash: str | None
    embed_tag: str | None
    parser_version: str | None
    expired: bool
    missing_runs: int


@dataclass
class Plan:
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    resurrected: list[str] = field(default_factory=list)
    meta_only: list[str] = field(default_factory=list)
    seen: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def to_embed(self) -> list[str]:
        return self.new + self.changed


@dataclass
class Confirmation:
    expire: dict[str, str] = field(default_factory=dict)
    keep: list[str] = field(default_factory=list)
    lookup_failed: bool = False


def snapshot(source: str) -> dict[str, Existing]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT nct_id, doc_hash, content_hash, embed_tag, parser_version, "
            "expired_at IS NOT NULL, missing_runs FROM trials WHERE source = %s",
            (source,),
        )
        return {r[0]: Existing(*r[1:]) for r in cur.fetchall()}


def corpus_tag(existing: dict[str, Existing]) -> str | None:
    """The embed_tag most active rows carry; 'unknown' rows abstain."""
    tags = Counter(
        e.embed_tag for e in existing.values()
        if not e.expired and e.embed_tag and e.embed_tag != UNKNOWN
    )
    return tags.most_common(1)[0][0] if tags else None


def plan(
    fresh: dict[str, dict],
    existing: dict[str, Existing],
    tag: str,
    parser_version: str,
) -> Plan:
    """Bucket every id once. Pure, so it is tested directly."""
    p = Plan()
    for nct in sorted(fresh):
        t = fresh[nct]
        old = existing.get(nct)
        if old is None:
            p.new.append(nct)
            continue
        same_text = old.doc_hash == doc_hash(t) and old.embed_tag == tag
        if not same_text:
            p.changed.append(nct)
        elif old.expired:
            p.resurrected.append(nct)
        elif old.content_hash != content_hash(t) or old.parser_version != parser_version:
            p.meta_only.append(nct)
        else:
            p.seen.append(nct)
    p.missing = sorted(n for n, e in existing.items() if not e.expired and n not in fresh)
    return p


def confirm(
    missing: list[str],
    existing: dict[str, Existing],
    statuses: list[str],
    lookup: Callable[[list[str]], dict[str, str | None]],
) -> Confirmation:
    """Decide which missing trials really left. A failed lookup expires nothing."""
    c = Confirmation()
    if not missing:
        return c
    try:
        current = lookup(missing)
    except Exception:
        c.keep = list(missing)
        c.lookup_failed = True
        return c
    in_scope = set(statuses)
    for nct in missing:
        status = current.get(nct)
        if status is None:
            c.expire[nct] = "not_found"
        elif status not in in_scope:
            c.expire[nct] = f"status:{status}"
        elif existing[nct].missing_runs + 1 >= SCOPE_MISSING_RUNS:
            # Still enrolling, but a complete crawl has not matched it this many
            # times running: its conditions no longer match the scope query.
            c.expire[nct] = "scope"
        else:
            c.keep.append(nct)
    return c


def embed(trials: list[dict], on_chunk: Callable[[], None] | None = None,
          chunk: int = PUBLISH_CHUNK) -> None:
    """Attach float32 embeddings in place, chunked so progress can heartbeat."""
    from trialguard.ingestion.embed import eligibility_text_for_embedding, embed_matrix

    for i in range(0, len(trials), chunk):
        part = trials[i : i + chunk]
        matrix = embed_matrix([eligibility_text_for_embedding(t) for t in part])
        for t, row in zip(part, matrix, strict=True):
            t["embedding"] = row
        if on_chunk:
            on_chunk()


def _as_list(t: dict) -> dict:
    emb = t["embedding"]
    return {**t, "embedding": emb.tolist() if hasattr(emb, "tolist") else emb}


def stage(cur, trials: list[dict], source: str) -> int:
    """Upsert embedded trials, converting vectors to lists one chunk at a time."""
    n = 0
    for i in range(0, len(trials), PUBLISH_CHUNK):
        n += upsert_rows(cur, [_as_list(t) for t in trials[i : i + PUBLISH_CHUNK]], source)
    return n


def publish(
    *,
    run_id: str,
    source: str,
    p: Plan,
    fresh: dict[str, dict],
    confirmation: Confirmation,
    summary: dict,
    gates: dict,
    data_timestamp: str | None,
    total_count: int,
    staged: bool = False,
) -> None:
    """Write the run atomically. `staged` means the embedded rows are already in."""
    outcome = "published" if _changes(p, confirmation) else "noop"
    with get_conn() as conn, conn.cursor() as cur:
        ledger._lock(cur, source)
        if not staged:
            stage(cur, [fresh[n] for n in p.to_embed], source)
        update_metadata(cur, [fresh[n] for n in p.meta_only + p.resurrected], source)
        if p.seen:
            cur.execute(
                "UPDATE trials SET last_seen_at = NOW(), missing_runs = 0 "
                "WHERE source = %s AND nct_id = ANY(%s)",
                (source, p.seen),
            )
        if confirmation.keep:
            cur.execute(
                "UPDATE trials SET missing_runs = missing_runs + 1 "
                "WHERE source = %s AND nct_id = ANY(%s)",
                (source, confirmation.keep),
            )
        if confirmation.expire:
            psycopg2.extras.execute_values(
                cur,
                "UPDATE trials AS t SET expired_at = NOW(), expired_reason = v.reason "
                "FROM (VALUES %s) AS v(nct_id, source, reason) "
                "WHERE t.nct_id = v.nct_id AND t.source = v.source AND t.expired_at IS NULL",
                [(n, source, r) for n, r in confirmation.expire.items()],
            )
        ledger.finish(
            run_id, outcome, cur=cur, counts=summary, gates=gates,
            ctgov_data_timestamp=data_timestamp, ctgov_total_count=total_count,
            fetched=len(fresh),
        )
        stamp = corpus_stamp(summary, run_id)
        _cache_put(cur, "last_refresh", stamp)
        if outcome == "published":
            _cache_put(cur, "version", {"run_id": run_id, "published_at": stamp["at"]})


def corpus_stamp(summary: dict, run_id: str) -> dict:
    """The ('corpus', 'last_refresh') value /api/health serves. `at` is when the
    corpus was last reconciled against CT.gov, which is what the probe ages."""
    return {**summary, "run_id": run_id, "at": utcnow()}


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _changes(p: Plan, c: Confirmation) -> bool:
    return bool(p.new or p.changed or p.resurrected or p.meta_only or c.expire)


def _cache_put(cur, key: str, value: dict) -> None:
    """cache_entries write inside the publish transaction, not cache_put's own."""
    cur.execute(
        "INSERT INTO cache_entries (namespace, key, value) VALUES ('corpus', %s, %s::jsonb) "
        "ON CONFLICT (namespace, key) DO UPDATE SET value = EXCLUDED.value, created_at = NOW()",
        (key, json.dumps(value)),
    )
