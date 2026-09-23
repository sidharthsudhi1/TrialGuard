"""HTTP routes: search, trials, assess (SSE), health, budget."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import uuid
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from trialguard.api.context import (
    SOURCE,
    _budget_exhausted_detail,
    _trace_handler,
)
from trialguard.api.runner import _run_assess_job
from trialguard.api.schemas import (
    SYNTHETIC_NOTICE,
    AssessCreated,
    AssessRequest,
    LimitsResponse,
    SearchRequest,
)
from trialguard.config import settings

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

_KEEPALIVE_SECONDS = 15.0


def _client_ip(request: Request) -> str:
    """Best available client identity for rate limiting.

    X-Forwarded-For is a list the client can prepend to, so its *leftmost* entry
    is whatever the caller typed. Reading it made both rate limits free to
    bypass: send a different value each request and every request lands in its
    own bucket.

    Fly-Client-IP is written by Fly's edge and overwritten on every hop, so the
    client cannot forge it, and it is trusted wherever present. X-Forwarded-For
    is consulted only when the deployment says a proxy is in front
    (api_trust_forwarded_for), and then only its *rightmost* entry, which is the
    hop appended closest to us; everything left of it is attacker-supplied. With
    no proxy there is nothing trustworthy in either header, so the socket peer is
    the only honest answer.
    """
    fly_ip = request.headers.get("fly-client-ip")
    if fly_ip:
        return fly_ip.strip()
    if settings.api_trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            hops = [h.strip() for h in forwarded.split(",") if h.strip()]
            if hops:
                return hops[-1]
    if request.client:
        return request.client.host
    return "unknown"


def _rate_or_429(request: Request, kind: str) -> None:
    limiters = request.app.state.rate_limiters
    limiter = limiters[kind]
    wait = limiter.take(_client_ip(request))
    if wait is None:
        return
    # Retry-After, with the real number rather than "shortly". A client that is
    # told to wait but not how long can only guess, and guessing wrong is how a
    # user turns one rate-limited request into five.
    seconds = max(1, math.ceil(wait))
    raise HTTPException(
        status_code=429,
        detail=(
            f"Rate limit reached for /api/{kind}: at most "
            f"{limiter.limit} requests per minute. Try again in {seconds}s."
        ),
        headers={"Retry-After": str(seconds)},
    )


def _reject_empty_or_injection(note: str) -> None:
    from trialguard.agent.sanitize import detect_injection, detect_phi

    if not note or not note.strip():
        raise HTTPException(status_code=400, detail="Patient note is required.")
    if detect_injection(note):
        raise HTTPException(
            status_code=400,
            detail=(
                "Note looks like a prompt-injection attempt and was rejected. "
                "Paste a synthetic clinical narrative, or pick a preset."
            ),
        )
    # Synthetic-only was a promise in a notice and nothing enforced it, while the
    # served path traces full prompts to Langfuse — so a pasted identifier left
    # this infrastructure and persisted. Refuse rather than redact: redaction
    # would mean accepting real PHI, processing it, and storing a modified copy,
    # which is not what this system tells users it does. The categories are
    # echoed, never the matched text, so the refusal cannot log what it rejects.
    found = detect_phi(note)
    if found:
        raise HTTPException(
            status_code=400,
            detail=(
                "Note appears to contain protected health information "
                f"({', '.join(found)}) and was rejected before any processing. "
                "TrialGuard accepts synthetic notes only. Remove the identifiers "
                "and resubmit, or pick a preset."
            ),
        )


def _cap_top_k(top_k: int) -> int:
    return max(1, min(int(top_k), settings.api_max_search_results))


@lru_cache(maxsize=1)
def _preset_notes() -> frozenset[str]:
    """Preset notes, or empty when the eval fixtures are not on disk.

    Presets come from the SIGIR queries file, which is a repo fixture rather than
    a serving dependency. A deploy that ships only `src/` must not 500 on every
    assess request because of it: with no fixtures nothing matches, every note is
    treated as free text, and the cache write is skipped — the conservative side
    of the WS-4 policy. Cached because the answer cannot change within a process.
    """
    from trialguard.api.demo_presets import DEMO_PRESET_NOTES
    from trialguard.demo import presets

    try:
        fixture = frozenset(presets().values())
    except (FileNotFoundError, OSError):
        fixture = frozenset()
    # The UI's own notes are always allowed, fixtures or not. They are the only
    # notes most users will ever submit, and they were the ones being excluded.
    return fixture | DEMO_PRESET_NOTES


def _is_preset(note: str) -> bool:
    return note.strip() in _preset_notes()






def _vector_cache_status() -> dict[str, Any]:
    from trialguard.retrieval.vector_cache import status

    return status()


def _corpus_freshness() -> dict[str, Any] | None:
    """When the corpus was last reconciled against CT.gov, or None if never.

    A single indexed key read, because this endpoint is polled every 30s by the
    Fly health check. The counts come from the refresh that wrote them rather
    than from a fresh aggregate for the same reason.
    """
    if not settings.database_url:
        return None
    from trialguard.db.cache import cache_get

    return cache_get("corpus", "last_refresh")


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Process up + pool leasable + MedCPT warm flag."""
    pool_ok = False
    # No database configured means the in-process store is the store, and it
    # works; there is no schema to be missing.
    store_ok = not settings.database_url
    pool_error: str | None = None
    store_error: str | None = None
    if settings.database_url:
        try:
            from trialguard.db.schema import get_conn

            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
                pool_ok = True
                # SELECT 1 proves a connection can be leased, not that the tables
                # the served path writes to exist. Production answered every
                # assess with 503 (UndefinedTable) for hours while health
                # reported pool_ok and the probe gated on it -- the monitor
                # reporting healthy precisely when the system was most broken.
                cur.execute("SELECT 1 FROM jobs LIMIT 1")
                cur.fetchall()
                store_ok = True
        except Exception as e:
            if pool_ok:
                store_error = type(e).__name__
            else:
                pool_error = type(e).__name__
    return {
        "ok": True,
        "pool_ok": pool_ok,
        "pool_error": pool_error,
        "store_ok": store_ok,
        "store_error": store_error,
        "medcpt_warm": bool(request.app.state.medcpt_warm),
        "vector_cache": _vector_cache_status(),
        "corpus_refresh": _corpus_freshness(),
        "prompt_version": os.environ.get("TG_PROMPT_VERSION", "v1"),
        "synthetic_only": True,
        "notice": SYNTHETIC_NOTICE,
    }


@router.get("/limits", response_model=LimitsResponse)
def limits() -> LimitsResponse:
    """Caps and measured per-trial cost, so a client can quote before it spends.

    The two rate figures are measurements, not targets: the USD figure is the
    2,000-call TREC 2022 prewarm divided by its trial count, and the latency is
    the median served assess span. Quoting anything else would put a number in
    front of a user that no run supports.
    """
    from trialguard.api.demo_presets import DEMO_PRESETS

    return LimitsResponse(
        max_assess_trials=settings.api_max_assess_trials,
        max_assess_trials_deep=settings.api_max_assess_trials_deep,
        assess_workers=settings.api_assess_workers,
        usd_per_trial=settings.api_assess_usd_per_trial,
        seconds_per_trial=settings.api_assess_seconds_per_trial,
        presets=[dict(p) for p in DEMO_PRESETS],
    )


@router.get("/budget")
def budget() -> dict[str, Any]:
    """Surface the global daily USD ledger for the UI."""
    from trialguard.llm.cost import active_ledger

    ledger = active_ledger()
    s = ledger.summary()
    return {
        "usd_spent": s["usd"],
        "usd_cap": s["usd_cap"],
        "remaining_usd": ledger.remaining_usd(),
        "exhausted": ledger.exhausted(),
        "calls": s["calls"],
        "date": s["date"],
    }


@router.post("/search")
def search(body: SearchRequest, request: Request) -> dict[str, Any]:
    """Ranked ctgov_live trials for a synthetic note. No assess."""
    _rate_or_429(request, "search")
    _reject_empty_or_injection(body.note)

    top_k = _cap_top_k(body.top_k)
    from trialguard.agent.ratelimit import BudgetExhausted
    from trialguard.db.queries import get_trials
    from trialguard.retrieval.pipeline import retrieve

    # A note that misses the keyword cache costs one LLM call, so search can trip
    # the daily cap just as assess can. Notes already cached never reach the
    # model, so they keep working after the budget is spent.
    request_id = uuid.uuid4().hex[:12]
    try:
        hits, latency = retrieve(
            body.note.strip(),
            top_k=top_k,
            source=SOURCE,
            use_keywords=True,
            handler=_trace_handler(request_id, "search"),
        )
    except BudgetExhausted as e:
        raise HTTPException(status_code=402, detail=_budget_exhausted_detail(e)) from e
    rows = get_trials([nct for nct, _ in hits], source=SOURCE)
    trials = []
    for nct, score in hits:
        t = rows.get(nct)
        if not t:
            continue
        trials.append(
            {
                "nct_id": nct,
                "title": t.get("title"),
                "status": t.get("status"),
                "phase": t.get("phase"),
                "conditions": t.get("conditions") or [],
                "last_updated": t.get("last_updated"),
                "score": round(float(score), 4),
            }
        )
    return {
        "trials": trials,
        "top_k": top_k,
        "latency_ms": latency,
        "request_id": request_id,
        "notice": SYNTHETIC_NOTICE,
    }


@router.get("/trials/{nct_id}")
def trial_detail(nct_id: str) -> dict[str, Any]:
    """Full trial row including eligibility_raw for quote highlighting."""
    from trialguard.db.queries import get_trial

    row = get_trial(nct_id, source=SOURCE)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Trial {nct_id} not found.")
    return row


@router.post("/assess", response_model=AssessCreated)
async def assess_start(body: AssessRequest, request: Request) -> AssessCreated:
    """Enqueue assess jobs for user-chosen NCT IDs; stream via GET /assess/{id}."""
    _rate_or_429(request, "assess")
    _reject_empty_or_injection(body.note)

    nct_ids = [n.strip() for n in body.nct_ids if n and n.strip()]
    if not nct_ids:
        raise HTTPException(status_code=400, detail="At least one nct_id is required.")
    # Depth is opt-in (H1: surfaced recall scales with the assessed pool). The
    # standard cap still applies to anyone who does not ask for it, so a client
    # cannot reach the expensive path by accident.
    cap = (
        settings.api_max_assess_trials_deep
        if body.deep
        else settings.api_max_assess_trials
    )
    if len(nct_ids) > cap:
        raise HTTPException(
            status_code=400,
            detail=(
                f"At most {cap} trials per assess request (got {len(nct_ids)})."
                + ("" if body.deep else " Set deep=true to raise the limit.")
            ),
        )

    from trialguard.llm.cost import active_ledger

    if active_ledger().exhausted():
        raise HTTPException(
            status_code=402,
            detail=_budget_exhausted_detail(RuntimeError("Daily spend cap reached.")),
        )

    note = body.note.strip()
    skip_cache = not _is_preset(note)
    store = request.app.state.jobs
    try:
        job = store.create(note, nct_ids, skip_cache_write=skip_cache)
    except Exception as e:
        # Without this the client gets a bare 500 and cannot tell a rejected
        # request from a broken one. Worse, returning a job id anyway would
        # promise work that nothing is doing and no stream can ever report on.
        # The message names the condition only: a psycopg2 OperationalError
        # carries the host it failed to reach, which is not the client's to see.
        logger.error("job store unavailable at create: %s", type(e).__name__)
        raise HTTPException(
            status_code=503,
            detail=(
                "The job store is unavailable, so this assessment was not "
                "started. Nothing was charged; please retry."
            ),
        ) from e
    # Keep a reference: asyncio holds only a weak one, so a fire-and-forget task
    # can be garbage collected mid-await. The job would stop silently and its SSE
    # stream would hang until the TTL expired, with nothing logged.
    task = asyncio.create_task(_run_assess_job(request.app, job.job_id))
    request.app.state.assess_tasks.add(task)
    task.add_done_callback(request.app.state.assess_tasks.discard)
    return AssessCreated(job_id=job.job_id)




def _resume_cursor(request: Request) -> int:
    """Where a reconnecting client left off, from the standard SSE header.

    A browser EventSource replays the last `id:` it saw as `Last-Event-ID` on its
    own reconnects, with no client code involved. A value that is not a
    non-negative integer is treated as "from the beginning" rather than rejected:
    the header is echoed by the client from a previous stream, so a stale or
    mangled one should cost a replay, not a 400 on a job the user has paid for.
    """
    raw = request.headers.get("last-event-id")
    if raw is None:
        return 0
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        return 0


@router.get("/assess/{job_id}")
async def assess_stream(job_id: str, request: Request) -> StreamingResponse:
    """SSE: provisional criterion events, one event per trial, then a summary.

    `criterion` events stream as the analyst produces them and carry
    provisional=True; the `trial` event for the same nct_id is the authoritative
    result and supersedes them.

    Resumable. Each event carries its per-job `seq` as the SSE `id:`, and a
    reconnecting client resumes from `Last-Event-ID`, so a dropped connection
    costs the events in flight at that instant and nothing else. A client that
    reconnects with no header replays the whole log, which is what a fresh reader
    of a finished job wants.
    """
    store = request.app.state.jobs
    job = store.get(job_id, with_events=False)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    resume_from = _resume_cursor(request)

    async def event_gen():
        # A per-job sequence, not a local count of what this generator sent. The
        # count died with the connection and could only ever mean "from the
        # start"; seq is stored with the event, so the same cursor identifies a
        # position in a log that outlives the stream. It also keeps the poll
        # linear: re-reading every event of a 200-event job every 150 ms is a
        # dict slice in memory and a table scan against Neon.
        cursor = resume_from
        idle = 0.0
        while True:
            current = store.get(job_id, with_events=False)
            if current is None:
                payload = {"type": "error", "error": "job_expired"}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                return
            pending = store.events_since(job_id, cursor)
            for seq, ev in pending:
                cursor = seq
                etype = ev.get("type", "trial")
                # SSE `id:` is what a reconnecting client echoes back as
                # Last-Event-ID. Emitting it costs nothing here and is the half
                # of resume that belongs with the sequence number.
                yield f"id: {seq}\nevent: {etype}\ndata: {json.dumps(ev)}\n\n"
                if etype in ("summary", "error"):
                    return
            # Status and events are written in one transaction, so a terminal
            # status is never visible before the event that closed the job.
            if current.status in ("done", "error") and not pending:
                return
            # An SSE comment, ignored by every client, so a stream that is waiting
            # on a slow trial still puts bytes on the wire. Criterion events cover
            # most of that wait, but a provider that hangs before emitting anything
            # produces silence for the whole per-trial deadline, which an
            # intermediary is entitled to read as a dead connection and close.
            if pending:
                idle = 0.0
            else:
                idle += 0.15
                if idle >= _KEEPALIVE_SECONDS:
                    idle = 0.0
                    yield ": keepalive\n\n"
            await asyncio.sleep(0.15)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
