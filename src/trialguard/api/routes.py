"""HTTP routes: search, trials, assess (SSE), health, budget."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

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

SOURCE = "ctgov_live"
MAX_CRITERIA = 24
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
    if not limiter.allow(_client_ip(request)):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for /api/{kind}. Try again shortly.",
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
    from trialguard.demo import presets

    try:
        return frozenset(presets().values())
    except (FileNotFoundError, OSError):
        return frozenset()


def _is_preset(note: str) -> bool:
    return note.strip() in _preset_notes()


def _trace_handler(session_id: str, kind: str):
    """Langfuse handler for one served request, or None when tracing is off.

    The serving path built no handler at all, so `trace_config` took its
    `handler is None` no-op branch and every live assessment ran untraced while
    the eval harness traced everything. `session_id` is the job id for assess and
    a per-request id for search, so a user-reported result can be found later.
    """
    from trialguard.tracing import get_langchain_handler

    return get_langchain_handler(session_id=session_id, tags=["served", kind])


def _budget_exhausted_detail(exc: BaseException) -> dict[str, Any]:
    from trialguard.llm.cost import active_ledger

    ledger = active_ledger()
    summary = ledger.summary()
    return {
        "error": "BudgetExhausted",
        "message": str(exc),
        "usd_spent": summary["usd"],
        "usd_cap": summary["usd_cap"],
        "remaining_usd": ledger.remaining_usd(),
    }


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
    pool_error: str | None = None
    if settings.database_url:
        try:
            from trialguard.db.schema import get_conn

            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            pool_ok = True
        except Exception as e:  # noqa: BLE001 — surface honestly in health JSON
            pool_error = type(e).__name__
    return {
        "ok": True,
        "pool_ok": pool_ok,
        "pool_error": pool_error,
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
    return LimitsResponse(
        max_assess_trials=settings.api_max_assess_trials,
        max_assess_trials_deep=settings.api_max_assess_trials_deep,
        assess_workers=settings.api_assess_workers,
        usd_per_trial=settings.api_assess_usd_per_trial,
        seconds_per_trial=settings.api_assess_seconds_per_trial,
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
    except Exception as e:  # noqa: BLE001 — the store is the only durable record
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


def _ensure_terminal(store: Any, job_id: str) -> None:
    """No job outlives its task in a non-terminal state.

    The heartbeat check in the store can only see a job whose whole *process* is
    gone; a task that dies inside a live process keeps the owning instance id and
    so never trips it. Here the fact is known rather than inferred, and this runs
    on every exit path including CancelledError, which the `except Exception`
    above does not catch. AD-13.
    """
    try:
        current = store.get(job_id, with_events=False)
        if current is not None and current.status in ("queued", "running"):
            store.fail(job_id, "worker_died: the job task exited without finishing")
    except Exception:  # noqa: BLE001 — must not mask whatever ended the job
        pass


async def _heartbeat(
    store: Any, job_id: str, interval: float, tasks: list[asyncio.Task]
) -> None:
    """Prove the owning process is alive, and stop the work when it is not.

    Its own task on purpose: the analyst call runs in the executor, so a provider
    that hangs to TG_LLM_TIMEOUT stalls a worker thread and not this loop. A job
    whose heartbeat stops has genuinely lost its worker (AD-13).

    The beat is also the cancellation channel. A machine can be suspended and
    never routed to again, so another reader declares its job dead -- correctly,
    because a job nobody can reach is stalled whether or not its process still
    exists. If that machine is later woken it must not resume spending on a job
    the user has already been told failed and has probably retried. The store
    reports that on the next beat, which is the one thing a running job does on a
    timer.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            alive = store.heartbeat(job_id)
        except Exception:  # noqa: BLE001 — a missed beat must not kill the job
            continue
        if not alive:
            for t in tasks:
                t.cancel()
            return


class _Faithfulness:
    """Per-request criterion tally: what was claimed, and what survived checking.

    Counted off the terminal trial events, which carry post-grounding verdicts, so
    this measures what the client was actually told rather than what the model
    first said.
    """

    def __init__(self) -> None:
        self.n = self.unverifiable = self.grounded = self.decisive = self.note_only = 0

    def add(self, assessments: list[dict[str, Any]]) -> None:
        for a in assessments:
            self.n += 1
            verdict = a.get("verdict")
            if verdict == "unverifiable":
                self.unverifiable += 1
            elif verdict in ("met", "not_met"):
                self.decisive += 1
            if a.get("grounded"):
                self.grounded += 1
            if a.get("grounded_in") == "note":
                self.note_only += 1

    def summary(self) -> dict[str, Any]:
        def _rate(x: int) -> float:
            return round(x / self.n, 4) if self.n else 0.0

        return {
            "n_criteria": self.n,
            "unverifiable": self.unverifiable,
            "unverifiable_rate": _rate(self.unverifiable),
            "grounded": self.grounded,
            "grounded_rate": _rate(self.grounded),
            "decisive": self.decisive,
            # WS-5a: decisive verdicts whose only evidence is the user's own note.
            "note_only_grounded": self.note_only,
        }

    def emit(self, job_id: str) -> None:
        """Same numbers onto the trace, where the monitor and dashboard read them."""
        if not self.n:
            return
        from trialguard.tracing import emit_scores

        s = self.summary()
        try:
            emit_scores(
                {
                    "unverifiable_rate": s["unverifiable_rate"],
                    "grounded_rate": s["grounded_rate"],
                    "note_only_grounded": float(s["note_only_grounded"]),
                },
                session_id=job_id,
            )
        except Exception:  # noqa: BLE001
            # Observability must not be able to fail a completed assessment.
            pass


async def _run_assess_job(app: Any, job_id: str) -> None:
    store = app.state.jobs
    job = store.get(job_id, with_events=False)
    if job is None:
        return

    from trialguard.agent.ratelimit import BudgetExhausted

    executor = app.state.assess_executor
    loop = asyncio.get_running_loop()

    deadline = settings.api_assess_trial_deadline_seconds

    async def _one(nct_id: str) -> dict[str, Any]:
        """One trial. Only BudgetExhausted escapes; everything else is an event."""
        # Set when the deadline fires, so the abandoned worker stops emitting
        # provisional criteria for a trial the client has already been told timed
        # out. The UI clears a trial's pending list when its trial event lands; a
        # late criterion would repopulate it and render progress on a closed trial.
        abandoned = threading.Event()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    _assess_one,
                    job.note,
                    nct_id,
                    job_id,
                    job.skip_cache_write,
                    store,
                    abandoned,
                ),
                timeout=deadline,
            )
        except (asyncio.TimeoutError, TimeoutError):
            # The executor thread cannot be interrupted, so it runs to completion
            # and its result is discarded. That costs nothing extra -- the
            # provider call was already made and already billed -- and it buys the
            # user a bounded, honest answer instead of an open-ended wait. It does
            # hold a worker slot until it finishes, so a hung trial narrows
            # concurrency for the rest of the job.
            abandoned.set()
            return {
                "type": "trial",
                "nct_id": nct_id,
                "error": (
                    f"Assessment exceeded the {deadline:.0f}s per-trial deadline "
                    "and was not completed. No verdict is claimed for this trial."
                ),
                "trial_verdict": "cannot_determine",
                "assessments": [],
                "timed_out": True,
            }
        except BudgetExhausted:
            raise
        except Exception as e:  # noqa: BLE001 — per-trial failure must not kill the job
            return {
                "type": "trial",
                "nct_id": nct_id,
                "error": str(e),
                "trial_verdict": "cannot_determine",
                "assessments": [],
            }

    # Every trial is submitted at once and streamed as it lands, rather than
    # awaited one at a time. Concurrency is still bounded by the executor's
    # worker count, which is deliberately also the spend concurrency limit
    # (config.api_assess_workers), so this changes scheduling and not cost: it is
    # the same calls, overlapped. Events carry nct_id and the SSE stream replays
    # them in append order, so completion order is not load-bearing.
    tasks = [asyncio.create_task(_one(n)) for n in job.nct_ids]
    beat = asyncio.create_task(
        _heartbeat(store, job_id, settings.api_job_heartbeat_seconds, tasks)
    )
    tally = _Faithfulness()
    try:
        try:
            for completed in asyncio.as_completed(tasks):
                event = await completed
                tally.add(event.get("assessments") or [])
                store.append(job_id, event)
        except BudgetExhausted as e:
            # Cancel the rest: work still queued in the executor has not started
            # and must not be paid for once the cap is hit.
            for t in tasks:
                t.cancel()
            store.fail(job_id, json.dumps(_budget_exhausted_detail(e)))
            return
        except asyncio.CancelledError:
            # The beat cancelled the work because the job is no longer ours. It
            # already carries a terminal status written by whoever took it.
            return
        store.complete(
            job_id,
            {
                "n_trials": len(job.nct_ids),
                "status": "done",
                # WS-5c. The served monitor is a nightly batch, so a run that
                # starts producing ungrounded verdicts at 09:00 is caught at
                # 21:00. This is the same number, per request, on the event the
                # client already reads and on the trace the monitor already
                # queries -- so the divergence is visible in minutes.
                "faithfulness": tally.summary(),
            },
        )
        # Off the request path, deliberately. emit_scores ends in a blocking
        # client.flush() measured at 4.1 s against a configured Langfuse, so
        # emitting inline would have added that to the wall clock of every
        # completed assessment -- paying latency to report on latency. The job is
        # already done and its events are already durable, so nothing waits on
        # this and nothing breaks if the process dies mid-flush.
        loop.run_in_executor(None, tally.emit, job_id)
    except Exception as e:  # noqa: BLE001
        if isinstance(e, BudgetExhausted):
            store.fail(job_id, json.dumps(_budget_exhausted_detail(e)))
        else:
            store.fail(job_id, str(e))
    finally:
        beat.cancel()
        _ensure_terminal(store, job_id)


def _assess_one(
    note: str,
    nct_id: str,
    job_id: str,
    skip_cache_write: bool,
    store=None,
    abandoned: threading.Event | None = None,
) -> dict[str, Any]:
    """Sync worker: load trial, build typed criteria, call assess() unchanged."""
    from trialguard.agent.graph import assess
    from trialguard.agent.schema import build_typed_criteria
    from trialguard.db.queries import get_trial

    trial = get_trial(nct_id, source=SOURCE)
    if trial is None:
        return {
            "type": "trial",
            "nct_id": nct_id,
            "error": "not_found",
            "trial_verdict": "cannot_determine",
            "assessments": [],
        }
    criteria, truncated = build_typed_criteria(trial, max_total=MAX_CRITERIA)
    if not criteria:
        return {
            "type": "trial",
            "nct_id": nct_id,
            "error": "no_criteria",
            "trial_verdict": "cannot_determine",
            "assessments": [],
            "title": trial.get("title"),
        }
    # L6: a trial is ~29 s of silence even though the model decides its first
    # criterion within a second or two. Emitting each one as it closes makes the
    # grounding check visible while it happens, which is the most persuasive
    # thing this system does. Events are explicitly provisional — they are
    # pre-grounding and a retry can supersede them — and the terminal trial
    # event remains the authority.
    def _emit(assessment: dict) -> None:
        if store is None or (abandoned is not None and abandoned.is_set()):
            return
        try:
            store.append(
                job_id,
                {
                    "type": "criterion",
                    "nct_id": nct_id,
                    "provisional": True,
                    "criterion": assessment.get("criterion", ""),
                    "verdict": assessment.get("verdict", "cannot_determine"),
                },
            )
        except Exception:  # noqa: BLE001
            # These events are explicitly provisional and the terminal trial
            # event is the authority, so a store that drops one costs a UI
            # update. Raising here would turn a transient write failure into a
            # failed trial that had actually succeeded.
            pass

    state = assess(
        note,
        nct_id,
        criteria,
        trial.get("eligibility_raw") or "",
        max_retries=2,
        handler=_trace_handler(job_id, "assess"),
        criteria_truncated=truncated,
        skip_cache_write=skip_cache_write,
        on_criterion=_emit,
    )
    return {
        "type": "trial",
        "nct_id": nct_id,
        "title": trial.get("title"),
        "status": trial.get("status"),
        "trial_verdict": state.get("trial_verdict", "cannot_determine"),
        "criteria_truncated": truncated,
        "assessments": state.get("assessments", []),
    }


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
