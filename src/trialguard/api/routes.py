"""HTTP routes: search, trials, assess (SSE), health, budget."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from trialguard.api.schemas import SYNTHETIC_NOTICE, AssessCreated, AssessRequest, SearchRequest
from trialguard.config import settings

router = APIRouter(prefix="/api")

SOURCE = "ctgov_live"
MAX_CRITERIA = 24


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
    from trialguard.agent.sanitize import detect_injection

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


def _cap_top_k(top_k: int) -> int:
    return max(1, min(int(top_k), settings.demo_max_top_k))


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
        "prompt_version": os.environ.get("TG_PROMPT_VERSION", "v1"),
        "synthetic_only": True,
        "notice": SYNTHETIC_NOTICE,
    }


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
    if len(nct_ids) > settings.api_max_assess_trials:
        raise HTTPException(
            status_code=400,
            detail=(
                f"At most {settings.api_max_assess_trials} trials per assess request "
                f"(got {len(nct_ids)})."
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
    job = store.create(note, nct_ids, skip_cache_write=skip_cache)
    asyncio.create_task(_run_assess_job(request.app, job.job_id))
    return AssessCreated(job_id=job.job_id)


async def _run_assess_job(app: Any, job_id: str) -> None:
    store = app.state.jobs
    job = store.get(job_id)
    if job is None:
        return

    executor = app.state.assess_executor
    prev_skip = os.environ.get("TG_SKIP_ANALYST_CACHE_WRITE")
    if job.skip_cache_write:
        os.environ["TG_SKIP_ANALYST_CACHE_WRITE"] = "1"
    try:
        for nct_id in job.nct_ids:
            try:
                event = await asyncio.get_running_loop().run_in_executor(
                    executor, _assess_one, job.note, nct_id, job_id
                )
                store.append(job_id, event)
            except Exception as e:  # noqa: BLE001 — per-trial failure must not kill the job
                from trialguard.agent.ratelimit import BudgetExhausted

                if isinstance(e, BudgetExhausted):
                    store.fail(job_id, json.dumps(_budget_exhausted_detail(e)))
                    return
                store.append(
                    job_id,
                    {
                        "type": "trial",
                        "nct_id": nct_id,
                        "error": str(e),
                        "trial_verdict": "cannot_determine",
                        "assessments": [],
                    },
                )
        store.complete(
            job_id,
            {
                "n_trials": len(job.nct_ids),
                "status": "done",
            },
        )
    except Exception as e:  # noqa: BLE001
        from trialguard.agent.ratelimit import BudgetExhausted

        if isinstance(e, BudgetExhausted):
            store.fail(job_id, json.dumps(_budget_exhausted_detail(e)))
        else:
            store.fail(job_id, str(e))
    finally:
        if job.skip_cache_write:
            if prev_skip is None:
                os.environ.pop("TG_SKIP_ANALYST_CACHE_WRITE", None)
            else:
                os.environ["TG_SKIP_ANALYST_CACHE_WRITE"] = prev_skip


def _assess_one(note: str, nct_id: str, job_id: str) -> dict[str, Any]:
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
    state = assess(
        note,
        nct_id,
        criteria,
        trial.get("eligibility_raw") or "",
        max_retries=2,
        handler=_trace_handler(job_id, "assess"),
        criteria_truncated=truncated,
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


@router.get("/assess/{job_id}")
async def assess_stream(job_id: str, request: Request) -> StreamingResponse:
    """SSE: one event per trial, then a terminal summary (or BudgetExhausted)."""
    store = request.app.state.jobs
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")

    async def event_gen():
        sent = 0
        while True:
            current = store.get(job_id)
            if current is None:
                payload = {"type": "error", "error": "job_expired"}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                return
            while sent < len(current.events):
                ev = current.events[sent]
                sent += 1
                etype = ev.get("type", "trial")
                yield f"event: {etype}\ndata: {json.dumps(ev)}\n\n"
                if etype in ("summary", "error"):
                    return
            if current.status in ("done", "error") and sent >= len(current.events):
                return
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
