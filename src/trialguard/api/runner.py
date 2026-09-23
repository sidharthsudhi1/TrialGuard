"""Assess job execution: the worker behind POST /api/assess.

Extracted from routes.py, which was 809 lines and held the HTTP handlers and
this engine together. Nothing here is transport: it owns job lifecycle, the
heartbeat that proves ownership and cancels a disowned job (AD-13), per-trial
execution against the analyst, and the faithfulness tally reported on `done`.
Behaviour is unchanged by the move.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from trialguard.api.context import (
    MAX_CRITERIA,
    SOURCE,
    _budget_exhausted_detail,
    _trace_handler,
)
from trialguard.config import settings

logger = logging.getLogger(__name__)


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
    except Exception:  # noqa: S110 -- must not mask whatever ended the job
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
        except Exception:  # noqa: S112 -- a failed beat must not kill the loop
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
        self.weak_absence = 0

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
            if a.get("weak_absence"):
                self.weak_absence += 1

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
            # Exclusion not_met resting on a quote that does not establish
            # absence. A weaker class, reported rather than refused.
            "weak_absence": self.weak_absence,
        }

    def emit(self, job_id: str) -> None:
        """Same numbers onto the trace, where the monitor and dashboard read them."""
        if not self.n:
            return
        from trialguard.tracing import emit_scores

        s = self.summary()
        try:  # noqa: SIM105 -- the reason lives in the block below
            emit_scores(
                {
                    "unverifiable_rate": s["unverifiable_rate"],
                    "grounded_rate": s["grounded_rate"],
                    "note_only_grounded": float(s["note_only_grounded"]),
                },
                session_id=job_id,
            )
        except Exception:  # noqa: S110
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
        except TimeoutError:
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
        except Exception as e:
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
    except Exception as e:
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
    if trial.get("expired_at"):
        # Left the enrolling set after the search that produced this id. Assessing
        # it would spend on a trial nobody can join.
        return {
            "type": "trial",
            "nct_id": nct_id,
            "error": "no_longer_recruiting",
            "expired_reason": trial.get("expired_reason"),
            "trial_verdict": "cannot_determine",
            "assessments": [],
            "title": trial.get("title"),
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
        try:  # noqa: SIM105 -- the reason lives in the block below
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
        except Exception:  # noqa: S110
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
        # True when every assessed criterion passed and the only thing standing
        # between this trial and `eligible` is the criteria the cap dropped. It
        # is a different statement from "we could not tell", and the UI says so.
        "truncated_block": state.get("truncated_block", False),
        "assessments": state.get("assessments", []),
    }
