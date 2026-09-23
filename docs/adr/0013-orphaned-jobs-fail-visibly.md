# AD-13 — Orphaned assess jobs fail visibly

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

Orphaned assess jobs fail visibly and the user retries; nothing requeues itself. A job row carries the writing process's instance id alongside a heartbeat, and a reader treats `running` as dead when a *different* instance sees it **or** the heartbeat is older than the threshold. Two conditions rather than one, because `fly.toml` suspends idle machines: suspend snapshots RAM, so the process resumes intact with a heartbeat frozen for the whole suspension, and an age-only test would report a live job failed. Deploy, crash and OOM each produce a new instance, so those are caught. Recovery is the durable event log, not the analyst cache — a free-text note sets `skip_cache_write`, so nothing a non-preset job computed is cached anywhere, and a retry is cheap only because completed `trial` events survive in Postgres and the client re-sends just the NCT IDs it never received. The note is stored under the same one-hour retention the in-process store already applied, enforced as a delete rather than eviction-on-read

## Alternatives considered

Requeue on startup (needs leader election or two instances duplicate the work, and spends money nobody asked for, against the opt-in cost posture Phase 9 built); TTL eviction only (the client polls a `running` row until it gives up and the operator sees a queue that never drains); heartbeat age alone (one column, but it cannot tell suspension from death); disabling suspend so age becomes sufficient (bills the machine around the clock, already reverted once for cost)

## Amended in WS-1

The rule as first written was a disjunction — a different instance **or** a stale heartbeat. Building it showed that is wrong in both directions. Age alone still fails a *suspended* job, which is the case the two columns existed to protect: the request that resumes a machine is a blocking read that can run before the resumed heartbeat task gets the event loop back. Instance alone fails a *healthy* job as soon as `auto_start_machines` puts a second machine behind the hostname and a poll is load-balanced away from the owner. Both clauses are necessary, so the test is a conjunction. That leaves one case a conjunction cannot see — a task dying inside a process that keeps running, where the instance never differs — and a timer is the wrong instrument for it: `_run_assess_job` now gives every job a terminal status in a `finally`, which is a known fact rather than an inferred one and also covers `CancelledError`, which the existing `except Exception` never caught

**Alternatives considered.** The disjunction as originally signed off (fails live jobs on suspend and under horizontal scaling); instance-only (no multi-machine safety); heartbeat-only (no suspend safety)

## Residual closed, WS-1

A machine can be suspended and then simply never routed to again, so its job is stalled whether or not the process still exists — "suspended but unreachable" and "gone" are the same thing to everyone waiting on it, and telling them apart is not worth attempting. Another reader failing that job is therefore correct. What must not happen is the original owner waking later and continuing to spend on a job the user was told failed and has probably retried. `heartbeat()` returns whether the job is still the caller's to run — the guard is on the `UPDATE`, so a terminal row is never touched again — and the beat cancels the trial tasks when it is not. The beat is the natural place for this: it is the one thing a running job already does on a timer

**Alternatives considered.** Querying the Fly machines API for the owner's state (an untestable platform dependency for an undecidable question); a dispute-then-confirm second read (assumes traffic resumes the owner, which it does not when the load balancer routes elsewhere)
