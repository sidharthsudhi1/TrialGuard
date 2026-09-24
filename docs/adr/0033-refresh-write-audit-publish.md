# AD-33 — The corpus refresh is write-audit-publish over content hashes

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

The refresh proves a crawl complete, diffs on content rather than dates, confirms every expiry with CT.gov, audits the result, and publishes it in one transaction. The previous refresh trusted whatever the crawl returned. It stopped quietly on a missing `nextPageToken` and truncated quietly at `max_trials`, and its only guard was a 50% count check, so a pull between half and all of the corpus deleted the rest. It diffed on `lastUpdatePostDate`, which is day-granular and says nothing about changes on our side. And the serving process loaded its vector matrix once and never again.

- **Completeness.** `pull_trials` requires `fetched == countTotal` and an unchanged `/version` `dataTimestamp` across the crawl, with one re-crawl allowed, and raises otherwise. Exceeding the cap raises instead of truncating. At `pageSize=1000` (the API cap) a crawl is 27 requests: live, 26,107 of 26,107 in 75 s, down from ~260 requests.
- **Provenance and the diff.** Every row records `doc_hash` (the exact string embedded), `content_hash`, `embed_tag` and `parser_version`. The migration verified the stored tag rather than assuming it: 200 rows re-embedded matched at cosine 1.0. A trial is re-embedded only when its `doc_hash` or tag moves. Everything else that changed is a metadata rewrite with no model call.
- **What the diff found on its first dry run.** 4,551 trials would re-embed. Of those, **4,537 had byte-identical raw eligibility text**: they were stored with criteria from parsers that were live 26 Jul – 16 Sep. 4,517 had an unchanged update date, so the date diff would never have corrected them. That is 17.4% of the served corpus answering from an outdated parse, invisible until rows recorded which parser built them. Only 14 were real CT.gov revisions. `PARSER_VERSION` is now pinned by a golden test over 30 real texts, so a parser change forces a version bump and the next refresh re-parses.
- **Expiry.** Expiry is soft (`expired_at`). A trial missing from a complete crawl is looked up by id and expires only on CT.gov's own word: a status outside scope, or no record at all. A trial that is still enrolling but has left the condition query expires after three consecutive misses. The same dry run expired 15, every one confirmed (6 COMPLETED, 5 ACTIVE_NOT_RECRUITING, 2 WITHDRAWN, 1 TERMINATED, 1 SUSPENDED). A returning trial with unchanged text costs no embedding call.
- **Audit gates.** Hard gates always block: malformed ids, off-scope statuses, an `embed_tag` that differs from the corpus's, and a content field missing from over half the pull. Calibrated gates (churn, field-missing drift, mean criteria per trial) log until `TG_REFRESH_GATES=enforce`, so that real baselines decide their thresholds. `embed_tag_match` cannot be overridden.
- **Publish and consumers.** Publish is one transaction under a transaction-scoped advisory lock; Neon's pooler cannot hold session locks. `refresh_runs` records every run, including aborts and lease conflicts. `/api/health` reports the last attempt and the failure streak, and the probe alerts on the second consecutive failure, not when the success stamp turns 48 h old. The serving matrix polls the published corpus version and swaps in a reload. The served analyst cache key carries a criteria fingerprint (`TG_CACHE_KEY_CRITERIA=1` on Fly only, so eval keys are byte-identical).
- **Schedule.** Hourly. A run where CT.gov has not published since the last success is one `/version` request.

## Alternatives considered

- **Tightening the count guard from 0.5 to 0.9 and keeping the date diff.** It still reads absence as expiry and cannot see parser drift, which the dry run measured as the larger defect by 324x.
- **K-consecutive-miss expiry for every missing trial.** This delays real expiries by K days, while the id lookup answers in one request per 100 ids.
- **A session advisory lock.** Not held on Neon's PgBouncer transaction pooling.
- **Enforcing every gate from day one.** Thresholds guessed without baselines would either block ordinary days or pass real failures. The hard gates cover the failures that are never legitimate.
- **Keying the analyst cache on criteria everywhere.** This re-rolls every committed eval key; `LEGACY_PAIR` and `|s1` already set the additive precedent.
- **Swapping ivfflat for HNSW.** The planner declines the ivfflat index at `probes=40` and the in-process matrix serves production dense search (AD-25), so the index only affects the SQL fallback. It is now built once data exists, instead of being trained on an empty table.

## Amendment: first production run (2026-09-24)

Run `0cbcccfb` published in 79 min with every count equal to the dry run: 27 new, 4,551 re-embedded, 21,529 metadata-only, and 15 expired, each confirmed. Embedding on the Fly performance-2x machine runs at about 1 trial/s (8.4 min per 500-trial chunk), roughly 5x slower than the Mac throughput the plan extrapolated from. The chunk heartbeat still sits well inside the 30-min lease, but a full-corpus re-embed would take about 7 h. The serving matrix reloaded to the new version (26,107 rows, 6.9 s) on its first search after the publish. The post-deploy SLO gate reads `served_slo.json`, which had not been given the two new gates; a test now requires both thresholds files to carry the same gates.
