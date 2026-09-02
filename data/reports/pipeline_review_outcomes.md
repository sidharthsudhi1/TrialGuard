# Pipeline review — outcomes

Resolution of every finding in the 2026-08-19 retrieval/orchestration review,
closed out 2026-09-01. The review itself is a plan document and stays out of the
repo; this records what was measured and decided.

## Findings

| | finding | outcome |
|---|---|---|
| R1 | 42% of trials truncated before embedding | **Rejected.** Chunking measured on TREC 2021 (n=75): recall@50 0.2859 -> 0.2853, MRR 0.6257 -> 0.6197. The truncation is real; it is not a cause. `chunkab_trec2021_*.json` |
| R2 | structured age/sex never filter candidates | **Shipped.** `pipeline.py:113`, fails open on missing metadata |
| R3 | keyword rank generated then discarded | **Tested, not adopted.** Every graded weighting beats uniform on both cohorts (+10.8% SIGIR, +3.4% TREC) but paired per-patient tests give p=0.0673 / p=0.3031, Fisher combined **p=0.0998**. `r3_findings.md` |
| R4 | reranking was tried and it hurt | **Re-tested, still loses.** The prior result varied model *and* query form. MedCPT recovers -33.0% to -9.5%; query form contributes nothing. 21s/patient on 8 vCPU disqualifies it regardless. `r4_findings.md` |
| R5 | fusion constants never swept | **Defaults kept.** 40 cells, both cohorts. SIGIR's best cell (k=10) gains 5.9% and loses 3.3% on TREC — they disagree in sign. `r5_findings.md` |
| R6 | `ts_rank_cd` is not Okapi BM25 | **Already validated** in `phase7_retrieval.md` WS-4a before the review was written. `r6_closure.md` |
| L1 | most analyst output is not evidence | Open. Recount over 16,456 assessments: quote is 7.8% of output chars, criterion echo 43.4% |
| L2 | trials assessed sequentially | **Shipped.** `c6ea39f`, with the worker count raised to the per-request cap afterwards |
| L3 | JSON requested in prose, not enforced | **Declined on measurement.** Output never approaches the 4096 cap (median 399, max 3783); recoverable failure class is 23 empty responses, 0.88%. `l3_findings.md` |
| L4 | retries re-run every criterion | **Measured, rejected.** Partial retry cuts the unverifiable rate 26% but `grounded` falls (845 -> 838): the failures become abstentions, not recoveries, costing 11 decisive `not_met` verdicts and 0.38x of tier lift. `l4_findings.md` |
| L5 | analyst cache does not survive a deploy | **Shipped.** `71bc29e` |
| L6 | no per-criterion streaming | Open |
| E1 | retrieval and agent never evaluated together | **Shipped**, and it exposed the parser bug below |
| E2 | served abstention unmeasured | **Shipped.** `served_monitor.py` reads the `served`-tagged traces nothing read before and diverges non-zero outside committed bands; daily workflow. First reading: abstention 0.22 on 7 trials vs eval 0.56-0.61, under the traffic floor |
| E3 | gate protects faithfulness only | **Recall half shipped.** CI now fails below recall@50 0.27 / recall@100 0.40 against the committed TREC 2021 report, bite proven in tests. Latency and cost floors remain post-deploy concerns, as the review specified |

## Not in the review

The four defects that mattered most were found by reading code and cached output,
not from this document:

- **`c8d9779`** — SIGIR strips `exclusion criteria` from its headers, leaving a
  bare `:`. All 36,826 criteria were filed as inclusions, so a correct "patient
  does not have <disqualifier>" read as a failed requirement and excluded
  eligible trials, across 2,991 trials.
- **`e3263fc`** — `attach_kinds` guessed criterion kind by position on 5.8% of
  assessments; a wrong guess inverts exclusion semantics.
- **`ba49b1d`** — disjunctive criteria ("any of the following:") were split into
  siblings and ANDed, so matching one alternative failed the rest.
- **`19c02a9`** — the rerank score cache was keyed on the note alone, so a second
  model would read back the first model's scores as its own.

Plus **A5** (`16d8cef`), the tiered contract, and **E1b** (`d457bc0`), the TREC
replication that retired A1's abstention diagnosis.

## What the measurements established

**Seven of eight experiments returned negative or already-solved.** The review's
component diagnoses were consistently accurate; its estimates of what they cost
were consistently wrong.

**The fusion stage does not respond to tuning.** Four independent levers — RRF
constant, pool depth, per-list weighting, and reranking the pool outright —
produced no significant gain, and one inverted between cohorts.

**The remaining gap is structural.** TREC retrieval delivers 0.048 against its
own 0.1316 top-10 ceiling, and the agent converts 12% of what it is handed
(`e1b_findings.md`). Neither is addressable by rearranging how existing lists
combine.

> **ANSWERED 2026-09-02.** This paragraph named three candidate causes — a
> different retrieval model, a different query representation, or a larger pool
> passed to the agent — and called the question unopened. It was the third, and
> it is now measured on both cohorts. Gold is not missing from the candidate
> space (recall 0.766@500 on TREC 2021, 0.788@500 on TREC 2022, under 6%
> unranked): it is present and misordered, so the binding constraint was the
> arbitrary k=10 hand-off, not the representation. Widening the assessed pool
> 10 -> 100 moves surfaced recall 0.0276 -> 0.1770 (6.4x) on 2021 and
> 0.0430 -> 0.2580 (6.0x) on 2022, with conversion and lift over pool base rate
> flat to rising through the full 10x dilution. The `0.048 of 0.1316` framing
> above measures a top-10 slice of a ranking whose gold sits at median rank
> ~130-180; it was never the system's ceiling. What is left is serving
> economics, not recall. See `h1_pool_curve_trec2021.md`,
> `h1_pool_curve_trec2022.md`, `gold_rank_depth_trec{2021,2022}.json`, and
> `docs/structural_recall_plan.md`.

**SIGIR-only conclusions have a poor record.** A1's abstention finding died on
TREC; R5's +5.9% inverted to -3.3%; R3's +10.8% shrank to +3.4% and lost
significance. TREC 2021 is the cohort of record, and every SIGIR trial-level
number predating `c8d9779` is not reproducible.
