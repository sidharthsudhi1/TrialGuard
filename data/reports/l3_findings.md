# L3 — schema-constrained JSON output, declined on measurement

Measured 2026-08-31 over the 2,622 responses in `data/cache/analyst/`. No LLM
calls; the question was whether the problem L3 exists to solve is present.

The review predicted L3 would "remove a class of parse failure" and "likely
shorten output". Neither holds.

## Output is nowhere near the cap

`_salvage` exists for one named failure mode. Its docstring: *"LLM output
truncated at the token cap mid-array."*

| | estimated output tokens |
|---|---|
| median | 399 |
| p90 | 1,092 |
| p99 | 1,756 |
| max | 3,783 |
| **cap (`_MAX_TOKENS["analyst"]`)** | **4,096** |

Responses within 10% of the cap: **2 of 2,622 (0.08%)**. At or over it: **0**.

Truncation is not happening, so there is no length pressure for structured
output to relieve — and `response_format` constrains grammar, not length. It
could not fix truncation even if it were occurring.

## The parse-failure class is 0.88%

23 of 2,622 cached responses are empty, which is the upper bound on what
`response_format` could recover — and only if every one is a JSON error rather
than the model genuinely returning nothing.

## Why that is not worth doing

Against <=0.88% upside, L3 costs a changed model call, a new cache namespace (the
key in `_cache_key` does not include `response_format`, so constrained and
unconstrained entries would mix unattributably), an A/B on both cohorts against
the committed faithfulness floors, and the spend to run it.

It also cannot fix the thing that *is* measurably wrong. A2 found 22% of
responses return a different criterion *count* than was asked. No JSON schema can
require the output list to match the input list, so structured output leaves that
untouched.

## Incidental

13 responses (0.50%) carry more assessments than the current `MAX_CRITERIA=24`
allows, up to 66. File dates spread across July and August, so these are stale
entries from an earlier cap rather than a live defect. A2's anchoring guard
already makes them safe: unmatched criteria resolve to `unknown` and land in
`needs_review` instead of being mislabelled by position.

## Standing

`response_format` is not set and the analyst call is unchanged. Revisit only if
the empty-response rate rises materially above 0.88%, or if a future prompt
pushes output toward the cap — L1 (dropping the echoed criterion, 43.4% of output
characters) moves output the other way, so that is unlikely.
