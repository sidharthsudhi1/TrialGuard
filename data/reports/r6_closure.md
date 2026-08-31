# R6 — closed on existing evidence

The pipeline review's R6 asked for `ts_rank_cd` to be "validated against gold
directly". That validation already existed when the review was written.

`phase7_retrieval.md` **WS-4a** loaded SIGIR's 2,991 trials into Postgres under
`source='sigir'` — the same corpus the FileIndex eval uses, so the corpus is held
constant — and ran recall through the production `retrieve()` path (pgvector
dense + FTS lexical + RRF, keyword queries), n=53, gold coverage 0.9635.

| metric | FileIndex (rank-bm25) | Production (FTS + pgvector) |
|---|---|---|
| recall@10 | 0.1345 | **0.1775** |
| recall@100 (adj) | 0.72 | 0.7165 |
| recall@200 (adj) | 0.81 | 0.798 |
| MRR | — | 0.327 |

Recall at depth is within noise of the rank-bm25 baseline and recall@10 improves
by +0.043. Under short keyword queries the lexical arm's contribution flips
positive, and FTS captures it. WS-4a's own conclusion: "No fallback (ParadeDB /
ts_rank weight tuning) needed."

## What was actually wrong

The review's *fact* is correct — `ts_rank_cd` has no IDF saturation and no length
normalisation, so it is not Okapi BM25, and `bm25.py`'s docstring says so. Its
*fix* was already satisfied, and `bm25.py` names the report that satisfied it.

The re-ranked plan then compounded this by listing R6 as P0 on the grounds that
the production lexical arm "has never been scored". That claim was not in the
review; it was added while ranking, without opening the referenced report.

Second occurrence of the same failure this week: G-5 in
`eval_gaps_and_priorities.md` had already retired the 87% criterion-matching
target, and was likewise rediscovered rather than read. Both times the repository
was right and the reader was not.

## What remains

Restoring true BM25 semantics is untested. It is a new experiment rather than a
validation, and R5 and R3 have since shown this fusion stage does not respond to
tuning — three levers, at most a few percent, none significant, one inverting
between cohorts. Not ranked.
