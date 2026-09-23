# AD-22 — Parser and matching fixes, measured together

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

Measured together, the parser and matching fixes take criteria never answered from **230 to 1 on TREC** (13.1% to 0.06%) and 43 to 2 on SIGIR, so roughly **99% of the shortfall was ours, not the analyst's**. Two of my own estimates were too high and are corrected here. The 3.3% genuine-omission figure in AD-21 is really 0.9%, measured with the retry off: the decomposition behind it used a cruder containment test than `align_assessments` and a length heuristic for headers, and attributed to the model what better matching resolves. And AD-18's recorded penalty of 12-14% of TREC surfaced recall **disappears and reverses** once the parser is fixed -- the retry now improves surfaced recall 0.0303 to 0.0316 and precision 0.6389 to 0.6486, because the old penalty was it dutifully re-asking about section headers and manufacturing noise from them. SIGIR improves on every axis; TREC improves on every criterion-level axis and gives back surfaced precision 0.6944 to 0.6486, which sits inside the 0.6267-0.7097 spread this system showed across configurations during the investigation and is not read either way on 20 patients

## Alternatives considered

Reporting the end state without re-measuring the retry, which would have left a cost in the log that no longer exists; treating the TREC trial-level dip as a regression (inside the observed spread) or as noise (unmeasured either way)
