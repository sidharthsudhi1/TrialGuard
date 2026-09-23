# AD-19 — A quote cannot prove absence, so the badge says so

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

A verbatim quote cannot establish absence, and where it pretends to, the badge says so rather than the verdict changing. An exclusion answered `not_met` claims the patient does **not** match a disqualifier; `is_absence_grounded` exists for exactly that claim but runs only as a fallback, so any verbatim quote pre-empts it. Measured across both cohorts: 48 of 107 quote-grounded exclusion `not_met` rows have an absence check that disagrees, 129 of 840 grounded criteria on TREC (15.4%). Making absence primary was the obvious fix and was measured first: about half that class are legitimate refutations it would have destroyed, such as *2+ aortic insufficiency* against *Severe aortic regurgitation*, *18-week sized uterus* against *Uterine size > 32 weeks*, and *no angiographically apparent flow-limiting coronary artery disease* against *Coronary artery disease*. Nothing deterministic separates a refutation from a contradiction: both quote the same subject and differ only by negation, severity, laterality or count. So the row is stamped `weak_absence`, counted per request and in the eval, and rendered in the UI as *quote verified, but it does not establish absence* instead of a clean grounded badge

## Alternatives considered

Absence primary with quote fallback (measured, destroys the correct half); requiring both (stricter still); requiring the quote to share a distinctive term with the criterion (catches three of the clear failures and misses the one that prompted this, whose quote contains *lung*); changing the verdict to unverifiable (the verdict is often right, it is the citation that does not carry it)
