# AD-17 — The analyst cannot be instructed into full coverage

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

The analyst cannot be instructed into answering every criterion. Under v4 it silently returns ~9-13% fewer assessment objects than it was handed on TREC, and v5 fixed that while also changing addressing, so the cause was unattributable. v6 is v4 plus exactly one rule — *return one object per criterion, every criterion* — generated from v4 in code so the single-variable claim is enforced rather than trusted. Over 1,071 paired criteria it moved coverage by −0.65 pp (z = −0.53): nothing, with the two screens disagreeing on the sign. Both v5 and v6 carry that instruction and only v5, which numbers the criteria, recovers coverage (0.9970 against v4's 0.9249). The gain is a property of numbering, not of being told — plausibly because a numbered list is checkable and a bulleted one is not. Kept behind `TG_PROMPT_VERSION=v6`. The remaining candidate is therefore not a prompt: re-ask for the criteria that came back missing, reusing the retry edge that already re-asks for named criteria after a grounding failure

## Alternatives considered

Adopting v5 for its coverage (measured, rejected in AD-16 at 26% of TREC surfaced recall); running v6's full quality A/B (400 calls on a prompt that had already failed its own premise — the cheap mechanism screen is what made this $0.065 instead of $0.23); leaving the shortfall unmeasured on the grounds that the roll-up reports unresolved criteria honestly, which is true and does not stop it calling a trial eligible over a subset
