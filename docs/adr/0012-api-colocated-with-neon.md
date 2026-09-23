# AD-12 — The API is pinned beside Neon

> Part of the TrialGuard architectural decision log ([index](../../README.md#architectural-decision-log)). Amendments are appended here rather than replacing what they correct.

## Decision

API pinned to `syd`, beside Neon in `ap-southeast-2`

## Alternatives considered

Leaving it in `iad`, where 60 round trips per search cost ~15 s in distance alone
