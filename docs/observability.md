# Observability (Phase 5, WS-2)

Every graph run is traced to Langfuse (`tracing.get_langchain_handler`), and every
eval run pushes run-level quality scores next to those traces
(`tracing.emit_scores`, called from `eval/agent_metrics.run`). Cost, latency and
retry structure come for free from the traces; faithfulness and coverage are
pushed as custom scores. Together they make one board where a quality regression
and a cost/latency change are visible side by side.

Tracing is no-op without credentials, so nothing here is required to run the
system — it is the ops layer, not a dependency.

## Custom scores (pushed per run, linked to the eval session)

Session id: `agent-eval-<cohort>`. Scores emitted by `_observability` over the
verified (thesis) arm:

| Score | Type | Source | Reads as |
|---|---|---|---|
| `faithfulness` | NUMERIC | verified `citation_precision` | grounded decisive verdicts / decisive attempts; the headline |
| `unsupported_verdict_rate` | NUMERIC | verified arm | hallucinated-citation proxy; `1 - faithfulness` |
| `abstention_rate` | NUMERIC | verified arm | coverage inverse; watch it does not creep up to fake faithfulness |
| `coverage` | NUMERIC | verified arm | criteria ending as a grounded decisive verdict |
| `mean_retries` | NUMERIC | verified arm | how hard the grounding back-edge worked per trial |

The same numbers are written into the report JSON under `observability`, so the
ops story survives even with tracing off.

## Native trace signals (no extra code)

Langfuse aggregates these from the spans the LangChain handler already emits:

- **Cost / token usage** per trace, session, and model (Groq is $0, but token
  volume is real and is what the free-tier quota is spent against).
- **Latency** p50 / p95 per trace and per node (analyst vs retry vs report).
- **Retry structure**: each retry is a `retry -> analyst` span pair in the trace
  tree, so retry depth is visible per trial without instrumentation; `mean_retries`
  is the run-level aggregate.

## Dashboard to build (manual, one time)

Langfuse → Dashboards → New. Panels, all filtered to `tags = agent-eval`:

1. **Faithfulness over time** — line, score `faithfulness`, grouped by session.
2. **Abstention vs coverage** — two lines, scores `abstention_rate` and `coverage`;
   the trade-off Phase 4 argued must be read jointly.
3. **Retry depth** — bar, score `mean_retries` by cohort.
4. **Cost / token volume** — Langfuse cost metric per session.
5. **Latency p50/p95** — Langfuse latency metric, broken down by node.

Optional: a monitor on `faithfulness` (alert if it drops below the
`baselines.json` floor) fired to Slack/GitHub — the runtime complement to the CI
regression gate, which guards the committed report.

Screenshot the finished board into `docs/observability_dashboard.png` for the
README / Phase 6 walkthrough. (A live board needs real trace volume; capture it
during the Phase 6 demo run.)
