# E5 — widening the gates, and what that caught

Done 2026-09-23. Five changes to what CI enforces, plus the defects the widened
rules found. Cost $0.

Before: `ruff` at `E,F,I`, `mypy` on three hand-picked files of ~10k lines,
`pip-audit` advisory with `continue-on-error: true`, no coverage gate, no
lockfile. That configuration passes whatever it is given, which is the problem —
a gate nobody can fail is not a gate.

## What the widened rules actually found

Not style. Five things worth having fixed:

**1. `assert` as a production state guard** (`eval/file_index.py`, 3 sites).
`assert self._loaded` vanishes under `python -O`, after which the method returns
uninitialised state instead of failing. Converted to an explicit `RuntimeError`.

**2. A silent-truncation class in every `zip`** (21 sites). Ruff's own autofix
writes `strict=False`, which silences the lint and preserves the behaviour —
cargo cult. Applied `strict=True` instead, so mismatched lengths raise rather
than quietly dropping rows, and let the suite judge. One test failed.

**3. A test whose mock violated the contract it was testing**
(`tests/test_refresh.py`). `embed_batch` was a bare `MagicMock`, which returns
one mock object rather than one vector per input, so `refresh()` zipped
mismatched lengths and the test passed anyway. The mock now returns
`[[0.0] * 768 for _ in texts]`. This is the only real find of the `strict=True`
sweep, and it was in the test rather than the code.

**4. An annotation that disagreed with its own function body**
(`eval/retrieval_metrics.py`). `ndcg_at_k(gold_labels: dict[str, int])` is
handed label *names* and does `label_map.get(gold_labels.get(nct))`. The type was
simply wrong and had been for as long as the function existed.

**5. One variable holding two different units** (`llm/cost.py`). `spent` is
tokens on line 202 and dollars on line 214, inside one function. Renamed to
`tokens_spent` and `usd_spent`. Nothing was broken; the next edit to that
function is where it would have gone wrong.

Also: a bare `return` on the corpus-refresh safety-abort path made explicit, and
four stale `# type: ignore` comments removed by `warn_unused_ignores`.

## Dependency CVEs: fixed, not accepted

`pip-audit` was advisory and therefore never read. Run properly it found live
CVEs in the served request path:

| package | was | now | CVEs cleared |
|---|---|---|---|
| aiohttp | 3.14.1 | **3.14.3** | PYSEC-2026-3545/3546/3547 |
| anyio | 4.14.0 | **4.15.1** | CVE-2026-63349/63374/64847 |
| torch | 2.12.1 | **2.14.0** | PYSEC-2025-194 |

All three arrive transitively (langchain, fastapi, sentence-transformers), so
`pyproject.toml` now carries explicit floors — a fresh resolve cannot walk back
to a flagged version.

`pip` and `setuptools` CVEs remain and are skipped by id with a reason in the
workflow: they are the runner's own toolchain, not anything this project ships,
and failing every build for something no commit here can fix would train people
to ignore the gate.

## The gate now

| check | before | after |
|---|---|---|
| ruff | `E,F,I` | `E,F,I,B,ASYNC,S,UP,SIM,RUF,PTH` |
| mypy | 3 files | **all 65 source files**, clean |
| coverage | none | `--cov-fail-under=55` |
| pip-audit | `continue-on-error: true` | **blocking** |
| lockfile | none | `requirements.lock`, 151 pins |

Coverage is **59.24%** overall, and the total is the least interesting number in
this report. Per-module is what matters:

| module | coverage |
|---|---|
| `verify/grounding.py` | **100%** |
| `agent/graph.py` | 97% |
| `api/rate_limit.py` | 97% |
| `agent/schema.py` | 95% |
| `agent/analyst.py` | 90% |

The faithfulness core is fully covered. The 41% uncovered is eval harnesses and
one-shot scripts, where tests are not the right instrument. The floor is set at
55 as a ratchet under the measured 59, so a real regression fails and ordinary
churn does not.

## What was ignored, and why

Ignoring a rule is a decision, so each is recorded in `pyproject.toml` rather
than left bare:

- **`SIM905`, `SIM117`, `PTH123`, `PTH105`** — stylistic. `SIM905` collapses a
  deliberately formatted word list into a 1098-character literal, which is how it
  was found: the autofix applied it and the line-length rule then flagged its own
  output.
- **`RUF001`-`003`** — prose in docstrings legitimately contains en dashes.
- **`RUF100`** — reads per-file-ignored rules as "not enabled" and strips the
  `noqa` directives suppressing them. Running it removed working suppressions and
  reintroduced two errors.
- **`S101` in tests** — `assert` is what a test is.
- **Two mypy module overrides**, for langchain's `SecretStr` constructors and
  langfuse's handler kwargs. Both are runtime-correct; casting to satisfy the
  stubs would change code that every committed result was produced under.

## What this does not claim

**Not `--strict` mypy.** `disallow_untyped_defs` would demand annotations
throughout research code for little return. The configuration catches real
defects and is honest about being a floor.

**59% is not a quality claim.** It is a ratchet anchor. A module can be 100%
covered and wrong.

**The lockfile is not installed by CI**, which resolves from `pyproject.toml` so
upgrades surface rather than being frozen out. It exists so a reported number can
be reproduced against the versions that produced it.
