import json
from unittest.mock import patch

from trialguard.agent import graph as G

SRC = "Inclusion Criteria: Histologically confirmed melanoma. Age 18 or older."


def _fake_analyst(grounded_quote):
    def _fn(note, nct_id, criteria, handler=None, **kw):
        # First attempt returns an ungrounded quote; retries return a grounded one.
        if "Retry" in note:
            return [{"criterion": "dx", "verdict": "met", "quote": "Histologically confirmed melanoma"}]
        return [{"criterion": "dx", "verdict": "met", "quote": grounded_quote}]
    return _fn


def test_single_pass_flags_unverifiable(monkeypatch):
    G._GRAPH = None
    with patch.object(G, "analyze_trial", _fake_analyst("fabricated diagnosis text here")):
        state = G.assess("patient", "NCT1", ["dx"], SRC, max_retries=0)
    a = state["assessments"][0]
    assert a["verdict"] == "unverifiable"
    assert state["trial_verdict"] == "cannot_determine"


def test_verified_retry_recovers_grounding(monkeypatch):
    G._GRAPH = None
    with patch.object(G, "analyze_trial", _fake_analyst("fabricated diagnosis text here")):
        state = G.assess("patient", "NCT1", ["dx"], SRC, max_retries=2)
    a = state["assessments"][0]
    assert a["verdict"] == "met" and a["grounded"]
    assert state["trial_verdict"] == "eligible"


def test_retry_capped(monkeypatch):
    G._GRAPH = None
    def always_bad(note, nct, crit, handler=None, **kw):
        return [{"criterion": "dx", "verdict": "met", "quote": "never in source at all here"}]
    with patch.object(G, "analyze_trial", always_bad):
        state = G.assess("patient", "NCT1", ["dx"], SRC, max_retries=2)
    # exhausted retries -> unverifiable, never forced to a verdict
    assert state["assessments"][0]["verdict"] == "unverifiable"


def test_retry_is_retrieval_aware(monkeypatch):
    # On retry the analyst must be handed the exact source span and the failed
    # criterion, not a generic nudge.
    G._GRAPH = None
    seen_notes = []

    def _capture(note, nct_id, criteria, handler=None, **kw):
        seen_notes.append(note)
        if "Retry" in note:
            return [{"criterion": "dx", "verdict": "met", "quote": "Histologically confirmed melanoma"}]
        return [{"criterion": "dx", "verdict": "met", "quote": "not in the source"}]

    with patch.object(G, "analyze_trial", _capture):
        G.assess("patient", "NCT1", ["dx"], SRC, max_retries=2)

    retry_note = next(n for n in seen_notes if "Retry" in n)
    assert SRC in retry_note            # the retrieved source span is injected
    assert "dx" in retry_note           # the specific failed criterion is named


def test_prompt_version_switches_cache_key(monkeypatch):
    from trialguard.agent import analyst as A
    monkeypatch.delenv("TG_PROMPT_VERSION", raising=False)
    assert A.prompt_version() == "v1"
    k1 = A._cache_key("note", "NCT1")
    monkeypatch.setenv("TG_PROMPT_VERSION", "v2")
    assert A.prompt_version() == "v2"
    assert A._cache_key("note", "NCT1") != k1   # v2 never collides with v1 cache
    assert "v2" in A._PROMPTS


def test_analyst_parse_salvages_truncated_json():
    from trialguard.agent.analyst import _parse
    good = '{"assessments":[{"criterion":"a","verdict":"met","quote":"x"},{"criterion":"b","verdict":"not_met","quote":"y"}]}'
    assert len(_parse(good)) == 2
    trunc = '{"assessments":[{"criterion":"a","verdict":"met","quote":"stage IV"},{"criterion":"b","verdict":"met","quote":"z"},{"criterion":"c","verdict":"met","quote":"cut off her'
    r = _parse(trunc)
    assert [o["criterion"] for o in r] == ["a", "b"]


def test_exclusion_met_excludes_trial():
    from trialguard.agent.schema import rollup_trial_verdict

    assert (
        rollup_trial_verdict(
            [
                {"kind": "inclusion", "verdict": "met"},
                {"kind": "exclusion", "verdict": "met"},
            ]
        )
        == "excluded"
    )


def test_exclusion_not_met_allows_eligible():
    from trialguard.agent.schema import rollup_trial_verdict

    assert (
        rollup_trial_verdict(
            [
                {"kind": "inclusion", "verdict": "met"},
                {"kind": "exclusion", "verdict": "not_met"},
            ]
        )
        == "eligible"
    )


def test_build_typed_criteria_includes_exclusion():
    from trialguard.agent.schema import build_typed_criteria

    trial = {
        "inclusion_criteria": ["Age >= 18"],
        "exclusion_criteria": ["Brain metastases"],
    }
    crit, truncated = build_typed_criteria(trial)
    assert not truncated
    assert crit == [
        {"text": "Age >= 18", "kind": "inclusion"},
        {"text": "Brain metastases", "kind": "exclusion"},
    ]


def test_criteria_truncation_flagged():
    """Truncation is reported, and the one exclusion criterion survives it.

    This asserted `all(kind == "inclusion")` until 2026-09-12, which codified the
    defect rather than the contract: the cap was filled with inclusion criteria
    and every disqualifier dropped. A trial that keeps none of its exclusion
    criteria can only come back eligible or cannot_determine.
    """
    from trialguard.agent.schema import MAX_CRITERIA, build_typed_criteria

    trial = {
        "inclusion_criteria": [f"inc{i}" for i in range(MAX_CRITERIA)],
        "exclusion_criteria": ["exc0"],
    }
    crit, truncated = build_typed_criteria(trial)
    assert truncated
    assert len(crit) == MAX_CRITERIA
    assert sum(1 for c in crit if c["kind"] == "exclusion") == 1


def test_cache_write_policy_does_not_leak_between_concurrent_assessments():
    """Two assessments in flight must not inherit each other's cache-write policy.

    This was process state: the caller set TG_SKIP_ANALYST_CACHE_WRITE around a
    request and cleared it afterwards. With a second request overlapping, the
    first one finishing cleared the flag out from under it, and a free-text note
    the policy said to never persist got written to disk anyway. The barrier
    forces exactly that overlap.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    seen: dict[str, bool] = {}
    both_inside = threading.Barrier(2, timeout=10)

    def _analyst(note, nct_id, criteria, handler=None, skip_cache_write=False):
        both_inside.wait()  # neither returns until both are mid-flight
        seen[nct_id] = skip_cache_write
        return [
            {
                "criterion": "Histologically confirmed melanoma",
                "verdict": "met",
                "quote": "Histologically confirmed melanoma",
                "rationale": "stated",
            }
        ]

    with patch.object(G, "analyze_trial", _analyst), ThreadPoolExecutor(2) as ex:
        futures = [
            ex.submit(
                G.assess, "free text", "NCT-FREE", ["c"], SRC,
                max_retries=0, skip_cache_write=True,
            ),
            ex.submit(
                G.assess, "preset", "NCT-PRESET", ["c"], SRC,
                max_retries=0, skip_cache_write=False,
            ),
        ]
        [f.result() for f in futures]

    assert seen == {"NCT-FREE": True, "NCT-PRESET": False}, seen


# --- L5: analyst cache survives a deploy via Postgres ---


def _fake_llm(payload):
    """Chat model stub. The analyst parses {"assessments": [...]}, not a bare list."""
    from unittest.mock import MagicMock

    m = MagicMock()
    m.invoke.return_value.content = json.dumps({"assessments": payload})
    return m


def test_analyst_reads_postgres_when_disk_is_cold(tmp_path, monkeypatch):
    """Fly replaces the filesystem on every deploy; disk-only meant every preset
    became a fresh paid call after each release."""
    from unittest.mock import patch as _patch

    from trialguard.agent import analyst

    monkeypatch.setattr(analyst, "CACHE_DIR", tmp_path / "cold")
    rows = [{"criterion": "age >= 18", "verdict": "met", "quote": "40-year-old"}]
    with (
        _patch("trialguard.db.cache.cache_get", return_value=rows) as cg,
        _patch.object(analyst, "_llm") as llm,
    ):
        got = analyst.analyze_trial("synthetic note", "NCT0001", ["age >= 18"])

    assert got == rows
    assert cg.call_args[0][0] == "analyst"
    llm.assert_not_called()  # the whole point: no LLM call


def test_analyst_ignores_a_postgres_row_of_the_wrong_shape(tmp_path, monkeypatch):
    """A shared durable store must not hand grounding something that is not a
    list of assessments; fall through to the model instead."""
    from unittest.mock import patch as _patch

    from trialguard.agent import analyst

    monkeypatch.setattr(analyst, "CACHE_DIR", tmp_path / "cold2")
    fresh = [{"criterion": "c", "verdict": "met", "quote": "q"}]
    with (
        _patch("trialguard.db.cache.cache_get", return_value={"not": "a list"}),
        _patch("trialguard.db.cache.cache_put"),
        _patch("trialguard.llm.cost.active_ledger"),
        _patch.object(analyst, "_llm", return_value=_fake_llm(fresh)) as llm,
    ):
        got = analyst.analyze_trial("synthetic note", "NCT0001", ["c"])

    # validate_assessments normalises the model's rows, so compare the fields
    # that matter rather than object identity.
    assert [a["criterion"] for a in got] == ["c"]
    assert got[0]["verdict"] == "met"
    llm.assert_called()  # it recomputed rather than trusting the bad row


def test_analyst_postgres_write_honours_skip_cache_write(tmp_path, monkeypatch):
    """skip_cache_write exists so a free-text note is never persisted. A shared
    durable store is a stronger reason to honour it, not a weaker one."""
    from unittest.mock import patch as _patch

    from trialguard.agent import analyst

    monkeypatch.setattr(analyst, "CACHE_DIR", tmp_path / "nowrite")
    fresh = [{"criterion": "c", "verdict": "met", "quote": "q"}]
    with (
        _patch("trialguard.db.cache.cache_get", return_value=None),
        _patch("trialguard.db.cache.cache_put") as cp,
        _patch("trialguard.llm.cost.active_ledger"),
        _patch.object(analyst, "_llm", return_value=_fake_llm(fresh)),
    ):
        analyst.analyze_trial("free text note", "NCT0002", ["c"], skip_cache_write=True)

    cp.assert_not_called()


def test_analyst_writes_both_stores_for_a_cacheable_note(tmp_path, monkeypatch):
    from unittest.mock import patch as _patch

    from trialguard.agent import analyst

    cache_dir = tmp_path / "write"
    monkeypatch.setattr(analyst, "CACHE_DIR", cache_dir)
    monkeypatch.delenv("TG_SKIP_ANALYST_CACHE_WRITE", raising=False)
    fresh = [{"criterion": "c", "verdict": "met", "quote": "q"}]
    with (
        _patch("trialguard.db.cache.cache_get", return_value=None),
        _patch("trialguard.db.cache.cache_put") as cp,
        _patch("trialguard.llm.cost.active_ledger"),
        _patch.object(analyst, "_llm", return_value=_fake_llm(fresh)),
    ):
        analyst.analyze_trial("synthetic note", "NCT0003", ["c"])

    assert list(cache_dir.glob("*.json")), "disk cache still authoritative"
    assert cp.call_args[0][0] == "analyst"
