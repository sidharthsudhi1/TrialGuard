"""WS-4: audit gates decide whether a complete pull is sane enough to publish."""

from __future__ import annotations

import pytest

from trialguard.ingestion.audit import AuditInput, audit, blocking, stats


def _input(**kw) -> AuditInput:
    base = dict(
        fetched=1000,
        total_count=1000,
        active=1000,
        to_embed=10,
        to_expire=5,
        field_missing={"title": 0, "status": 0, "eligibility_raw": 0, "last_updated": 0},
        statuses=["RECRUITING"] * 1000,
        requested_statuses=["RECRUITING", "NOT_YET_RECRUITING"],
        ids=[f"NCT{i:08d}" for i in range(1000)],
        mean_criteria=20.0,
        current_tag="medcpt_excl",
        corpus_tag="medcpt_excl",
    )
    base.update(kw)
    return AuditInput(**base)


def _gate(gates, name):
    return next(g for g in gates if g.name == name)


@pytest.fixture(autouse=True)
def log_mode(monkeypatch):
    monkeypatch.delenv("TG_REFRESH_GATES", raising=False)


def test_an_ordinary_day_passes_everything():
    gates = audit(_input())
    assert all(g.passed for g in gates), [g for g in gates if not g.passed]


def test_a_renamed_eligibility_field_blocks_even_in_log_mode():
    """F2: every eligibilityCriteria missing would have published empty criteria."""
    gates = audit(_input(field_missing={"eligibility_raw": 1000, "last_updated": 0}))

    names = {g.name for g in blocking(gates)}
    assert "missing_eligibility_raw_hard" in names


def test_calibrated_gates_log_until_enforced(monkeypatch):
    a = _input(to_embed=5000)  # 50% churn

    assert _gate(audit(a), "reembed_churn").passed is False
    assert not blocking(audit(a))

    monkeypatch.setenv("TG_REFRESH_GATES", "enforce")
    assert [g.name for g in blocking(audit(a))] == ["reembed_churn"]


def test_an_embedding_config_leak_blocks_and_cannot_be_allowed():
    """F4: TG_INDEX_EXCLUSION=0 leaking from a TREC run would mix vector spaces."""
    a = _input(current_tag="medcpt_noexcl")

    gates = audit(a, allow=frozenset({"embed_tag_match"}))

    assert _gate(gates, "embed_tag_match").overridden is False
    assert [g.name for g in blocking(gates)] == ["embed_tag_match"]


def test_an_all_unknown_corpus_is_not_a_tag_mismatch():
    assert _gate(audit(_input(corpus_tag=None)), "embed_tag_match").passed


def test_allow_overrides_one_named_gate(monkeypatch):
    monkeypatch.setenv("TG_REFRESH_GATES", "enforce")
    a = _input(to_embed=5000, to_expire=900)

    gates = audit(a, allow=frozenset({"reembed_churn"}))

    assert [g.name for g in blocking(gates)] == ["expiry_churn"]


def test_malformed_ids_and_off_scope_statuses_are_hard():
    a = _input(ids=["NCT1", *[f"NCT{i:08d}" for i in range(999)]],
               statuses=["COMPLETED"] + ["RECRUITING"] * 999)
    names = {g.name for g in blocking(audit(a))}
    assert {"id_format", "status_in_scope"} <= names


def test_bootstrap_skips_the_churn_gates():
    gates = audit(_input(active=0, to_embed=26000, bootstrap=True))
    assert not {"min_size", "reembed_churn", "expiry_churn"} & {g.name for g in gates}


def test_drift_is_measured_against_the_previous_run(monkeypatch):
    monkeypatch.setenv("TG_REFRESH_GATES", "enforce")
    prev = stats(_input())
    a = _input(mean_criteria=12.0, previous=prev)  # parser regression: -40%

    assert "mean_criteria" in {g.name for g in blocking(audit(a))}


def test_a_small_pull_fails_min_size(monkeypatch):
    monkeypatch.setenv("TG_REFRESH_GATES", "enforce")
    a = _input(fetched=800, total_count=800, ids=[f"NCT{i:08d}" for i in range(800)],
               statuses=["RECRUITING"] * 800)
    assert "min_size" in {g.name for g in blocking(audit(a))}
