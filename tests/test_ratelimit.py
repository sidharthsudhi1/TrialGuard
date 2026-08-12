"""Budget primitives and the daily spend circuit breaker (Phase 5 WS-3 / Phase 8 WS-2)."""

import pytest

from trialguard.agent import ratelimit
from trialguard.agent.ratelimit import BudgetExhausted, analyst_delay, estimate_tokens
from trialguard.config import settings
from trialguard.llm import cost
from trialguard.llm.cost import CostLedger, call_usd, table_usd

DEEPINFRA = ("deepinfra", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
GROQ = ("groq", "llama-3.3-70b-versatile")


def _ledger(tmp_path, usd_cap=1.0, token_cap=None):
    return CostLedger(path=tmp_path / "ledger.json", usd_cap=usd_cap, token_cap=token_cap)


def _usage(in_tok=1000, out_tok=500, usd=None, served=None):
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "provider_usd": usd,
        "served_model": served,
    }


# --- Token ceiling (free tier) ------------------------------------------------


def test_token_cap_accumulates_and_blocks(tmp_path):
    """The Groq TPD wall is external: past it the API refuses regardless of budget."""
    led = _ledger(tmp_path, token_cap=1000)
    led.record(_usage(300, 200), *GROQ)  # 500 tokens
    assert led.spent_tokens() == 500
    led.check(400, *GROQ)  # 900 < 1000, fine
    with pytest.raises(BudgetExhausted, match="token cap"):
        led.check(500, *GROQ)


def test_no_token_cap_on_metered_provider(tmp_path):
    """A metered provider has no per-day token wall — only the USD cap bounds it."""
    led = _ledger(tmp_path, usd_cap=10.0, token_cap=None)
    led.record(_usage(500_000, 500_000, usd=0.001), *DEEPINFRA)
    led.check(1_000_000, *DEEPINFRA)  # no raise


# --- USD ceiling --------------------------------------------------------------


def test_usd_cap_blocks_before_the_call(tmp_path):
    """Fails closed on the write path: the point is to make overspend impossible,
    not merely visible afterwards."""
    led = _ledger(tmp_path, usd_cap=0.01)
    led.record(_usage(usd=0.009), *DEEPINFRA)
    with pytest.raises(BudgetExhausted, match="spend cap"):
        led.check(100_000, *DEEPINFRA)


def test_groq_priced_at_zero_never_trips_the_usd_cap(tmp_path):
    """The free tier bills nothing, so it is priced at zero rather than given an
    invented paid-tier rate. Its ceiling is tokens, not money."""
    led = _ledger(tmp_path, usd_cap=0.0001, token_cap=10**9)
    led.record(_usage(50_000, 20_000), *GROQ)
    assert led.spent_usd() == 0.0
    led.check(10_000, *GROQ)  # no raise


def test_remaining_and_exhausted(tmp_path):
    led = _ledger(tmp_path, usd_cap=1.0)
    led.record(_usage(usd=0.25), *DEEPINFRA)
    assert led.remaining_usd() == pytest.approx(0.75)
    assert not led.exhausted()
    led.record(_usage(usd=0.75), *DEEPINFRA)
    assert led.exhausted()


def test_new_day_resets_spend(tmp_path, monkeypatch):
    led = _ledger(tmp_path, usd_cap=1.0)
    monkeypatch.setattr(ratelimit, "_today", lambda: "2026-07-29")
    led.record(_usage(usd=0.8), *DEEPINFRA)
    assert led.spent_usd() == pytest.approx(0.8)
    monkeypatch.setattr(ratelimit, "_today", lambda: "2026-07-30")
    assert led.spent_usd() == 0.0
    assert not led.exhausted()


# --- Pricing ------------------------------------------------------------------


def test_table_price_matches_the_measured_deepinfra_rate():
    """Derived from two live calls: 12/2 -> $1.84e-6 and 25/6 -> $4.42e-6."""
    assert table_usd(*DEEPINFRA, 12, 2) == pytest.approx(1.84e-06)
    assert table_usd(*DEEPINFRA, 25, 6) == pytest.approx(4.42e-06)


def test_provider_reported_cost_wins_over_the_table():
    usd, source = call_usd(_usage(25, 6, usd=4.42e-06), *DEEPINFRA)
    assert source == "provider"
    assert usd == pytest.approx(4.42e-06)


def test_table_used_when_provider_reports_nothing():
    usd, source = call_usd(_usage(1000, 500), *DEEPINFRA)
    assert source == "table"
    assert usd == pytest.approx((1000 * 0.10 + 500 * 0.32) / 1e6)


def test_unpriced_pair_records_zero_and_warns(caplog):
    """Undercounting must be loud. Silently billing $0 for an unknown model is
    how a spend cap stops meaning anything."""
    usd, source = call_usd(_usage(1000, 500), "together", "some/model")
    assert usd == 0.0
    assert source == "unknown"
    assert "not in PRICES" in caplog.text


def test_price_drift_between_provider_and_table_warns(caplog):
    """A stale table makes every figure billed from it wrong, so divergence is
    surfaced rather than silently tolerated."""
    call_usd(_usage(1000, 500, usd=0.05), *DEEPINFRA)  # table says ~0.00031
    assert "drift" in caplog.text.lower()


def test_no_warning_when_provider_and_table_agree(caplog):
    expected = table_usd(*DEEPINFRA, 1000, 500)
    call_usd(_usage(1000, 500, usd=expected), *DEEPINFRA)
    assert "drift" not in caplog.text.lower()


# --- Attribution --------------------------------------------------------------


def test_spend_is_keyed_by_the_served_model_not_the_requested_one(tmp_path):
    """DeepInfra aliases "-Instruct" to the FP8 "-Instruct-Turbo" build with no
    warning; keying on what was served is how that becomes visible."""
    led = _ledger(tmp_path)
    led.record(
        _usage(usd=0.001, served="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        "deepinfra",
        "meta-llama/Llama-3.3-70B-Instruct",  # requested the alias
    )
    keys = list(led.summary()["by_model"])
    assert keys == ["deepinfra|meta-llama/Llama-3.3-70B-Instruct-Turbo"]


def test_summary_reports_calls_and_cap(tmp_path):
    led = _ledger(tmp_path, usd_cap=2.0)
    led.record(_usage(usd=0.01), *DEEPINFRA)
    led.record(_usage(usd=0.02), *DEEPINFRA)
    s = led.summary()
    assert s["calls"] == 2
    assert s["usd"] == pytest.approx(0.03)
    assert s["usd_cap"] == 2.0


# --- Pacing -------------------------------------------------------------------


def test_analyst_delay_paces_groq_but_not_metered(monkeypatch):
    monkeypatch.delenv("TG_ANALYST_DELAY", raising=False)
    monkeypatch.setattr(settings, "llm_provider", "groq")
    assert analyst_delay() == ratelimit.GROQ_ANALYST_DELAY
    monkeypatch.setattr(settings, "llm_provider", "deepinfra")
    assert analyst_delay() == 0.0


def test_explicit_delay_env_wins(monkeypatch):
    """app.py sets this for the Spaces demo, which runs on Groq but issues one
    interactive call at a time and needs no batch pacing."""
    monkeypatch.setenv("TG_ANALYST_DELAY", "0")
    monkeypatch.setattr(settings, "llm_provider", "groq")
    assert analyst_delay() == 0.0


def test_estimate_tokens_is_gate_only():
    assert estimate_tokens("a" * 400) == 100


def test_active_ledger_applies_token_cap_only_on_groq(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "groq")
    assert cost.active_ledger().token_cap == ratelimit.GROQ_TPD_CAP
    monkeypatch.setattr(settings, "llm_provider", "deepinfra")
    assert cost.active_ledger().token_cap is None
