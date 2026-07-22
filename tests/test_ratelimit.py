import pytest

from trialguard.agent import ratelimit
from trialguard.agent.ratelimit import BudgetExhausted, TokenBudget


def _budget(tmp_path, cap):
    return TokenBudget(path=tmp_path / "budget.json", cap=cap)


def test_record_accumulates_and_reports_remaining(tmp_path):
    b = _budget(tmp_path, cap=1000)
    assert b.spent() == 0 and b.remaining() == 1000
    b.record(300)
    b.record(200)
    assert b.spent() == 500
    assert b.remaining() == 500
    assert not b.exhausted()


def test_check_raises_when_estimate_crosses_cap(tmp_path):
    b = _budget(tmp_path, cap=1000)
    b.record(900)
    b.check(50)  # 950 < 1000, fine
    with pytest.raises(BudgetExhausted):
        b.check(100)  # 900 + 100 >= 1000


def test_exhausted_after_cap_reached(tmp_path):
    b = _budget(tmp_path, cap=500)
    b.record(500)
    assert b.exhausted()
    assert b.remaining() == 0
    with pytest.raises(BudgetExhausted):
        b.check()


def test_new_day_resets_spend(tmp_path, monkeypatch):
    b = _budget(tmp_path, cap=1000)
    monkeypatch.setattr(ratelimit, "_today", lambda: "2026-07-22")
    b.record(800)
    assert b.spent() == 800
    monkeypatch.setattr(ratelimit, "_today", lambda: "2026-07-23")
    assert b.spent() == 0  # rolled over
    assert not b.exhausted()
