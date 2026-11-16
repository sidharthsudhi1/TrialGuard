"""Bars written before the E4 and R4 runs. These tests lock the formulas, not a score."""

from trialguard.eval.cascade import drop_kills, drop_rate, seventy_b_count, tail_depth
from trialguard.eval.e4 import clears_kill, entail_from_tokens, sentence_containing


def test_sentence_is_the_one_holding_the_quote():
    note = "Age 58. HER2 positive disease. No mets."
    assert sentence_containing(note, "HER2 positive disease") == "HER2 positive disease."


def test_missing_quote_does_not_invent_a_sentence():
    assert sentence_containing("Age 58. No mets.", "HER2 positive") is None


def test_hallucination_token_lowers_entailment():
    tokens = [
        {"pred": 0, "prob": 0.99},
        {"pred": 1, "prob": 0.4},
        {"label": "hallucination", "confidence": 0.7},
    ]
    assert entail_from_tokens(tokens) == 0.3


def test_no_hallucination_token_is_full_entailment():
    assert entail_from_tokens([{"pred": 0, "prob": 0.2}]) == 1.0


def test_kill_needs_both_damage_and_catch():
    assert clears_kill([{"sound_citations_destroyed": 2, "recall": 0.25}])
    assert not clears_kill([{"sound_citations_destroyed": 3, "recall": 0.9}])
    assert not clears_kill([{"sound_citations_destroyed": 0, "recall": 0.24}])


def test_tail_depth_fits_the_reserve():
    # 2.70 / (0.0002 * 1.5 * 125) = 72
    assert tail_depth(0.0002, 125) == 72
    assert tail_depth(0.00001, 10) == 400


def test_seventy_b_count_respects_the_floor():
    assert seventy_b_count(0.24, 0.0005, 125, 72) == 0
    # (0.80 - 0.05) / (0.0005 * 1.3 * 125) = 9
    assert seventy_b_count(0.80, 0.0005, 125, 72) == 9


def test_drop_rate_counts_only_gold_eligible():
    rows = [
        {"gold": "eligible", "tier": "excluded"},
        {"gold": "eligible", "tier": "needs_review"},
        {"gold": "excluded", "tier": "excluded"},
    ]
    assert drop_rate(rows) == {"eligible": 2, "excluded": 1, "rate": 0.5}
    assert drop_kills(0.0501)
    assert not drop_kills(0.05)
    assert not drop_kills(None)
