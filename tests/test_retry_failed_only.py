from trialguard.agent.graph import _merge_retry, _retry_failed_only


def _a(criterion, verdict="met", failure=False, quote="q"):
    return {
        "criterion": criterion,
        "verdict": verdict,
        "quote": quote,
        "grounding_failure": failure,
    }


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("TG_RETRY_FAILED_ONLY", raising=False)
    assert _retry_failed_only() is False


def test_merge_preserves_passing_verdicts():
    prior = [_a("A"), _a("B", "not_met", failure=True), _a("C")]
    retried = [_a("B", "not_met", quote="found it")]
    out = _merge_retry(prior, retried)
    assert [x["criterion"] for x in out] == ["A", "B", "C"]
    assert out[0] is prior[0] and out[2] is prior[2]
    assert out[1]["quote"] == "found it"


def test_merge_matches_by_criterion_text_not_position():
    prior = [_a("A", failure=True), _a("B"), _a("C", failure=True)]
    # Model returns the two failed criteria in the opposite order.
    retried = [_a("C", quote="c-quote"), _a("A", quote="a-quote")]
    out = _merge_retry(prior, retried)
    assert out[0]["quote"] == "a-quote"
    assert out[1] is prior[1]
    assert out[2]["quote"] == "c-quote"


def test_merge_falls_back_positionally_when_echo_differs():
    # The model paraphrased the criterion, so text matching cannot align it.
    prior = [_a("A", failure=True), _a("B")]
    retried = [_a("A (paraphrased)", quote="recovered")]
    out = _merge_retry(prior, retried)
    assert out[0]["quote"] == "recovered"
    assert out[1] is prior[1]


def test_merge_keeps_prior_when_retry_returns_nothing():
    prior = [_a("A", failure=True), _a("B")]
    out = _merge_retry(prior, [])
    assert out == prior


def test_merge_never_overwrites_a_passing_criterion():
    # A retry that answers a criterion which did not fail must not displace it.
    prior = [_a("A"), _a("B", failure=True)]
    retried = [_a("A", verdict="not_met", quote="wrong"), _a("B", quote="right")]
    out = _merge_retry(prior, retried)
    assert out[0] is prior[0]
    assert out[1]["quote"] == "right"
