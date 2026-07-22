from trialguard.agent import analyst as A
from trialguard.eval.regression_gate import registry_check


def test_every_live_prompt_is_registered():
    assert set(A._PROMPTS) == set(A.PROMPT_REGISTRY)


def test_frozen_hashes_match_current_text():
    for v, spec in A.PROMPT_REGISTRY.items():
        if spec["frozen"]:
            assert A.prompt_hash(v) == spec["sha16"], v


def test_registry_clean_by_default():
    assert A.registry_violations() == []
    assert registry_check()["prompt_registry_intact"] == 1.0


def test_frozen_prompt_drift_is_detected(monkeypatch):
    monkeypatch.setitem(A._PROMPTS, "v1", A._PROMPTS["v1"] + "\nsneaky edit")
    violations = A.registry_violations()
    assert any("v1" in v and "changed" in v for v in violations)
    assert registry_check()["prompt_registry_intact"] == 0.0


def test_unregistered_prompt_is_detected(monkeypatch):
    monkeypatch.setitem(A._PROMPTS, "v9", "an unregistered prompt")
    assert any("v9" in v for v in A.registry_violations())


def test_v3_is_not_frozen():
    # v3 backs no committed number yet, so it may still change without gate failure
    assert A.PROMPT_REGISTRY["v3"]["frozen"] is False
