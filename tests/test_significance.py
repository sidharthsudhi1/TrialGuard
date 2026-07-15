from trialguard.eval.significance import matched_ab


def test_matched_set_is_the_intersection():
    baseline = {"A": {"decisive": 10, "unsupported": 4}, "B": {"decisive": 5, "unsupported": 1}}
    verified = {"A": {"decisive": 10, "unsupported": 1}, "C": {"decisive": 5, "unsupported": 0}}
    # only trial A is in both arms
    out = matched_ab(baseline, verified)
    assert out["matched_trials"] == 1
    assert out["table"]["baseline"]["decisive"] == 10
    assert out["table"]["verified"]["unsupported"] == 1


def test_clear_reduction_is_significant():
    # 40 unsupported/200 -> 2 unsupported/200 is a large, significant drop
    baseline = {"t": {"decisive": 200, "unsupported": 40}}
    verified = {"t": {"decisive": 200, "unsupported": 2}}
    out = matched_ab(baseline, verified)
    assert out["fisher_p"] < 0.05
    assert out["significant_05"] is True
    assert out["relative_change"] < 0


def test_no_difference_is_not_significant():
    baseline = {"t": {"decisive": 100, "unsupported": 12}}
    verified = {"t": {"decisive": 100, "unsupported": 11}}
    out = matched_ab(baseline, verified)
    assert out["fisher_p"] > 0.05
    assert out["significant_05"] is False


def test_empty_matched_set_does_not_crash():
    a = {"a": {"decisive": 1, "unsupported": 0}}
    b = {"b": {"decisive": 1, "unsupported": 0}}
    out = matched_ab(a, b)
    assert out["matched_trials"] == 0
    assert out["fisher_p"] == 1.0
