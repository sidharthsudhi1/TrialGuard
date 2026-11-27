"""Phase 40: the synthetic planted-name set the NER refusal is measured on."""

import pytest

from trialguard.eval.planted_names import FIRST, LAST, build, plant


def test_plant_puts_the_whole_name_in_the_note():
    note = "A 45-year-old woman with stage IV NSCLC."
    for template in range(6):
        out, name = plant(note, "Amara", "Okafor", template)
        assert name in out
        assert "NSCLC" in out


def test_build_is_deterministic_and_large_enough():
    a, b = build(seed=40), build(seed=40)
    assert a == b
    assert len(a) >= 200
    assert all(x["name"] in x["note"] for x in a)
    assert len({x["template"] for x in a}) >= 6


def test_names_cover_many_origins_and_word_like_names():
    assert len(FIRST) >= 60 and len(LAST) >= 60
    # Names that are also English words are the hard case; leaving them out
    # would flatter recall.
    assert {"Hope", "Grace"} & set(FIRST)
    assert {"Young", "Long"} & set(LAST)


def test_base_notes_carry_no_name_of_their_own():
    # Two cohort notes already name the patient; planting into them would count
    # a hit on the wrong name.
    assert not any(x["base_id"] in ("sigir-201515", "trec-202148") for x in build(seed=40))


@pytest.mark.parametrize("holdout", [1, 2])
def test_holdout_sets_are_separate_and_large_enough(holdout):
    held = build(seed=40, holdout=holdout)
    assert len(held) >= 200
    assert all(x["name"] in x["note"] for x in held)
    others = {x["note"] for h in {0, 1, 2} - {holdout} for x in build(seed=40, holdout=h)}
    assert {x["note"] for x in held}.isdisjoint(others)
