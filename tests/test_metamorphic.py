from trialguard.eval.metamorphic import directional, invariance

NOTE = (
    "A 63-year-old man presents with chest pain. "
    "He takes a daily multivitamin. "
    "The symptoms have lasted three months."
)


def test_directional_deletes_the_quoted_sentence():
    edited = directional(NOTE, "63-year-old man presents with chest pain")
    assert edited is not None
    assert "chest pain" not in edited
    assert "multivitamin" in edited


def test_invariance_keeps_the_quote_and_moves_another_sentence():
    edited = invariance(NOTE, "63-year-old man presents with chest pain")
    assert edited is not None
    assert "chest pain" in edited
    assert edited != NOTE
    assert "multivitamin" in edited
