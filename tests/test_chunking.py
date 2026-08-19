"""Document chunking — the fix for the 512-token truncation blind spot."""

from __future__ import annotations

import numpy as np

from trialguard.ingestion.embed import CHUNK_MAX_CHARS, chunk_document


def test_every_chunk_fits_the_encoder_window():
    """The whole point: no passage may exceed what the encoder will read."""
    trial = {
        "title": "Phase 3 study of something with a fairly long title",
        "inclusion_criteria": ["inclusion " + "x" * 700 for _ in range(6)],
        "exclusion_criteria": ["exclusion " + "y" * 700 for _ in range(6)],
    }
    chunks = chunk_document(trial)
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_MAX_CHARS for c in chunks)


def test_exclusion_criteria_survive_chunking():
    """Unchunked, exclusions are concatenated last and truncated first."""
    trial = {
        "title": "T",
        "inclusion_criteria": ["a" * 1700],
        "exclusion_criteria": ["prior immunotherapy is not permitted"],
    }
    assert any("prior immunotherapy" in c for c in chunk_document(trial))


def test_title_repeats_so_a_passage_knows_its_disease():
    trial = {
        "title": "Metastatic NSCLC trial",
        "inclusion_criteria": ["a" * 1500],
        "exclusion_criteria": ["b" * 1500],
    }
    chunks = chunk_document(trial)
    assert len(chunks) >= 2
    assert all(c.startswith("Metastatic NSCLC trial | ") for c in chunks)


def test_one_oversized_criterion_is_split_not_dropped():
    trial = {"title": "T", "inclusion_criteria": ["z" * 6000], "exclusion_criteria": []}
    chunks = chunk_document(trial)
    assert all(len(c) <= CHUNK_MAX_CHARS for c in chunks)
    assert sum(c.count("z") for c in chunks) == 6000


def test_short_trial_stays_a_single_chunk():
    trial = {"title": "T", "inclusion_criteria": ["age 18 or older"], "exclusion_criteria": []}
    assert chunk_document(trial) == ["T | age 18 or older"]


def test_trial_scores_as_its_best_passage_not_its_total():
    """Max, not sum: a long trial must not outrank a precise one by having more
    chances to match."""
    from trialguard.eval.file_index import FileIndex

    idx = FileIndex("sigir")
    idx._nct_ids = ["SHORT", "LONG"]
    idx._chunk_owners = ["SHORT", "LONG", "LONG", "LONG"]
    idx._matrix = np.array(
        [
            [1.0, 0.0],   # SHORT: one excellent passage
            [0.6, 0.8],   # LONG: three mediocre passages
            [0.6, 0.8],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )
    hits = idx._dense(np.array([1.0, 0.0], dtype=np.float32), pool=2)
    assert [n for n, _ in hits] == ["SHORT", "LONG"]
    assert len(hits) == 2, "passages of one trial must collapse to a single hit"
