"""WS-5: the content-hash diff and confirmed expiry, tested as pure functions."""

from __future__ import annotations

from trialguard.ingestion.provenance import content_hash, doc_hash
from trialguard.ingestion.publish import Existing, confirm, corpus_tag, plan

TAG = "medcpt_excl"
PV = "2026-09-23.1"


def _t(nct: str, **kw) -> dict:
    t = {
        "nct_id": nct,
        "title": f"Trial {nct}",
        "status": "RECRUITING",
        "inclusion_criteria": ["Adults"],
        "exclusion_criteria": ["Pregnancy"],
        "last_updated": "2026-09-01",
    }
    t.update(kw)
    return t


def _stored(t: dict, *, tag=TAG, pv=PV, expired=False, missing_runs=0) -> Existing:
    return Existing(doc_hash(t), content_hash(t), tag, pv, expired, missing_runs)


def test_every_id_lands_in_exactly_one_bucket():
    fresh = {n: _t(n) for n in ("NCT1", "NCT2", "NCT3", "NCT4", "NCT5", "NCT6")}
    existing = {
        "NCT2": _stored(_t("NCT2", inclusion_criteria=["Children"])),   # text moved
        "NCT3": _stored(_t("NCT3"), expired=True),                        # came back
        "NCT4": _stored(_t("NCT4", status="NOT_YET_RECRUITING")),        # metadata only
        "NCT5": _stored(_t("NCT5")),                                      # unchanged
        "NCT6": _stored(_t("NCT6"), pv="unknown"),                        # parser behind
        "NCT7": _stored(_t("NCT7")),                                      # gone
        "NCT8": _stored(_t("NCT8"), expired=True),                        # already expired
    }

    p = plan(fresh, existing, TAG, PV)

    assert p.new == ["NCT1"]
    assert p.changed == ["NCT2"]
    assert p.resurrected == ["NCT3"]
    assert p.meta_only == ["NCT4", "NCT6"]
    assert p.seen == ["NCT5"]
    assert p.missing == ["NCT7"]


def test_a_same_day_revision_is_caught_by_the_hash_not_the_date():
    """F5: lastUpdatePostDate is a day. The text moved; the date did not."""
    old = _t("NCT1")
    new = _t("NCT1", exclusion_criteria=["Pregnancy", "Active hepatitis B"])
    p = plan({"NCT1": new}, {"NCT1": _stored(old)}, TAG, PV)
    assert p.changed == ["NCT1"]


def test_a_status_change_never_re_embeds():
    p = plan({"NCT1": _t("NCT1", status="ENROLLING_BY_INVITATION")},
             {"NCT1": _stored(_t("NCT1"))}, TAG, PV)
    assert p.to_embed == []
    assert p.meta_only == ["NCT1"]


def test_a_row_embedded_under_another_config_is_re_embedded():
    p = plan({"NCT1": _t("NCT1")}, {"NCT1": _stored(_t("NCT1"), tag="unknown")}, TAG, PV)
    assert p.changed == ["NCT1"]


def test_an_unchanged_corpus_embeds_nothing():
    fresh = {n: _t(n) for n in ("NCT1", "NCT2")}
    p = plan(fresh, {n: _stored(t) for n, t in fresh.items()}, TAG, PV)
    assert (p.to_embed, p.meta_only, p.missing) == ([], [], [])
    assert p.seen == ["NCT1", "NCT2"]


# --- confirmation -------------------------------------------------------------


def _existing(missing_runs=0):
    return {n: _stored(_t(n), missing_runs=missing_runs) for n in ("NCT1", "NCT2", "NCT3")}


def test_only_a_confirmed_departure_expires():
    """F3: absence from a crawl is not evidence; CT.gov's own status is."""
    lookup = lambda ids: {"NCT1": "COMPLETED", "NCT2": None, "NCT3": "RECRUITING"}  # noqa: E731

    c = confirm(["NCT1", "NCT2", "NCT3"], _existing(), ["RECRUITING"], lookup)

    assert c.expire == {"NCT1": "status:COMPLETED", "NCT2": "not_found"}
    assert c.keep == ["NCT3"]


def test_a_trial_out_of_the_condition_query_expires_after_repeated_misses():
    lookup = lambda ids: dict.fromkeys(ids, "RECRUITING")  # noqa: E731

    c = confirm(["NCT1"], _existing(missing_runs=2), ["RECRUITING"], lookup)

    assert c.expire == {"NCT1": "scope"}


def test_a_failed_lookup_expires_nothing():
    def boom(ids):
        raise RuntimeError("CT.gov down")

    c = confirm(["NCT1", "NCT2"], _existing(), ["RECRUITING"], boom)

    assert c.expire == {}
    assert c.keep == ["NCT1", "NCT2"]
    assert c.lookup_failed


def test_corpus_tag_ignores_unknown_and_expired_rows():
    existing = {
        "NCT1": _stored(_t("NCT1"), tag="unknown"),
        "NCT2": _stored(_t("NCT2"), tag="bge_excl", expired=True),
        "NCT3": _stored(_t("NCT3"), tag=TAG),
    }
    assert corpus_tag(existing) == TAG
    assert corpus_tag({}) is None
