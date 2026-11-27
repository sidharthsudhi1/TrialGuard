"""Phase 40: served traces carry structure only, never note text."""

import json
import re
from pathlib import Path

from trialguard import tracing
from trialguard.eval.served_monitor import compute

NOTE = (
    "A 52-year-old woman with HER2-positive metastatic breast cancer progressed on "
    "trastuzumab and pertuzumab. She has grade 2 peripheral neuropathy and an ECOG "
    "performance status of 1. Brain MRI shows two small treated lesions."
)


def _state(note: str = NOTE) -> dict:
    return {
        "patient_note": note,
        "nct_id": "NCT01234567",
        "criteria": [{"text": "HER2-positive breast cancer", "kind": "inclusion"}],
        "source_text": "Inclusion Criteria: HER2-positive breast cancer",
        "retries": 1,
        "assessments": [
            {
                "criterion": "HER2-positive breast cancer",
                "verdict": "met",
                "kind": "inclusion",
                "quote": "HER2-positive metastatic breast cancer progressed on trastuzumab",
                "rationale": "The note states HER2-positive metastatic breast cancer.",
                "grounded": True,
                "grounding_failure": False,
            },
            {
                "criterion": "No peripheral neuropathy grade 2 or higher",
                "verdict": "unverifiable",
                "kind": "exclusion",
                "quote": "grade 2 peripheral neuropathy and an ECOG",
                "rationale": "",
                "grounded": False,
                "grounding_failure": True,
                "weak_absence": True,
            },
            {"criterion": "ECOG 0-1", "verdict": "cannot_determine", "kind": "inclusion",
             "quote": "", "grounded": False},
        ],
        "trial_verdict": "cannot_determine",
        "trial_tier": "needs_review",
        "n_unknown": 1,
        "unknown_criteria": ["ECOG 0-1"],
        "disqualifying_criteria": [],
        "criteria_truncated": False,
    }


def _windows(text: str, n: int = 4) -> set[str]:
    words = re.findall(r"\S+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _leaks(exported: str, note: str = NOTE) -> set[str]:
    flat = " ".join(re.findall(r"\S+", exported.lower()))
    return {w for w in _windows(note) if w in flat}


def test_scrub_keeps_structure_and_drops_text():
    out = tracing.scrub(_state())
    assert out["nct_id"] == "NCT01234567"
    assert out["retries"] == 1
    assert out["trial_verdict"] == "cannot_determine"
    assert out["trial_tier"] == "needs_review"
    assert [a["verdict"] for a in out["assessments"]] == ["met", "unverifiable", "cannot_determine"]
    assert [a["kind"] for a in out["assessments"]] == ["inclusion", "exclusion", "inclusion"]
    assert out["assessments"][1]["weak_absence"] is True
    assert out["assessments"][1]["grounding_failure"] is True
    assert not _leaks(json.dumps(out))
    assert out["patient_note"] == tracing.REDACTED
    assert out["assessments"][0]["quote"] == tracing.REDACTED


def test_scrub_redacts_a_safe_key_holding_free_text():
    # A verdict field is an enum; anything with a space in it is not one.
    out = tracing.scrub({"verdict": "met because the note says HER2-positive"})
    assert out["verdict"] == tracing.REDACTED


def test_scrub_handles_message_lists_and_scalars():
    msgs = [{"role": "system", "content": "You are an analyst."},
            {"role": "user", "content": NOTE}]
    out = tracing.scrub(msgs)
    assert [m["role"] for m in out] == ["system", "user"]
    assert all(m["content"] == tracing.REDACTED for m in out)
    assert tracing.scrub(NOTE) == tracing.REDACTED
    assert tracing.scrub(3.5) == 3.5
    assert tracing.scrub(None) is None


def test_monitor_rates_survive_the_scrub():
    states = [_state(), {**_state(), "retries": 0, "trial_verdict": "eligible"}]
    assert compute([tracing.scrub(s) for s in states]) == compute(states)


def _exported_attributes(monkeypatch, structure_only: bool) -> str:
    from langchain_core.runnables import RunnableLambda
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    # Langfuse keeps one client per public key; each run needs its own.
    key = f"pk-lf-scrub-{'masked' if structure_only else 'raw'}"
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=key,
        secret_key="sk-lf-scrub-test",
        base_url="http://localhost:1",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
        mask_otel_spans=tracing.mask_served_spans if structure_only else None,
    )
    handler = CallbackHandler(public_key=key)
    graph = RunnableLambda(lambda state: {**state, **_state(state["patient_note"])})
    graph.invoke(
        {"patient_note": NOTE},
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_tags": ["served", "assess"], "nct_id": "NCT01234567",
                         "note_echo": NOTE},
        },
    )
    with client.start_as_current_observation(name="keywords", as_type="generation") as gen:
        gen.update(input=[{"role": "user", "content": NOTE}],
                   output="HER2-positive metastatic breast cancer; trastuzumab",
                   model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                   status_message=f"failed on {NOTE}")
    client.flush()
    spans = exporter.get_finished_spans()
    assert spans, "nothing was exported"
    return json.dumps([dict(s.attributes) for s in spans], default=str)


def test_unmasked_export_carries_the_note(monkeypatch):
    # Control: proves the leak check can see a leak at all.
    assert _leaks(_exported_attributes(monkeypatch, structure_only=False))


def test_served_export_carries_no_note_substring_over_three_tokens(monkeypatch):
    exported = _exported_attributes(monkeypatch, structure_only=True)
    assert not _leaks(exported)
    assert "unverifiable" in exported
    assert "NCT01234567" in exported
    assert "served" in exported


def test_structure_only_is_set_for_the_served_app():
    import tomllib

    env = tomllib.loads(Path("fly.toml").read_text())["env"]
    assert env.get("TRACE_STRUCTURE_ONLY") == "1"


def test_structure_only_never_hands_out_an_unmasked_client(monkeypatch):
    """Langfuse keeps one client per key; one built unmasked first must not be reused."""
    from langfuse import Langfuse

    from trialguard.config import settings

    Langfuse(public_key="pk-lf-early", secret_key="sk", base_url="http://localhost:1")
    monkeypatch.setattr(settings, "trace_structure_only", True)
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-early")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk")
    monkeypatch.setattr(tracing, "_MASKED_CLIENT", None)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-early")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert tracing.get_client() is None
    assert tracing.get_langchain_handler(tags=["served"]) is None


def test_structure_only_client_carries_the_mask(monkeypatch):
    from trialguard.config import settings

    monkeypatch.setattr(settings, "trace_structure_only", True)
    monkeypatch.setattr(settings, "tracing_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-fresh")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk")
    monkeypatch.setattr(settings, "langfuse_base_url", "http://localhost:1")
    monkeypatch.setattr(tracing, "_MASKED_CLIENT", None)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-fresh")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:1")
    client = tracing.get_client()
    assert client._resources.mask_otel_spans is tracing.mask_served_spans
