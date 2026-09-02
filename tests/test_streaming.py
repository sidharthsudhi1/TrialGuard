import json

from trialguard.agent.analyst import _replay, _stream_assessments


class _Chunk:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Yields a JSON assessments array split at arbitrary boundaries."""

    def __init__(self, pieces):
        self.pieces = pieces

    def stream(self, messages, config=None):
        for p in self.pieces:
            yield _Chunk(p)


def _payload(n):
    return json.dumps({
        "assessments": [
            {"criterion": f"c{i}", "verdict": "met", "quote": f"q{i}", "rationale": "r"}
            for i in range(n)
        ]
    })


def _split(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_stream_emits_each_criterion_once_as_it_closes():
    seen = []
    raw = _stream_assessments(
        _FakeLLM(_split(_payload(3), 7)), [], None, seen.append
    )
    assert [a["criterion"] for a in seen] == ["c0", "c1", "c2"]
    assert json.loads(raw)["assessments"][0]["criterion"] == "c0"


def test_stream_emits_before_the_response_completes():
    # The point of L6: the first criterion must surface without the last one.
    payload = _payload(2)
    cut = payload.index("c1")
    seen = []
    _stream_assessments(_FakeLLM([payload[:cut]]), [], None, seen.append)
    assert [a["criterion"] for a in seen] == ["c0"]


def test_stream_survives_a_split_inside_a_string():
    payload = _payload(2)
    seen = []
    _stream_assessments(_FakeLLM(_split(payload, 1)), [], None, seen.append)
    assert [a["criterion"] for a in seen] == ["c0", "c1"]


def test_stream_ignores_empty_chunks():
    seen = []
    pieces = _split(_payload(1), 5)
    _stream_assessments(_FakeLLM([""] + pieces + [""]), [], None, seen.append)
    assert len(seen) == 1


def test_replay_feeds_cached_assessments_through_the_callback():
    cached = [{"criterion": "a"}, {"criterion": "b"}]
    seen = []
    out = _replay(cached, seen.append)
    assert out == cached
    assert [a["criterion"] for a in seen] == ["a", "b"]


def test_replay_without_a_callback_is_a_passthrough():
    cached = [{"criterion": "a"}]
    assert _replay(cached, None) == cached
