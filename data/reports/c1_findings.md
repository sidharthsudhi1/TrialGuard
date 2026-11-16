# C1: keywords stream, the first one arrives before the call ends

Run 2026-11-10. Three cold synthetic notes, none previously cached. Kill, written before the run: the final ranking for a given keyword list stays identical to the blocking path (locked by `tests/test_keyword_stream.py`), and time-to-first keyword is lower than the full call. Raw: `c1_stream_time.json`.

| note | keywords | first ms | total ms |
|---|---|---|---|
| 1 | 9 | 60796.4 | 66966.3 |
| 2 | 9 | 5051.2 | 9911.7 |
| 3 | 10 | 5602.1 | 14140.8 |

All three fire the first keyword before the call returns, so the kill is met. Note 1 is a stalled call: the first keyword is already a minute in, and the rest of the list adds six seconds. Notes 2 and 3 are the ordinary cold path. The first keyword is about half the call, still around 5 seconds. The workstream target of 1.5 seconds to first results is not met. Retrieval has not started at `first_ms`, and a raw-note list is a separate path (C2).

## Decision

DONE-ADOPTED for the stream (`TG_KEYWORD_STREAM` defaults on). A streamed list that the full parse does not keep is discarded and the parsed list is searched, which is what keeps the ranking identical. The cold path is still an LLM call.
