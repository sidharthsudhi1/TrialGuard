# 3-minute walkthrough storyboard (Phase 6, WS-5)

For the recorded demo. Timings are targets; the faithfulness beat (Scene 3) is the
one that must land — it is the whole point. Record at 1080p, show the browser and a
terminal/trace tab.

## Scene 1 — The problem & thesis (0:00–0:25)
- **On screen:** the README title / the running Gradio app.
- **Say:** "Matching patients to clinical trials is a real bottleneck, and the
  dangerous failure mode is an AI that confidently says *eligible* based on a
  criterion it misread. TrialGuard's thesis: faithfulness is the product. Every
  verdict is backed by a verbatim citation from the trial, or it's flagged
  unverifiable — never forced."

## Scene 2 — A clean match (0:25–1:05)
- **Do:** pick the first synthetic patient preset, click **Assess eligibility**.
- **On screen:** ranked trials, a 🟢 Eligible roll-up, criteria with ✅ met and green
  grounded citations.
- **Say:** "Pick a synthetic patient — no real data ever. It retrieves candidate
  oncology trials, then assesses each criterion. Every *met* verdict shows the exact
  quote it's standing on, checked character-for-character against the trial text."

## Scene 3 — The faithfulness beat (1:05–1:55) — the money shot
- **Do:** pick a preset (or paste a note) that produces an ⚠️ *unverifiable* criterion.
- **On screen:** a criterion downgraded to unverifiable, shown, not hidden.
- **Say:** "Here's the difference. The analyst tried to claim this criterion, but its
  quote wasn't verbatim in the source — so a deterministic Python check caught it and
  downgraded it to *unverifiable* instead of passing it through. A hallucinated
  citation cannot survive. On our benchmark this cut hallucinated citations ~64% on
  SIGIR, and the verifier catches 100% of corrupted quotes by construction."

## Scene 4 — Under the hood & cost (1:55–2:35)
- **Do:** open the "How this works" panel; optionally show a Langfuse trace and the
  green CI badge.
- **Say:** "Retrieval is MedCPT dense plus BM25 fused with RRF. The agent is a
  LangGraph loop: analyst, deterministic grounding, bounded retry. Every run is traced
  in Langfuse, and a CI regression gate fails the build if faithfulness ever drops.
  The whole stack — inference, embeddings, vector store, hosting — runs at $0."

## Scene 5 — Close (2:35–3:00)
- **Say:** "So: a cited, ranked trial shortlist for a synthetic patient, where every
  verdict is either grounded in a real quote or honestly flagged unverifiable. That's
  faithfulness as a measurable product, not a promise. Code and method are in the
  repo."

## Shot list / checklist
- [ ] Preset that yields a clear 🟢 Eligible (Scene 2).
- [ ] Preset/note that yields an ⚠️ unverifiable criterion (Scene 3) — pre-test which.
- [ ] Langfuse trace tab open (optional, Scene 4).
- [ ] Green CI check on the repo visible (optional, Scene 4).
- [ ] Presets pre-cached so nothing waits on a live Groq call mid-recording.
