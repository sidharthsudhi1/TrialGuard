# Deploying the demo to Hugging Face Spaces (Phase 6, WS-3)

The demo runs at $0 on a free HF Spaces CPU: retrieval is the self-contained
`FileIndex` (numpy + cached MedCPT embeddings), and inference is the Groq free tier.
No database. Deploy is user-gated (your HF account + Groq key).

## Space README front-matter

A Space needs a `README.md` at its root with this YAML front-matter (kept separate
from the portfolio README so the GitHub case study is not polluted). Copy this into
the Space's `README.md`:

```yaml
---
title: TrialGuard
emoji: 🧬
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
short_description: Self-verifying clinical-trial eligibility with cited verdicts
---
```

## Files the Space needs

The app entry (`app.py`), the package (`src/`, `pyproject.toml`, `requirements.txt`),
and the demo data — which is gitignored in the main repo and must be copied into the
Space:

- `data/indexes/sigir_medcpt_excl_embeddings.npy` and `..._ids.json` (cached MedCPT
  index, ~9 MB)
- the SIGIR corpus + patient files under `data/eval/sigir/` that `FileIndex` and the
  presets load

Mark the `.npy` for LFS in the Space (`.gitattributes`: `*.npy filter=lfs diff=lfs
merge=lfs -text`).

## Secrets (Space settings → Secrets)

- `GROQ_API_KEY` — required (analyst + keyword extraction).
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` — optional (tracing/scores; the app is
  a no-op without them).

## Deploy steps

1. Create a new Gradio Space on huggingface.co.
2. Add the Space as a git remote and push `app.py`, `src/`, `pyproject.toml`,
   `requirements.txt`, the Space `README.md`, `.gitattributes`, and the demo data.
3. Set the secrets above.
4. The Space builds from `requirements.txt` and launches `app.py`. First build
   downloads the MedCPT weights (cached thereafter).

## Quota note

A public demo shares your Groq daily cap. Presets whose analyst results are already
in `data/cache/analyst/` render for free; fresh free-text queries spend tokens and
degrade gracefully via `TokenBudget` when the cap is hit. Pre-cache the presets (run
each once locally) before deploying so the common path costs nothing.
