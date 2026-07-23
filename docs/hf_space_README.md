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

# TrialGuard

Self-verifying, multi-agent clinical-trial eligibility. Every eligibility verdict is
backed by a verbatim citation from the source trial, or explicitly flagged
*unverifiable* — never forced. All patient notes in this demo are synthetic.

Pick a synthetic patient, or paste your own synthetic note. TrialGuard retrieves
candidate oncology trials (MedCPT dense + BM25, RRF), assesses each criterion with a
verbatim quote, and deterministically checks every quote against the source before
letting a verdict stand.

Code and method: https://github.com/sidharthsudhi1/TrialGuard
