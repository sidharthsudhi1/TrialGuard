# E7 follow-up — the NLI question at n=350

Run 2026-09-23. The experiment E7 was built for: does an operating point exist at
n≈350 that did not exist at n=50? Gold:
[`e7_combined_gold.jsonl`](e7_combined_gold.jsonl) — 50 human-adjudicated
(WS-5b) plus 300 model-adjudicated, disjoint by construction and asserted so.
Cost $0.

**Answer: no, and the ranking signal is much weaker than n=50 suggested.**

## Results

| checkpoint | set | n | AUC | 95% CI | best precision | base rate |
|---|---|---|---|---|---|---|
| PubMedBERT-MedNLI | human | 50 | 0.7648 | [0.622, 0.890] | | |
| PubMedBERT-MedNLI | model | 300 | 0.5456 | [0.471, 0.617] | | |
| PubMedBERT-MedNLI | **all** | **350** | **0.5758** | [0.512, 0.642] | 0.3692 | 0.2971 |
| DeBERTa-v3-lg-fever | human | 50 | 0.6901 | [0.536, 0.828] | | |
| DeBERTa-v3-lg-fever | model | 300 | 0.5886 | [0.522, 0.656] | | |
| DeBERTa-v3-lg-fever | **all** | **350** | **0.6071** | [0.547, 0.664] | 0.4348 | 0.2971 |

CIs are bootstrap, 2000 resamples, seeded.

**No operating point.** Best precision across every threshold is 0.369 and 0.435
against a base rate of 0.297 — barely above guessing, and each is bought by
destroying sound citations in bulk (MedNLI's 0.3692 point rejects 130 items and
destroys 82 sound ones). The bar was a threshold that almost never rejects a
sound citation. Nothing comes close.

**The n=50 signal shrank.** MedNLI fell from AUC 0.7648 to 0.5758, and its
350-item CI [0.512, 0.642] sits barely off 0.5. Seven times the data moved the
estimate toward chance rather than sharpening it.

## The confound, stated first

**86% of these labels are mine, not a human's.** They were measured at kappa
0.7626 against held-out human items and are systematically lenient
([`e7_model_adjudication_findings.md`](e7_model_adjudication_findings.md)), so
this is not a human-validated result and must not be quoted as one.

The split above is what that confound looks like, and it is not clean:

- **MedNLI**: human 0.7648 [0.622, 0.890] vs model 0.5456 [0.471, 0.617]. The
  intervals barely fail to overlap.
- **DeBERTa-v3**: human 0.6901 [0.536, 0.828] vs model 0.5886 [0.522, 0.656].
  They overlap substantially.

One marginal separation and one overlap is **weak evidence** for a systematic
label-source effect, not proof of one. Two readings survive, and this experiment
cannot separate them:

1. The n=50 AUCs were small-sample overestimates. Both human CIs are wide enough
   ([0.62, 0.89] and [0.54, 0.83]) to contain the model-set point estimates, so
   regression toward chance is the ordinary explanation.
2. The model labels encode something different enough from human judgment to
   flatten the NLI signal, which would be an instance of exactly the
   correlated-judgment problem AD-3 raises.

Reading 1 is the more parsimonious and the CIs support it. Reading 2 cannot be
dismissed, and the way to separate them is human labels on the same 300 items.

## What this does and does not change

**Strengthens AD-23's rejection.** It held that off-the-shelf NLI ranks
respectably and cannot be thresholded. At 350 items it does not even rank
respectably: two architectures, one biomedical and one general, sit at AUC 0.58
and 0.61 with best precision within 0.14 of the base rate. The claim "this route
is not close" is now supported by seven times the evidence it had.

**Does not close the route by itself**, because of the confound above. What it
does is reorder the decision: a human labelling pass was previously the
prerequisite for finding a threshold. It is now the test of whether a threshold
could exist at all, and the prior against has grown.

**`deberta-large-mnli` was not re-run.** Its repository publishes no safetensors
(the HF cache carries a `.no_exist` marker for `model.safetensors`) and the
`.bin` is ~1.6 GB against 1.2 GB free on this machine. Its n=50 AUC of 0.7899
stands as the prior; no n=350 figure is claimed for it.

## Recommendation

Do not fund several hundred more adjudications to find a threshold. Fund a
**smaller, targeted human pass** instead: re-label a stratified 100 of the 300
and re-run these two checkpoints on human labels only. That answers both open
questions at once — whether the AUC collapse is sample size or label source, and
whether any operating point survives on clean labels — for roughly a third of
the effort.

If AUC on 150 clean human items still sits near 0.6, the NLI route is closed on
evidence and AD-23's P2 becomes a rejection rather than a deferral.
