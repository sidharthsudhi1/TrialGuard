# E4: span detectors do not clear the entailment bar

Run 2026-11-16 on a Modal L4. Gold is the existing 350 items
(`e7_combined_gold.jsonl`): 50 human, 300 model-labelled. The kill uses the
human set only. A passing threshold damages at most 2 sound citations and
catches at least 25% of the unsound ones. Thresholds are 0.05 to 0.95 by 0.05.
Raw: `e4_probe.json`.

MiniCheck is `lytang/MiniCheck-Flan-T5-Large`, scored the way the published
code scores it (token ids 3 and 209). Those ids reproduce the README pair:
supported 0.9806, unsupported 0.0071. LettuceDetect is
`KRLabsOrg/lettucedect-base-modernbert-en-v1`. Its score is 1 minus the
highest hallucination-token probability. The quote is one premise. The
sentence that contains the quote is the other. 29 of 350 quotes are not in
the note, so those sentence-arm items use the quote.

## Human set (n=50, 18 unsound)

| model | premise | AUC | best catch at ≤2 sound damaged |
|---|---|---|---|
| MiniCheck | quote | 0.7005 | none (least damage is 6, at recall 0.333) |
| MiniCheck | sentence | 0.7049 | none (least damage is 6, at recall 0.333) |
| LettuceDetect | quote | 0.5938 | none (0.05 already damages 30 and rejects 48) |
| LettuceDetect | sentence | 0.6285 | none (0.05 already damages 29 and rejects 47) |

MiniCheck ranks better than the earlier NLI checkpoints did on this human
set, and it still has no operating point. Catching a third of the unsound
citations costs 6 sound ones. LettuceDetect barely ranks and rejects almost
every citation at every threshold on the grid.

## Model-labelled set (n=300, not used for the bar)

AUC: MiniCheck 0.6269 quote / 0.6487 sentence, LettuceDetect 0.6055 / 0.5863.
Same shape as the human set, weaker. These labels do not set a threshold.

## Decision

DONE-REJECTED. No badge. `TG_WEAK_ENTAILMENT` stays off. This says nothing
about Bespoke-MiniCheck-7B, which was not run.
