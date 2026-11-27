"""Phase 40: words common in trial eligibility text, for the name refusal.

A capitalised token that appears in 20+ TREC 2021 trials (Foley, Hodgkin,
Patient, Long) is medical or ordinary vocabulary, not evidence of a name on its
own. Rebuild only if the corpus changes:

    python scripts/build_common_words.py
"""

import collections
import json
import re
from pathlib import Path

CORPUS = Path("data/eval/trec_2021/trec_2021_corpus.jsonl")
OUT = Path("src/trialguard/agent/common_words.txt")
MIN_TRIALS = 20

df: collections.Counter[str] = collections.Counter()
with CORPUS.open() as handle:
    for line in handle:
        obj = json.loads(line)
        text = f"{obj.get('title', '')} {obj.get('text', '')}".lower()
        df.update(set(re.findall(r"[a-z][a-z'-]+", text)))
words = sorted(w for w, n in df.items() if n >= MIN_TRIALS)
OUT.write_text("\n".join(words) + "\n")
print(f"wrote {OUT}: {len(words)} words")
