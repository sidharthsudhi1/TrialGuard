"""Turn the TREC 2023 download into queries, qrels, and a trial jsonl.

    python -m trialguard.scripts.prepare_trec_2023
"""

from trialguard.eval.trec_2023 import prepare

if __name__ == "__main__":
    stats = prepare()
    print(stats)
