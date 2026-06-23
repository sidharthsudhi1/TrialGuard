"""Load TrialGPT eval corpora (SIGIR + TREC 2021/2022) into trials table.

Usage:
    python -m trialguard.scripts.load_eval_corpus
"""

from trialguard.eval.corpus_loader import load_all_eval_corpora

if __name__ == "__main__":
    load_all_eval_corpora()
