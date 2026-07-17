"""
CodeLangID -- measuring the cost of splitting by snippet instead of by task.

Runs BOTH models (TF-IDF+NB baseline and the char-CNN) so the report can cite one script.

The main pipeline splits BY TASK: every snippet derived from a Rosetta task lands in
exactly one split. This script quantifies what happens if you split BY SNIPPET instead,
which is the obvious-but-wrong thing to do: two solutions to the same task then straddle
train and test, and the model is scored partly on what it memorised.

Everything else (data, cleaning, model, seed) is held fixed. Only the split changes.

Usage:  python3 src/leakage_experiment.py
Output: results/leakage_experiment.json
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from cnn import DEV, CharCNN, encode, evaluate

SEED = 360
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"


def load_all():
    """All balanced snippets, pooled back across the by-task splits."""
    rows = []
    for split in ("train", "val", "test"):
        for line in (DATA / f"raw_{split}.jsonl").open():
            r = json.loads(line)
            r["split_by_task"] = split
            rows.append(r)
    return rows


def fit_eval(train, test):
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2,
                        sublinear_tf=True, lowercase=False),
        MultinomialNB(),
    )
    clf.fit([r["text"] for r in train], [r["label"] for r in train])
    p = clf.predict([r["text"] for r in test])
    return float(accuracy_score([r["label"] for r in test], p))


def fit_eval_cnn(train, val, test, epochs=30, patience=6):
    """Same architecture/hparams as src/cnn.py; only the split differs."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    def ds(rs):
        return TensorDataset(
            torch.tensor([encode(r["text"]) for r in rs], dtype=torch.long),
            torch.tensor([r["label"] for r in rs], dtype=torch.long))

    tr = DataLoader(ds(train), batch_size=64, shuffle=True)
    va, te = DataLoader(ds(val), batch_size=256), DataLoader(ds(test), batch_size=256)
    model = CharCNN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, bad, state = 0.0, 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        _, vacc, _, _ = evaluate(model, va)
        if vacc > best:
            best, bad = vacc, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    _, acc, _, _ = evaluate(model, te)
    return float(acc)


def main():
    rows = load_all()
    print(f"pooled {len(rows)} snippets over {len({r['task'] for r in rows})} tasks")

    # ---- A: the pipeline's real split (by task)
    tr_task = [r for r in rows if r["split_by_task"] == "train"]
    te_task = [r for r in rows if r["split_by_task"] == "test"]
    acc_task = fit_eval(tr_task, te_task)
    print(f"  A. split BY TASK   (correct): train={len(tr_task)} test={len(te_task)} "
          f"acc={acc_task*100:.2f}")

    # ---- B: the naive split (by snippet), same sizes, same seed
    shuffled = rows[:]
    random.Random(SEED).shuffle(shuffled)
    n_tr, n_te = len(tr_task), len(te_task)
    tr_snip = shuffled[:n_tr]
    te_snip = shuffled[n_tr:n_tr + n_te]
    acc_snip = fit_eval(tr_snip, te_snip)
    print(f"  B. split BY SNIPPET (naive): train={len(tr_snip)} test={len(te_snip)} "
          f"acc={acc_snip*100:.2f}")

    # ---- how much of B's test set shares a task with B's train set?
    tr_tasks = {r["task"] for r in tr_snip}
    contaminated = sum(1 for r in te_snip if r["task"] in tr_tasks)
    frac = contaminated / len(te_snip)

    inflation = (acc_snip - acc_task) * 100
    print(f"\n  contaminated test snippets (share a task with train): "
          f"{contaminated}/{len(te_snip)} = {frac*100:.1f}%")
    print(f"  INFLATION from splitting by snippet: +{inflation:.2f} points "
          f"({acc_task*100:.2f} -> {acc_snip*100:.2f})")

    # ---- same comparison for the char-CNN
    print("\n  char-CNN (same architecture and hyper-parameters, only the split differs):")
    va_task = [r for r in rows if r["split_by_task"] == "val"]
    cnn_task = fit_eval_cnn(tr_task, va_task, te_task)
    print(f"  A. split BY TASK   (correct): acc={cnn_task*100:.2f}")
    n_va = len(va_task)
    cnn_snip = fit_eval_cnn(shuffled[:n_tr],
                            shuffled[n_tr:n_tr + n_va],
                            shuffled[n_tr + n_va:n_tr + n_va + n_te])
    print(f"  B. split BY SNIPPET (naive): acc={cnn_snip*100:.2f}")
    cnn_infl = (cnn_snip - cnn_task) * 100
    print(f"  CNN inflation from splitting by snippet: {cnn_infl:+.2f} points")

    out = {
        "note": "Same data, cleaning, models and seed; only the split differs.",
        "cnn_by_task": cnn_task,
        "cnn_by_snippet": cnn_snip,
        "cnn_inflation_points": cnn_infl,
        "by_task": {"train_n": len(tr_task), "test_n": len(te_task), "accuracy": acc_task},
        "by_snippet": {"train_n": len(tr_snip), "test_n": len(te_snip), "accuracy": acc_snip},
        "contaminated_test_snippets": contaminated,
        "contaminated_fraction": frac,
        "inflation_points": inflation,
        "seed": SEED,
    }
    (RES / "leakage_experiment.json").write_text(json.dumps(out, indent=2))
    print(f"wrote -> {RES/'leakage_experiment.json'}")


if __name__ == "__main__":
    main()
