"""
CodeLangID -- baseline model: TF-IDF over character n-grams + Multinomial Naive Bayes.

Deliberately classical: no deep learning, no tuning beyond scikit-learn defaults
and a fixed n-gram range. This is the bar the char-CNN must clear.

Usage:  python3 src/baseline.py
Output: results/baseline.json
"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

LABELS = json.loads((DATA / "stats.json").read_text())["labels"]


def load(name):
    xs, ys = [], []
    for line in (DATA / f"{name}.jsonl").open():
        r = json.loads(line)
        xs.append(r["text"])
        ys.append(r["label"])
    return xs, np.array(ys)


def run(variant):
    Xtr, ytr = load(f"{variant}_train")
    Xva, yva = load(f"{variant}_val")
    Xte, yte = load(f"{variant}_test")
    Xho, yho = load(f"{variant}_heldout")

    clf = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2,
                        sublinear_tf=True, lowercase=False),
        MultinomialNB(),
    )
    clf.fit(Xtr, ytr)

    out = {}
    for split, (X, y) in {"val": (Xva, yva), "test": (Xte, yte),
                          "heldout": (Xho, yho)}.items():
        p = clf.predict(X)
        out[split] = {
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro")),
        }
        if split in ("test", "heldout"):
            out[split]["confusion"] = confusion_matrix(
                y, p, labels=range(len(LABELS))).tolist()

    n_feat = len(clf.named_steps["tfidfvectorizer"].vocabulary_)
    out["n_features"] = n_feat
    print(f"[{variant}] tfidf features={n_feat}  "
          f"val={out['val']['accuracy']:.4f}  test={out['test']['accuracy']:.4f}  "
          f"heldout={out['heldout']['accuracy']:.4f}")
    return out


def main():
    res = {v: run(v) for v in ("raw", "nocomment")}
    res["labels"] = LABELS
    (RES / "baseline.json").write_text(json.dumps(res, indent=2))
    print(f"wrote -> {RES/'baseline.json'}")


if __name__ == "__main__":
    main()
