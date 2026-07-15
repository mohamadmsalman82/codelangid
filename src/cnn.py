"""
CodeLangID -- primary model: character-level CNN.

Architecture (per snippet, 256 chars):
    char IDs -> Embedding(V, 32)
             -> three parallel Conv1d branches, kernel widths 3/5/7, 128 filters
             -> ReLU -> global max-pool over time -> concat (384)
             -> Dropout(0.5) -> Linear(384, 10) -> softmax

Trained with cross-entropy + Adam. Early stopping on validation accuracy.

Usage:  python3 src/cnn.py --variant raw
Output: results/cnn_<variant>.json, results/cnn_<variant>.pt
"""

import argparse
import json
import random
import string
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 360
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

LABELS = json.loads((DATA / "stats.json").read_text())["labels"]
WINDOW = 256

# Fixed vocabulary: printable ASCII, plus <pad>=0 and <unk>=1.
VOCAB = ["<pad>", "<unk>"] + list(string.printable)
STOI = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)

DEV = ("mps" if torch.backends.mps.is_available()
       else "cuda" if torch.cuda.is_available() else "cpu")


def encode(text):
    ids = [STOI.get(c, 1) for c in text[:WINDOW]]
    return ids + [0] * (WINDOW - len(ids))


def load(name):
    xs, ys = [], []
    for line in (DATA / f"{name}.jsonl").open():
        r = json.loads(line)
        xs.append(encode(r["text"]))
        ys.append(r["label"])
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


class CharCNN(nn.Module):
    def __init__(self, vocab=V, emb=32, filters=128, widths=(3, 5, 7),
                 classes=len(LABELS), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb, filters, w, padding=w // 2) for w in widths])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(widths), classes)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)              # B, emb, T
        h = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(h, dim=1)))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss = correct = n = 0
    preds, gold = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEV), yb.to(DEV)
        out = model(xb)
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        p = out.argmax(1)
        correct += (p == yb).sum().item()
        n += yb.size(0)
        preds.append(p.cpu())
        gold.append(yb.cpu())
    return loss / n, correct / n, torch.cat(preds), torch.cat(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="raw", choices=["raw", "nocomment"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    a = ap.parse_args()

    Xtr, ytr = load(f"{a.variant}_train")
    Xva, yva = load(f"{a.variant}_val")
    Xte, yte = load(f"{a.variant}_test")
    Xho, yho = load(f"{a.variant}_heldout")

    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=a.bs, shuffle=True)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=256)
    te = DataLoader(TensorDataset(Xte, yte), batch_size=256)
    ho = DataLoader(TensorDataset(Xho, yho), batch_size=256)

    model = CharCNN().to(DEV)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{a.variant}] device={DEV}  vocab={V}  params={nparam:,}  "
          f"train={len(ytr)} val={len(yva)} test={len(yte)} heldout={len(yho)}")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best, best_ep, bad = 0.0, -1, 0
    ckpt = RES / f"cnn_{a.variant}.pt"

    for ep in range(a.epochs):
        model.train()
        tot = corr = n = 0
        for xb, yb in tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * yb.size(0)
            corr += (out.argmax(1) == yb).sum().item()
            n += yb.size(0)
        vl, vacc, _, _ = evaluate(model, va)
        hist["train_loss"].append(tot / n)
        hist["train_acc"].append(corr / n)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(vacc)
        flag = ""
        if vacc > best:
            best, best_ep, bad = vacc, ep, 0
            torch.save(model.state_dict(), ckpt)
            flag = " *"
        else:
            bad += 1
        print(f"  ep {ep:2d}  train_loss {tot/n:.4f} acc {corr/n:.4f} | "
              f"val_loss {vl:.4f} acc {vacc:.4f}{flag}")
        if bad >= a.patience:
            print(f"  early stop (no val improvement for {a.patience} epochs)")
            break

    model.load_state_dict(torch.load(ckpt))
    out = {"variant": a.variant, "params": nparam, "vocab": V,
           "best_epoch": best_ep, "history": hist, "labels": LABELS,
           "device": DEV, "hparams": vars(a)}

    for split, loader, ys in (("val", va, yva), ("test", te, yte), ("heldout", ho, yho)):
        l, acc, preds, gold = evaluate(model, loader)
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
        for g, p in zip(gold.tolist(), preds.tolist()):
            cm[g][p] += 1
        f1s = []
        for c in range(len(LABELS)):
            tp, fp, fn = cm[c][c], cm[:, c].sum() - cm[c][c], cm[c].sum() - cm[c][c]
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        out[split] = {"loss": l, "accuracy": acc, "macro_f1": float(np.mean(f1s)),
                      "per_class_f1": f1s, "confusion": cm.tolist()}
        print(f"  {split:8s} loss {l:.4f}  acc {acc:.4f}  macroF1 {np.mean(f1s):.4f}")

    # qualitative: misclassified held-out snippets, kept for the report
    _, _, preds, gold = evaluate(model, ho)
    texts = [json.loads(l) for l in (DATA / f"{a.variant}_heldout.jsonl").open()]
    miss = [{"true": LABELS[g], "pred": LABELS[p], "repo": texts[i]["repo"],
             "path": texts[i]["path"], "text": texts[i]["text"]}
            for i, (g, p) in enumerate(zip(gold.tolist(), preds.tolist())) if g != p]
    out["heldout_errors"] = miss
    print(f"  {len(miss)} held-out errors saved for qualitative analysis")

    (RES / f"cnn_{a.variant}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote -> {RES/f'cnn_{a.variant}.json'}")


if __name__ == "__main__":
    main()
