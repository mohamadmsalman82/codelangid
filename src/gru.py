"""
CodeLangID -- alternative architecture: bidirectional GRU over the same
character sequence. Reported as a comparison against the char-CNN.

Usage:  python3 src/gru.py --variant raw
Output: results/gru_<variant>.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cnn import DEV, LABELS, RES, V, evaluate, load

SEED = 360
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class CharBiGRU(nn.Module):
    def __init__(self, vocab=V, emb=32, hidden=64, classes=len(LABELS), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, classes)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.gru(e)
        return self.fc(self.drop(h.max(dim=1).values))


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

    model = CharBiGRU().to(DEV)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[gru/{a.variant}] device={DEV} params={nparam:,}")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best, bad = 0.0, 0
    ckpt = RES / f"gru_{a.variant}.pt"
    t0 = time.time()

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
            best, bad = vacc, 0
            torch.save(model.state_dict(), ckpt)
            flag = " *"
        else:
            bad += 1
        print(f"  ep {ep:2d} train_loss {tot/n:.4f} acc {corr/n:.4f} | "
              f"val_loss {vl:.4f} acc {vacc:.4f}{flag}")
        if bad >= a.patience:
            print("  early stop")
            break
    train_time = time.time() - t0

    model.load_state_dict(torch.load(ckpt))
    out = {"variant": a.variant, "params": nparam, "history": hist,
           "train_time_s": train_time, "labels": LABELS}
    for split, loader in (("val", va), ("test", te), ("heldout", ho)):
        l, acc, _, _ = evaluate(model, loader)
        out[split] = {"loss": l, "accuracy": acc}
        print(f"  {split:8s} loss {l:.4f} acc {acc:.4f}")
    (RES / f"gru_{a.variant}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote -> {RES/f'gru_{a.variant}.json'}  ({train_time:.0f}s)")


if __name__ == "__main__":
    main()
