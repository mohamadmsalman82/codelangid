"""
CodeLangID -- dataset construction from the Rosetta Code corpus.

Produces train/val/test splits of 256-character code snippets over 10 languages,
in two variants: comments kept ("raw") and comments stripped ("nocomment").

Splitting is done by TASK, not by file: every snippet derived from a given
Rosetta task lands in exactly one split, so no task leaks across splits.

Usage:  python3 src/build_dataset.py
Output: data/processed/{raw,nocomment}_{train,val,test}.jsonl
        data/processed/stats.json
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 360
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data" / "RosettaCodeData-main" / "Task"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Rosetta directory name -> canonical label
LANGS = {
    "Python": "Python",
    "Java": "Java",
    "C++": "C++",
    "JavaScript": "JavaScript",
    "Go": "Go",
    "C": "C",
    "Ruby": "Ruby",
    "PHP": "PHP",
    "Rust": "Rust",
    "C-sharp": "C#",
}
LABELS = sorted(set(LANGS.values()))
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

WINDOW = 256          # fixed character window
MAX_WIN_PER_FILE = 3  # up to N windows sampled per source file
MIN_LINES = 3         # discard fragments shorter than this
MIN_CHARS = 64
MAX_COMMENT_FRAC = 0.5  # discard fragments that are mostly comments/markup

HASH_FAMILY = {"C", "Python", "Ruby", "PHP"}
SLASH_FAMILY = {"C", "C++", "Java", "JavaScript", "Go", "Rust", "C#", "PHP"}


# ---------------------------------------------------------------- cleaning

def strip_comments(text, lang):
    """Remove comments for `lang`. Deliberately conservative: string literals
    containing comment markers may be over-stripped, which is acceptable here
    because the same function is applied uniformly to every class."""
    out = text
    if lang in SLASH_FAMILY:
        out = re.sub(r"/\*.*?\*/", " ", out, flags=re.S)   # /* block */
        out = re.sub(r"(?m)//[^\n]*", "", out)              # // line
    if lang in HASH_FAMILY:
        out = re.sub(r"(?m)(?<!\$)#[^\n]*", "", out)        # # line (not PHP $#)
    if lang == "Python":
        out = re.sub(r'"""(?:.|\n)*?"""', " ", out)         # docstrings
        out = re.sub(r"'''(?:.|\n)*?'''", " ", out)
    if lang == "Ruby":
        out = re.sub(r"(?m)^=begin.*?^=end", " ", out, flags=re.S)
    return out


def comment_fraction(text, lang):
    stripped = strip_comments(text, lang)
    a, b = len(re.sub(r"\s", "", text)), len(re.sub(r"\s", "", stripped))
    return 0.0 if a == 0 else (a - b) / a


def normalise(text):
    """Aggressive normalisation used only for duplicate detection."""
    return re.sub(r"\s+", " ", text).strip()


def windows(text, k=MAX_WIN_PER_FILE):
    """Up to k non-overlapping WINDOW-char windows, each starting on a line
    boundary so fragments look like something a human would paste."""
    lines, res, buf, start = text.split("\n"), [], [], 0
    starts = [0]
    for i, ln in enumerate(lines):
        start += len(ln) + 1
        if start > WINDOW * len(starts) and len(starts) < k:
            starts.append(i)
    for s in starts:
        frag = "\n".join(lines[s:])[:WINDOW]
        if len(frag) >= MIN_CHARS and frag.count("\n") + 1 >= MIN_LINES:
            res.append(frag)
    return res[:k]


# ------------------------------------------------------- near-dup (MinHash)

def shingles(text, n=5):
    t = normalise(text)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def minhash_signature(sh, perms=64, mod=(1 << 61) - 1):
    if not sh:
        return np.zeros(perms, dtype=np.int64)
    h = np.array([int(hashlib.md5(s.encode()).hexdigest()[:15], 16) for s in sh],
                 dtype=np.int64)
    rng = np.random.default_rng(SEED)
    a = rng.integers(1, mod, perms)
    b = rng.integers(0, mod, perms)
    return np.array([((a[i] * h + b[i]) % mod).min() for i in range(perms)],
                    dtype=np.int64)


def lsh_dedup(records, perms=64, bands=16, thresh=0.8):
    """Drop near-duplicates (Jaccard >= thresh) within each language."""
    keep, dropped = [], 0
    by_lang = defaultdict(list)
    for r in records:
        by_lang[r["lang"]].append(r)

    for lang, rs in by_lang.items():
        sigs = [minhash_signature(shingles(r["text"]), perms) for r in rs]
        rows = perms // bands
        buckets = defaultdict(list)
        for i, sig in enumerate(sigs):
            for b in range(bands):
                key = (b, tuple(sig[b * rows:(b + 1) * rows].tolist()))
                buckets[key].append(i)

        dead = set()
        for idxs in buckets.values():
            if len(idxs) < 2:
                continue
            for x in range(len(idxs)):
                i = idxs[x]
                if i in dead:
                    continue
                for y in range(x + 1, len(idxs)):
                    j = idxs[y]
                    if j in dead:
                        continue
                    jac = (sigs[i] == sigs[j]).mean()
                    if jac >= thresh:
                        dead.add(j)
        dropped += len(dead)
        keep.extend(r for i, r in enumerate(rs) if i not in dead)
    return keep, dropped


# ------------------------------------------------------------------ build

def collect():
    raw, stats = [], Counter()
    exact = set()
    n_files = 0
    for task_dir in sorted(TASKS.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for d, label in LANGS.items():
            ld = task_dir / d
            if not ld.is_dir():
                continue
            for f in sorted(ld.iterdir()):
                if not f.is_file():
                    continue
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                n_files += 1
                if len(text.strip()) < MIN_CHARS:
                    stats["drop_tiny_file"] += 1
                    continue
                for frag in windows(text):
                    if comment_fraction(frag, label) > MAX_COMMENT_FRAC:
                        stats["drop_mostly_comment"] += 1
                        continue
                    h = hashlib.sha1(normalise(frag).encode()).hexdigest()
                    if h in exact:
                        stats["drop_exact_dup"] += 1
                        continue
                    exact.add(h)
                    raw.append({"task": task, "lang": label, "text": frag})
    stats["source_files"] = n_files
    return raw, stats


def main():
    print("collecting ...")
    recs, stats = collect()
    print(f"  {stats['source_files']} source files -> {len(recs)} candidate snippets")
    print(f"  dropped: tiny={stats['drop_tiny_file']} "
          f"mostly_comment={stats['drop_mostly_comment']} exact_dup={stats['drop_exact_dup']}")

    print("near-duplicate removal (MinHash/LSH, Jaccard>=0.8) ...")
    recs, ndup = lsh_dedup(recs)
    print(f"  dropped {ndup} near-duplicates -> {len(recs)} snippets")

    # ---- balance: cap every class at the size of the smallest class
    by_lang = defaultdict(list)
    for r in recs:
        by_lang[r["lang"]].append(r)
    cap = min(len(v) for v in by_lang.values())
    print(f"balancing: capping every class at {cap} (limiting class = "
          f"{min(by_lang, key=lambda k: len(by_lang[k]))})")
    balanced = []
    for lang, rs in by_lang.items():
        random.shuffle(rs)
        balanced.extend(rs[:cap])

    # ---- split BY TASK so no task appears in two splits
    tasks = sorted({r["task"] for r in balanced})
    random.shuffle(tasks)
    n = len(tasks)
    tr = set(tasks[: int(0.70 * n)])
    va = set(tasks[int(0.70 * n): int(0.85 * n)])
    split_of = lambda t: "train" if t in tr else ("val" if t in va else "test")

    buckets = defaultdict(list)
    for r in balanced:
        buckets[split_of(r["task"])].append(r)

    counts = {}
    for variant in ("raw", "nocomment"):
        for split, rs in buckets.items():
            path = OUT / f"{variant}_{split}.jsonl"
            with path.open("w") as fh:
                kept = 0
                for r in rs:
                    text = r["text"]
                    if variant == "nocomment":
                        text = strip_comments(text, r["lang"])
                        text = re.sub(r"\n\s*\n+", "\n", text).strip()
                        if len(text) < MIN_CHARS:
                            continue
                    fh.write(json.dumps({
                        "text": text,
                        "lang": r["lang"],
                        "label": LABEL2ID[r["lang"]],
                        "task": r["task"],
                    }) + "\n")
                    kept += 1
            counts[f"{variant}_{split}"] = kept

    per_class = {s: dict(Counter(r["lang"] for r in rs)) for s, rs in buckets.items()}
    stats_out = {
        "labels": LABELS,
        "source_files": stats["source_files"],
        "dropped": {k: stats[k] for k in
                    ("drop_tiny_file", "drop_mostly_comment", "drop_exact_dup")},
        "dropped_near_dup": ndup,
        "cap_per_class": cap,
        "n_tasks": {"train": len(tr), "val": len(va), "test": n - len(tr) - len(va)},
        "counts": counts,
        "per_class": per_class,
        "window": WINDOW,
        "seed": SEED,
    }
    (OUT / "stats.json").write_text(json.dumps(stats_out, indent=2))

    print("\nsplit sizes:")
    for k, v in sorted(counts.items()):
        print(f"  {k:22s} {v}")
    print(f"\ntasks: train={len(tr)} val={len(va)} test={n - len(tr) - len(va)}")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
