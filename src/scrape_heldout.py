"""
CodeLangID -- fresh held-out test set, collected from real GitHub repositories.

This data is NOT used anywhere in training, validation, or the Rosetta test
split. It is deliberately drawn from a different *distribution*: idiomatic
production library code, rather than Rosetta Code's short puzzle solutions.
It is the never-before-seen set the final model is judged on.

Every snippet is passed through the identical cleaning pipeline used for the
training corpus (src/build_dataset.py), then checked against the training
hashes so nothing that appears in training can appear here.

Usage:  python3 src/scrape_heldout.py
Output: data/processed/{raw,nocomment}_heldout.jsonl
        data/processed/heldout_stats.json
"""

import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

from build_dataset import (LABEL2ID, MIN_CHARS, MAX_COMMENT_FRAC, SEED,
                           comment_fraction, normalise, strip_comments, windows)

random.seed(SEED)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"

# Two real, widely-used libraries per language. None is a Rosetta Code source.
REPOS = {
    "Python":     [("psf/requests", ".py"), ("pallets/flask", ".py")],
    "Java":       [("google/gson", ".java"), ("square/retrofit", ".java")],
    "C++":        [("fmtlib/fmt", (".cc", ".cpp", ".hpp")), ("nlohmann/json", (".cpp", ".hpp"))],
    "JavaScript": [("lodash/lodash", ".js"), ("expressjs/express", ".js")],
    "Go":         [("spf13/cobra", ".go"), ("gin-gonic/gin", ".go"),
                   ("sirupsen/logrus", ".go")],
    "C":          [("nothings/stb", ".c"), ("antirez/sds", ".c"),
                   ("redis/redis", ".c"), ("curl/curl", ".c")],
    "Ruby":       [("sinatra/sinatra", ".rb"), ("rack/rack", ".rb")],
    "PHP":        [("guzzle/guzzle", ".php"), ("briannesbitt/Carbon", ".php")],
    "Rust":       [("BurntSushi/ripgrep", ".rs"), ("clap-rs/clap", ".rs")],
    "C#":         [("JamesNK/Newtonsoft.Json", ".cs"), ("AutoMapper/AutoMapper", ".cs"),
                   ("restsharp/RestSharp", ".cs")],
}

SKIP = ("vendor/", "node_modules/", "third_party/", "/dist/", ".min.")
MAX_FILES_PER_REPO = 25
S = requests.Session()
S.headers["User-Agent"] = "codelangid-aps360"


def tree(repo):
    for branch in ("HEAD", "main", "master"):
        r = S.get(f"https://api.github.com/repos/{repo}/git/trees/{branch}",
                  params={"recursive": "1"}, timeout=30)
        if r.status_code == 200:
            return r.json().get("tree", [])
        if r.status_code == 403:
            raise SystemExit(f"rate-limited on {repo}: {r.json().get('message')}")
    return []


def fetch(repo, path):
    for branch in ("main", "master"):
        r = S.get(f"https://raw.githubusercontent.com/{repo}/{branch}/{path}", timeout=30)
        if r.status_code == 200:
            return r.text
    return None


def main():
    train_hashes = set()
    for v in ("raw", "nocomment"):
        for s in ("train", "val", "test"):
            p = OUT / f"{v}_{s}.jsonl"
            if p.exists():
                for line in p.open():
                    train_hashes.add(
                        hashlib.sha1(normalise(json.loads(line)["text"]).encode()).hexdigest())
    print(f"loaded {len(train_hashes)} training-corpus hashes for leakage check")

    records, stats, seen = [], Counter(), set()
    for lang, repos in REPOS.items():
        got = 0
        for repo, exts in repos:
            exts = (exts,) if isinstance(exts, str) else exts
            try:
                entries = tree(repo)
            except SystemExit as e:
                print(e)
                return
            files = [e["path"] for e in entries
                     if e["type"] == "blob"
                     and e["path"].endswith(exts)
                     and not any(s in e["path"] for s in SKIP)
                     and e.get("size", 0) > 500]
            random.shuffle(files)
            for path in files[:MAX_FILES_PER_REPO]:
                text = fetch(repo, path)
                if not text:
                    continue
                for frag in windows(text):
                    if comment_fraction(frag, lang) > MAX_COMMENT_FRAC:
                        stats["drop_mostly_comment"] += 1
                        continue
                    h = hashlib.sha1(normalise(frag).encode()).hexdigest()
                    if h in seen:
                        stats["drop_dup"] += 1
                        continue
                    if h in train_hashes:
                        stats["drop_leak"] += 1
                        continue
                    seen.add(h)
                    records.append({"text": frag, "lang": lang, "repo": repo, "path": path})
                    got += 1
                time.sleep(0.05)
            print(f"  {lang:11s} {repo:28s} -> {got:4d} snippets so far")
    print(f"\ncollected {len(records)} snippets "
          f"(dropped: comment={stats['drop_mostly_comment']} "
          f"dup={stats['drop_dup']} leak={stats['drop_leak']})")

    by_lang = defaultdict(list)
    for r in records:
        by_lang[r["lang"]].append(r)
    cap = min(len(v) for v in by_lang.values())
    print(f"balancing held-out at {cap}/class "
          f"(limiting = {min(by_lang, key=lambda k: len(by_lang[k]))})")
    balanced = []
    for rs in by_lang.values():
        random.shuffle(rs)
        balanced.extend(rs[:cap])

    counts = {}
    for variant in ("raw", "nocomment"):
        path = OUT / f"{variant}_heldout.jsonl"
        with path.open("w") as fh:
            kept = 0
            for r in balanced:
                text = r["text"]
                if variant == "nocomment":
                    text = strip_comments(text, r["lang"])
                    text = re.sub(r"\n\s*\n+", "\n", text).strip()
                    if len(text) < MIN_CHARS:
                        continue
                fh.write(json.dumps({
                    "text": text, "lang": r["lang"], "label": LABEL2ID[r["lang"]],
                    "repo": r["repo"], "path": r["path"],
                }) + "\n")
                kept += 1
        counts[variant] = kept

    (OUT / "heldout_stats.json").write_text(json.dumps({
        "repos": {k: [r for r, _ in v] for k, v in REPOS.items()},
        "counts": counts,
        "cap_per_class": cap,
        "dropped": dict(stats),
        "per_class": dict(Counter(r["lang"] for r in balanced)),
    }, indent=2))
    print(f"wrote held-out: {counts}")


if __name__ == "__main__":
    main()
