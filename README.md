# CodeLangID

**Identifying the programming language of a source-code snippet with a character-level CNN.**

APS360: Applied Fundamentals of Deep Learning — Mohamad Salman (solo project).
Given a raw code snippet with no filename or extension, predict which of 10 languages
it is written in (Python, Java, C++, JavaScript, Go, C, Ruby, PHP, Rust, C#) from the
characters alone.

## Results

| Model | Variant | Params | Val | Test (Rosetta) | Held-out (GitHub) |
|---|---|---:|---:|---:|---:|
| TF-IDF + Naive Bayes (baseline) | raw | 43,573 ft | 91.80 | 90.14 | 79.59 |
| TF-IDF + Naive Bayes (baseline) | nocomment | 41,472 ft | 91.48 | 89.27 | 81.89 |
| Char-BiGRU (tried, rejected) | raw | 42,186 | 92.59 | 90.27 | 80.54 |
| **Char-CNN (primary)** | raw | 68,938 | **94.31** | **92.74** | **82.84** |
| **Char-CNN (primary)** | nocomment | 68,938 | 94.69 | 92.49 | 82.16 |

Accuracy %. Chance = 10% (the dataset is exactly balanced). "Held-out" is a
never-before-seen set collected from 24 real GitHub repositories — a different
distribution from Rosetta Code, never trained or tuned on, with 0 snippets leaked
from training.

## Architecture

```
raw snippet -> char IDs (256, vocab V=102)
            -> Embedding(102, 32)
            -> 3 parallel Conv1d branches, kernel widths 3 / 5 / 7, 128 filters + ReLU
            -> global max-pool per branch -> concat (384)
            -> Dropout(0.5) -> Linear(384, 10) -> softmax
```

68,938 trainable parameters. Trained from scratch with cross-entropy + Adam
(lr 1e-3, batch 64, early stopping on validation accuracy). ~35 s on an Apple
M-series GPU (MPS).

## Reproducing

Requires Python 3 with `torch`, `numpy`, `scikit-learn`, `matplotlib`, `requests`.
Everything is seeded (seed 360).

```bash
python3 src/build_dataset.py    # Rosetta Code -> balanced 10-class corpus (needs data/, see below)
python3 src/scrape_heldout.py   # 740 never-before-seen snippets from 24 GitHub repos
python3 src/baseline.py         # TF-IDF + Naive Bayes baseline
python3 src/cnn.py --variant raw        # primary char-CNN
python3 src/gru.py --variant raw        # BiGRU comparison (slow: ~11 min)
python3 src/figures.py          # report figures
```

The raw corpora are not committed (624 MB). Fetch them first:

```bash
cd data
curl -L -o rosetta.tar.gz https://codeload.github.com/acmeism/RosettaCodeData/tar.gz/refs/heads/main
tar xzf rosetta.tar.gz
```

Then build the report:

```bash
cd report && pdflatex progress_report && bibtex progress_report && pdflatex progress_report && pdflatex progress_report
```

## Data pipeline

Rosetta Code implements the same tasks across many languages, so labels come from
directory structure rather than a guess.

| # | Operation | Removed | Remaining |
|---|---|---:|---|
| 1 | Read 10 language directories | — | 18,654 files |
| 2 | Window: ≤3 line-aligned 256-char windows/file (≥3 lines, ≥64 char) | 1,125 files | |
| 3 | Comment/markup filter: drop if comments >50% of non-space chars | 3,460 win. | |
| 4 | Exact dedup: SHA-1 of whitespace-normalized text | 327 win. | 38,394 snippets |
| 5 | Near-dup dedup: MinHash (64 perm, 16 LSH bands), Jaccard ≥ 0.8 | 353 | 38,041 |
| 6 | Balance: cap each class at 1,036 = smallest class (PHP) | 27,681 | **10,360** |
| 7 | Split **by task** 70/15/15 (no task spans two splits) | — | 7,225 / 1,634 / 1,501 |
| 8 | Encode: vocab V=102 (ASCII + `<pad>` + `<unk>`), pad/truncate to 256 | — | — |

Splitting is by *task*, not by file — otherwise two solutions to "Bubble sort" land in
both train and test and inflate accuracy. Every snippet is emitted in two variants
(`raw` and `nocomment`) to test how much the model leans on comment prose.

## Two findings worth knowing

**The model reads code, not comments.** Stripping comments costs the CNN only 0.25
points (92.74 → 92.49). The bag-of-n-grams baseline behaves oppositely — removing
comments *improves* its held-out score by 2.30 points (79.59 → 81.89) — so comment
prose actively misleads it on real-world code while the CNN is largely immune.

**PHP → Rust is a textbook distribution shift.** 13 of the 16 PHP→Rust errors contain
the token `#[`. In training, `#[` appears in 0.0% of PHP snippets (0/753) but 9.0% of
Rust (63/703), so the model learned the entirely reasonable rule `#[` ⇒ Rust. But PHP 8
spells attributes identically (`#[Group('...')]`), and those appear in 18.9% of held-out
PHP (14/74). Rosetta's PHP predates PHP 8, so the construct is invisible during training.
PHP's F1 collapses from 0.97 in-distribution to 0.66 held-out.

## Layout

```
src/      build_dataset.py, scrape_heldout.py, baseline.py, cnn.py, gru.py, figures.py
data/     processed/*.jsonl (committed), raw corpora (not committed)
results/  baseline.json, cnn_{raw,nocomment}.json, gru_raw.json
figures/  fig_pipeline, fig_confusion (pdf + png)
report/   progress_report.tex -> progress_report.pdf
```
