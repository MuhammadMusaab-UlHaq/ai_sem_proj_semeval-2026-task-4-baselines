"""
Assignment 1: Comparative analysis of synthetic vs human/LLM datasets

This script loads the human dev set (`data/dev_track_a.jsonl`) and one or more
synthetic JSONL files (for example `data/combined_synthetic_for_training.jsonl`),
computes summary statistics and n-gram counts, and writes CSV summaries and
optional plots to `reports/assignment_1/`.

Usage:
    python assignment_1/assignment_1_compare.py

It auto-detects the data files in the repository's `data/` folder by default.
"""
from pathlib import Path
import json
import re
from collections import Counter
import csv
import statistics
import sys

try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "reports" / "assignment_1"
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


_token_re = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    if not text:
        return []
    return _token_re.findall(text.lower())


def ngrams(tokens, n):
    return zip(*(tokens[i:] for i in range(n)))


def analyze_texts(texts, top_k=50):
    tokens_per = []
    vocab = Counter()
    uni = Counter()
    bi = Counter()
    tri = Counter()
    for t in texts:
        toks = tokenize(t)
        tokens_per.append(len(toks))
        vocab.update(toks)
        uni.update(toks)
        bi.update([" ".join(bg) for bg in ngrams(toks, 2)])
        tri.update([" ".join(tg) for tg in ngrams(toks, 3)])

    summary = {
        "count_texts": len(texts),
        "avg_tokens": statistics.mean(tokens_per) if tokens_per else 0,
        "median_tokens": statistics.median(tokens_per) if tokens_per else 0,
        "stdev_tokens": statistics.pstdev(tokens_per) if tokens_per else 0,
        "vocab_size": len(vocab),
        "tokens_per": tokens_per,
    }
    summary.update({
        "top_unigrams": uni.most_common(top_k),
        "top_bigrams": bi.most_common(top_k),
        "top_trigrams": tri.most_common(top_k),
    })
    return summary


def write_summary_csv(prefix, stats):
    out = OUT_DIR / f"{prefix}_summary.csv"
    rows = [["metric", "value"]]
    rows.append(["count_texts", stats["count_texts"]])
    rows.append(["avg_tokens", stats["avg_tokens"]])
    rows.append(["median_tokens", stats["median_tokens"]])
    rows.append(["stdev_tokens", stats["stdev_tokens"]])
    rows.append(["vocab_size", stats["vocab_size"]])
    with open(out, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    def write_ngrams(name, items):
        p = OUT_DIR / f"{prefix}_{name}.csv"
        with open(p, "w", newline='', encoding="utf-8") as g:
            w = csv.writer(g)
            w.writerow([name, "count"])
            for k, c in items:
                w.writerow([k, c])

    write_ngrams("top_unigrams", stats["top_unigrams"])
    write_ngrams("top_bigrams", stats["top_bigrams"])
    write_ngrams("top_trigrams", stats["top_trigrams"])


def analyze_human_dev(path):
    texts = []
    label_counts = Counter()
    records = list(read_jsonl(path))
    for r in records:
        if "text_a" in r:
            texts.append(r.get("text_a", ""))
        if "text_b" in r:
            texts.append(r.get("text_b", ""))
        if "text_a_is_closer" in r:
            label_counts.update([str(bool(r.get("text_a_is_closer")))])

    stats = analyze_texts(texts)
    stats["num_pairs"] = len(records)
    stats["label_counts"] = dict(label_counts)
    return stats


def analyze_synthetic(path):
    texts = []
    records = 0
    for r in read_jsonl(path):
        records += 1
        for key in ("anchor_story", "similar_story", "dissimilar_story"):
            if key in r:
                texts.append(r.get(key, ""))
    stats = analyze_texts(texts)
    stats["num_records"] = records
    return stats


def plot_token_length_hist(human_tokens, synth_tokens):
    plt.figure(figsize=(8,5))
    bins = range(0, max(max(human_tokens or [0]), max(synth_tokens or [0])) + 20, 10)
    plt.hist(human_tokens, bins=bins, alpha=0.6, label='human')
    plt.hist(synth_tokens, bins=bins, alpha=0.6, label='synthetic')
    plt.xlabel('Tokens per text')
    plt.ylabel('Count')
    plt.title('Token length distribution (human vs synthetic)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'token_length_hist.png')
    # also save PDF in plots folder
    plt.savefig(PLOT_DIR / 'token_length_hist.pdf')
    plt.close()


def plot_top_unigrams(name, items):
    tokens = [t for t, _ in items]
    counts = [c for _, c in items]
    plt.figure(figsize=(10,4))
    plt.bar(tokens[::-1], counts[::-1])
    plt.xticks(rotation=45, ha='right')
    plt.title(name)
    plt.tight_layout()
    filename_png = name.lower().replace(' ', '_') + '.png'
    filename_pdf = name.lower().replace(' ', '_') + '.pdf'
    plt.savefig(OUT_DIR / filename_png)
    # save PDF into plots folder as requested
    plt.savefig(PLOT_DIR / filename_pdf)
    plt.close()


def main():
    human_file = DATA_DIR / "dev_track_a.jsonl"
    synth_file = DATA_DIR / "combined_synthetic_for_training.jsonl"

    outputs = {}
    if human_file.exists():
        print(f"Analyzing human dev set: {human_file}")
        outputs["human"] = analyze_human_dev(human_file)
        write_summary_csv("human_dev", outputs["human"])
    else:
        print("Human dev file not found at", human_file)

    if synth_file.exists():
        print(f"Analyzing synthetic set: {synth_file}")
        outputs["synthetic"] = analyze_synthetic(synth_file)
        write_summary_csv("synthetic_combined", outputs["synthetic"])
    else:
        print("Synthetic file not found at", synth_file)

    for k, v in outputs.items():
        print("---", k)
        for kk in ("count_texts", "vocab_size", "avg_tokens", "median_tokens"):
            print(f"{kk}: {v.get(kk)}")

    if MATPLOTLIB_AVAILABLE:
        print('Matplotlib available: generating plots...')
        human_tokens = outputs.get('human', {}).get('tokens_per', [])
        synth_tokens = outputs.get('synthetic', {}).get('tokens_per', [])
        plot_token_length_hist(human_tokens, synth_tokens)

        # top 20 unigrams
        human_unis = outputs.get('human', {}).get('top_unigrams', [])[:20]
        synth_unis = outputs.get('synthetic', {}).get('top_unigrams', [])[:20]
        if human_unis:
            plot_top_unigrams('human_top_unigrams', human_unis)
        if synth_unis:
            plot_top_unigrams('synthetic_top_unigrams', synth_unis)
        print('Plots saved to', OUT_DIR)
    else:
        print('Matplotlib not available — install matplotlib to generate plots')

    print("Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
