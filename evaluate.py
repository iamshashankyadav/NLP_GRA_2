"""Step 4 - quantitative evaluation (Part 5).

    python evaluate.py

Metrics:
  perplexity        val + test, from the best checkpoint
  distinct-1/2      unique unigrams (bigrams) / total, pooled over all samples of a model+config
  repetition rate   per sample: 1 - (unique trigrams / total trigrams); averaged. 0 = no repeated trigram.
  avg length        mean generated tokens
  premature         ended with <eos> before min_len (default 10) tokens
  excessive         hit max_new without ever emitting <eos>
Prompts are the first `prompt_len` tokens of held-out TEST stories.
"""
import argparse, csv, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import StoryDataset, Vocab, load_model, make_collate, perplexity, tokenize
from generate import CONFIGS, generate


def distinct_n(seqs, n):
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(seq, n=3):
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="outputs")
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--prompt_len", type=int, default=15)
    p.add_argument("--max_new", type=int, default=150)
    p.add_argument("--min_len", type=int, default=10)
    p.add_argument("--max_len", type=int, default=200)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocab.load(f"{a.data}/vocab.json")
    ids = torch.load(f"{a.data}/ids.pt")
    coll = make_collate(vocab.pad)
    loader = lambda k: DataLoader(StoryDataset(ids[k], vocab, a.max_len), 64, collate_fn=coll)

    rows, hists = [], {}
    texts = json.load(open(f"{a.data}/test_texts.json"))["test_texts"][: a.n_prompts]
    prompts = [" ".join(tokenize(t)[: a.prompt_len]) for t in texts]   # raw words, not ids, so no <unk> text
    for kind in ("rnn", "lstm"):
        model = load_model(f"{a.runs}/{kind}.pt", dev)
        h = json.load(open(f"{a.runs}/{kind}_history.json")); hists[kind] = h
        v_ppl, _ = perplexity(model, loader("val"), vocab.pad, dev)
        t_ppl, _ = perplexity(model, loader("test"), vocab.pad, dev)
        for cn, cfg in CONFIGS.items():
            outs, fins, lens = [], [], []
            for i, pr in enumerate(prompts):
                _, fin, n, toks = generate(model, vocab, pr, **cfg, max_new=a.max_new, seed=i, device=dev)
                outs.append(toks); fins.append(fin); lens.append(n)
            rows.append(dict(
                model=kind, config=cn, params=h["params"], train_seconds=h["seconds"],
                val_ppl=round(v_ppl, 2), test_ppl=round(t_ppl, 2),
                distinct1=round(distinct_n(outs, 1), 4), distinct2=round(distinct_n(outs, 2), 4),
                repetition=round(float(np.mean([repetition_rate(o) for o in outs])), 4),
                avg_len=round(float(np.mean(lens)), 1),
                premature=round(float(np.mean([f and n < a.min_len for f, n in zip(fins, lens)])), 3),
                excessive=round(float(np.mean([not f for f in fins])), 3)))
            print(rows[-1])

    with open(f"{a.out}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for kind, h in hists.items():
        e = range(1, len(h["train_loss"]) + 1)
        ax[0].plot(e, h["train_loss"], label=f"{kind} train"); ax[0].plot(e, h["val_loss"], "--", label=f"{kind} val")
        ax[1].plot(e, h["val_ppl"], label=kind)
    ax[0].set(title="Loss", xlabel="epoch"); ax[1].set(title="Validation perplexity", xlabel="epoch")
    ax[0].legend(); ax[1].legend(); plt.tight_layout(); plt.savefig(f"{a.out}/loss_curves.png", dpi=150)
    print(f"saved {a.out}/metrics.csv and loss_curves.png")
