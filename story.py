"""The Plot-Twist Generator: type the beginning of a story, get a continuation.

    python story.py                                   # interactive, asks for openings
    python story.py --prompt "The old clock stopped." # one-shot
    python story.py --model both --strategy topk --top_k 40 --seed 3
"""
import argparse

import torch

from common import Vocab, load_model
from generate import generate


def run(models, vocab, prompt, a, dev):
    for name, model in models.items():
        if a.strategy == "greedy":
            cfg = dict(strategy="greedy")
        elif a.strategy == "temp":
            cfg = dict(strategy="sample", temperature=a.temperature)
        else:
            cfg = dict(strategy="sample", temperature=a.temperature, top_k=a.top_k)
        text, finished, n, _ = generate(model, vocab, prompt, **cfg, max_new=a.max_new, seed=a.seed, device=dev)
        print(f"\n[{name} | {a.strategy} | {n} tokens{'' if finished else ' (hit length limit)'}]")
        print(f"{prompt} {text}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default=None, help="story opening; omit for interactive mode")
    p.add_argument("--model", choices=["rnn", "lstm", "both"], default="lstm")
    p.add_argument("--strategy", choices=["greedy", "temp", "topk"], default="topk")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--max_new", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--runs", default="runs")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocab.load(f"{a.data}/vocab.json")
    kinds = ["rnn", "lstm"] if a.model == "both" else [a.model]
    models = {k: load_model(f"{a.runs}/{k}.pt", dev) for k in kinds}

    if a.prompt:
        run(models, vocab, a.prompt, a, dev)
    else:
        print("Plot-Twist Generator. Type a story opening (empty line to quit).")
        while True:
            prompt = input("\nOpening> ").strip()
            if not prompt:
                break
            run(models, vocab, prompt, a, dev)
