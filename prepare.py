"""Step 1 - build splits (story level), vocabulary, and token-id files.

    python prepare.py --n_train 20000 --n_val 1000 --n_test 1000

Splitting happens BEFORE any sequence is made, so no story is shared across splits.
Val/test both come from TinyStories' own 'validation' split (disjoint slices).
"""
import argparse, json, os, random

import torch
from common import Vocab, set_seed, tokenize


def load_stories(a):
    if a.demo: 
        r = random.Random(0)
        names, things = ["tom", "mia", "sam", "lily"], ["ball", "kite", "cat", "box"]
        mk = lambda: (f"{r.choice(names)} found a {r.choice(things)}. "
                      f"{r.choice(names)} was happy. they played all day. the end.")
        return [mk() for _ in range(a.n_train)], [mk() for _ in range(a.n_val)], [mk() for _ in range(a.n_test)]
    
    raw = f"{a.out}/raw_stories.json"
    if os.path.exists(raw):
        d = json.load(open(raw))
        return d["train"], d["val"], d["test"]
    from datasets import load_dataset
    take = lambda split, n: [r["text"] for _, r in zip(range(n), load_dataset(
        "roneneldan/TinyStories", split=split, streaming=True).shuffle(seed=a.seed, buffer_size=10000))]
    tr = take("train", a.n_train)
    va = take("validation", a.n_val + a.n_test)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(train=tr, val=va[: a.n_val], test=va[a.n_val:]), open(raw, "w"))
    return tr, va[: a.n_val], va[a.n_val:]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--min_freq", type=int, default=3)
    p.add_argument("--max_vocab", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--out", default="data")
    a = p.parse_args()
    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    train, val, test = load_stories(a)
    # drop empty/very short stories
    train, val, test = ([s for s in sp if len(s.split()) >= 10] for sp in (train, val, test))
    tok = {k: [tokenize(s) for s in sp] for k, sp in dict(train=train, val=val, test=test).items()}

    vocab = Vocab.build(tok["train"], a.min_freq, a.max_vocab)   # train split only
    vocab.save(f"{a.out}/vocab.json")
    ids = {k: [vocab.encode(t) for t in v] for k, v in tok.items()}
    torch.save(ids, f"{a.out}/ids.pt")
    json.dump({"test_texts": test[:500]}, open(f"{a.out}/test_texts.json", "w"))

    unk = sum(i == vocab.unk for s in ids["train"] for i in s)
    total = sum(len(s) for s in ids["train"])
    info = dict(train=len(train), val=len(val), test=len(test), vocab=len(vocab),
                train_tokens=total, unk_rate=round(unk / total, 4), seed=a.seed)
    json.dump(info, open(f"{a.out}/data_info.json", "w"), indent=1)
    print(info)
