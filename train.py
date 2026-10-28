"""Step 2 - train one model. Run it twice, once per kind, with identical settings.

    python train.py --kind rnn
    python train.py --kind lstm
"""
import argparse, json, os, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import LM, StoryDataset, Vocab, make_collate, perplexity, set_seed

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["rnn", "lstm"], required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--emb", type=int, default=256)
    p.add_argument("--hid", type=int, default=512)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    vocab = Vocab.load(f"{a.data}/vocab.json")
    ids = torch.load(f"{a.data}/ids.pt")
    coll = make_collate(vocab.pad)
    mk = lambda k, sh: DataLoader(StoryDataset(ids[k], vocab, a.max_len), a.batch, shuffle=sh, collate_fn=coll)
    tr, va = mk("train", True), mk("val", False)

    model = LM(a.kind, len(vocab), a.emb, a.hid, a.layers, a.dropout, vocab.pad).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{a.kind}: {n_params:,} params on {dev}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad)   # padding never counts

    hist = dict(train_loss=[], val_loss=[], val_ppl=[])
    best, bad, t0 = float("inf"), 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            tot += loss.item(); n += 1
        v_ppl, v_loss = perplexity(model, va, vocab.pad, dev)
        hist["train_loss"].append(tot / n); hist["val_loss"].append(v_loss); hist["val_ppl"].append(v_ppl)
        print(f"ep {ep}: train {tot/n:.3f} | val {v_loss:.3f} | val ppl {v_ppl:.2f}")
        if v_loss < best:
            best, bad = v_loss, 0
            torch.save(dict(cfg=model.cfg, state=model.state_dict()), f"{a.out}/{a.kind}.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break

    hist.update(kind=a.kind, params=n_params, seconds=round(time.time() - t0, 1),
                best_val_loss=best, seed=a.seed, args=vars(a))
    json.dump(hist, open(f"{a.out}/{a.kind}_history.json", "w"), indent=1)
