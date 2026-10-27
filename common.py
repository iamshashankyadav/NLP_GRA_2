"""Shared pieces: cleaning, tokenisation, vocabulary, dataset, and the two LMs."""
import json, random, re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]          # ids 0,1,2,3
TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9']")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def clean(text):
    text = (text.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text):
    return TOKEN_RE.findall(clean(text))


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r" ([.,!?;:])", r"\1", text)
    return text


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad, self.unk, self.bos, self.eos = (self.stoi[t] for t in SPECIALS)

    @classmethod
    def build(cls, token_lists, min_freq=3, max_size=10000):
        counts = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counts.most_common(max_size - len(SPECIALS)) if c >= min_freq]
        return cls(SPECIALS + words)

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk) for t in tokens]

    def decode(self, ids):
        return detokenize([self.itos[i] for i in ids if i not in (self.pad, self.bos, self.eos)])

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        json.dump(self.itos, open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))


class StoryDataset(Dataset):
    """One item = one whole story: <bos> w1 ... wn <eos>, cut to max_len.
    Stories are never concatenated, so the RNN state always restarts at a story boundary."""
    def __init__(self, id_lists, vocab, max_len=200):
        self.items = [[vocab.bos] + ids[: max_len - 2] + [vocab.eos] for ids in id_lists]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id):
    def collate(batch):
        L = max(len(s) for s in batch)
        padded = torch.full((len(batch), L), pad_id, dtype=torch.long)
        for i, s in enumerate(batch):
            padded[i, : len(s)] = torch.tensor(s)
        return padded[:, :-1], padded[:, 1:]      # input, next-token target
    return collate


class LM(nn.Module):
    """Embedding -> (RNN | LSTM) -> Linear. Only `kind` differs between Model A and B."""
    def __init__(self, kind, vocab_size, emb=256, hid=512, layers=1, dropout=0.3, pad_id=0):
        super().__init__()
        self.cfg = dict(kind=kind, vocab_size=vocab_size, emb=emb, hid=hid,
                        layers=layers, dropout=dropout, pad_id=pad_id)
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[kind]
        kw = dict(batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.rnn = rnn_cls(emb, hid, layers, **kw)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hid, vocab_size)

    def forward(self, x, h=None):
        o, h = self.rnn(self.drop(self.embed(x)), h)
        return self.out(self.drop(o)), h


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = LM(**ck["cfg"]).to(device)
    model.load_state_dict(ck["state"])
    return model.eval()


@torch.no_grad()
def perplexity(model, loader, pad_id, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        total += loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += (y != pad_id).sum().item()
    return float(np.exp(total / n)), total / n
