"""Step 5 - blinded human evaluation.

    python human_eval.py make --n 12 --raters A B C
    (each rater fills their own outputs/ratings/rater_X.csv, 1-5 per column)
    python human_eval.py aggregate

`make` samples from Part-4 outputs, hides the model name, shuffles, and writes the key separately.
Don't open key.json until the ratings are done.
"""
import argparse, csv, glob, json, os, random
from collections import defaultdict
from itertools import combinations

import numpy as np

CRITERIA = ["coherence", "grammar", "relevance", "creativity"]


def make(a):
    samples = [s for s in json.load(open(f"{a.out}/samples.json")) if s["part"] == 4]
    random.Random(a.seed).shuffle(samples)
    # balance models as much as possible
    by = defaultdict(list)
    for s in samples: by[s["model"]].append(s)
    picked = []
    for i in range(max(map(len, by.values()))):
        for m in by:
            if i < len(by[m]) and len(picked) < a.n: picked.append(by[m][i])
    random.Random(a.seed + 1).shuffle(picked)
    os.makedirs(f"{a.out}/ratings", exist_ok=True)
    key = {}
    with open(f"{a.out}/blinded_samples.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "prompt", "continuation"])
        for i, s in enumerate(picked, 1):
            w.writerow([i, s["prompt"], s["text"]])
            key[i] = dict(model=s["model"], config=s["config"], prompt_id=s["prompt_id"])
    json.dump(key, open(f"{a.out}/key.json", "w"), indent=1)
    for r in a.raters:
        with open(f"{a.out}/ratings/rater_{r}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id"] + CRITERIA)
            for i in key: w.writerow([i] + [""] * len(CRITERIA))
    print(f"{len(picked)} blinded samples -> {a.out}/blinded_samples.csv ; rating templates in {a.out}/ratings/")


def aggregate(a):
    key = json.load(open(f"{a.out}/key.json"))
    ratings = {}  # rater -> id -> {crit: val}
    for fp in sorted(glob.glob(f"{a.out}/ratings/rater_*.csv")):
        rows = list(csv.DictReader(open(fp)))
        rows = [r for r in rows if all(r[c].strip() for c in CRITERIA)]
        if rows:
            ratings[os.path.basename(fp)[6:-4]] = {r["id"]: {c: float(r[c]) for c in CRITERIA} for r in rows}
    if not ratings:
        print("no filled ratings yet"); return
    print(f"raters: {list(ratings)}")
    for group in ("model", "config"):
        print(f"\n== mean score by {group} ==")
        agg = defaultdict(lambda: defaultdict(list))
        for rid in ratings.values():
            for i, sc in rid.items():
                for c in CRITERIA: agg[key[i][group]][c].append(sc[c])
        for g, d in agg.items():
            print(g, {c: round(float(np.mean(v)), 2) for c, v in d.items()})
    print("\n== disagreement: mean abs diff between rater pairs (lower = more agreement) ==")
    for c in CRITERIA:
        diffs = [abs(ratings[x][i][c] - ratings[y][i][c])
                 for x, y in combinations(ratings, 2) for i in ratings[x] if i in ratings[y]]
        print(c, round(float(np.mean(diffs)), 2) if diffs else "n/a (need >=2 raters)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["make", "aggregate"])
    p.add_argument("--out", default="outputs")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--raters", nargs="+", default=["A", "B", "C"])
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    make(a) if a.cmd == "make" else aggregate(a)
