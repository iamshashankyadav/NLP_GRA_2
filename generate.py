"""Step 3 - decoding strategies + Part 3 (decoding grid) and Part 4 (plot-twist challenge).

    python generate.py            # writes outputs/part3_decoding.md, outputs/part4_challenge.md, outputs/samples.json

Generated text and human commentary are kept in separate blocks. Nothing is edited by hand.
"""
import argparse, json, os

import torch

from common import Vocab, load_model, set_seed, tokenize

# Decoding configs. greedy ignores temperature/top_k.
CONFIGS = {
    "greedy":   dict(strategy="greedy"),
    "temp0.5":  dict(strategy="sample", temperature=0.5),
    "temp0.8":  dict(strategy="sample", temperature=0.8),
    "temp1.2":  dict(strategy="sample", temperature=1.2),
    "topk40":   dict(strategy="sample", temperature=1.0, top_k=40),
}

# NOTE: TinyStories has a small, child-level vocabulary. Words like "detective" or "telephone"
# may be <unk>; that is a real limitation and should be discussed in the report.
PART3_PROMPTS = {
    "mystery":            "One night, Mia heard a strange noise coming from the old box in the attic. She opened it and",
    "surprising_discovery": "Tom was digging in the garden when he found a shiny key. Next to it there was",
    "unusual_character":  "There was once a little dragon who was afraid of fire. Every day he",
    "problem_to_solve":   "The bridge to the village was broken and the children could not get to school. Sam said,",
    "unexpected_ending":  "Lily baked a big cake for her friend. She carried it carefully to the park, but when she got there,",
}
PART4_PROMPTS = {
    "exam_paper":   "A student opened her exam paper and saw that it had the answers to tomorrow's questions.",
    "robot_message": "An old robot got a message from its future self. The message said,",
    "detective":    "The detective solved the case before the crime had even happened.",
    "village":      "One morning the whole village woke up and nobody could remember the day before.",
    "letter":       "Mia got a letter with her name on it. It was written by someone from 100 years in the future.",
}
CHALLENGE_CONFIGS = ["greedy", "topk40"]    # at least two strategies, per the brief


@torch.no_grad()
def generate(model, vocab, prompt, strategy="greedy", temperature=1.0, top_k=None,
             max_new=150, seed=0, device="cpu"):
    """Returns (continuation_text, finished_with_eos, n_generated_tokens, token_ids)."""
    g = torch.Generator().manual_seed(seed)
    ids = [vocab.bos] + vocab.encode(tokenize(prompt))
    logits, h = model(torch.tensor([ids], device=device))     # run the prompt through the model
    last, out, finished = logits[0, -1], [], False
    for _ in range(max_new):
        last = last.clone()
        last[vocab.pad] = float("-inf"); last[vocab.bos] = float("-inf")   # never emit these
        if strategy == "greedy":
            nxt = int(last.argmax())
        else:
            last = last / temperature
            if top_k:
                kth = torch.topk(last, top_k).values[-1]
                last[last < kth] = float("-inf")                          # keep only k best
            probs = torch.softmax(last, -1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=g))
        if nxt == vocab.eos:
            finished = True; break
        out.append(nxt)
        logits, h = model(torch.tensor([[nxt]], device=device), h)
        last = logits[0, -1]
    return vocab.decode(out), finished, len(out), out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="outputs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--max_new", type=int, default=150)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocab.load(f"{a.data}/vocab.json")
    models = {k: load_model(f"{a.runs}/{k}.pt", dev) for k in ("rnn", "lstm")}
    samples = []

    # Part 3: every prompt x model x decoding config x seed
    md = ["# Part 3 - decoding strategies\n", f"seeds: {a.seeds}, max_new={a.max_new}\n"]
    for name, prompt in PART3_PROMPTS.items():
        md.append(f"\n## {name}\n\n**Prompt:** {prompt}\n")
        for mk, model in models.items():
            for cn, cfg in CONFIGS.items():
                for s in (a.seeds if cfg["strategy"] == "sample" else a.seeds[:1]):
                    txt, fin, n, _ = generate(model, vocab, prompt, **cfg, max_new=a.max_new, seed=s, device=dev)
                    md.append(f"- `{mk}` / `{cn}` / seed {s} ({n} tokens, ended={fin}): {txt}")
                    samples.append(dict(part=3, prompt_id=name, prompt=prompt, model=mk, config=cn,
                                        seed=s, text=txt, finished=fin, n_tokens=n))
    open(f"{a.out}/part3_decoding.md", "w").write("\n".join(md))

    # Part 4: plot-twist challenge, both models, two strategies
    md = ["# Part 4 - Plot-Twist Challenge\n",
          "Model output is in the quote blocks. Commentary lines are for YOU to fill in after reading.\n"]
    for name, prompt in PART4_PROMPTS.items():
        md.append(f"\n## {name}\n\n**Opening:** {prompt}\n")
        for mk, model in models.items():
            for cn in CHALLENGE_CONFIGS:
                txt, fin, n, _ = generate(model, vocab, prompt, **CONFIGS[cn], max_new=a.max_new,
                                          seed=a.seeds[0], device=dev)
                md.append(f"### {mk} / {cn}\n> {prompt} {txt}\n")
                md.append("- Commentary: connected to prompt? / twist? / consistent? / repetition? -> _fill in_\n")
                samples.append(dict(part=4, prompt_id=name, prompt=prompt, model=mk, config=cn,
                                    seed=a.seeds[0], text=txt, finished=fin, n_tokens=n))
    open(f"{a.out}/part4_challenge.md", "w").write("\n".join(md))
    json.dump(samples, open(f"{a.out}/samples.json", "w"), indent=1)
    print(f"wrote {len(samples)} samples to {a.out}/")
