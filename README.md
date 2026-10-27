# Plot-Twist Generator (Q4)

Vanilla RNN vs LSTM word-level language models on TinyStories.

    pip install torch datasets numpy matplotlib

Run in order:

    python prepare.py --n_train 20000 --n_val 1000 --n_test 1000   # splits + vocab (seed 42)
    python train.py --kind rnn
    python train.py --kind lstm                                     # same settings, only the cell differs
    python generate.py                                              # Part 3 grid + Part 4 challenge
    python evaluate.py                                              # perplexity, distinct-n, repetition, curves
    python human_eval.py make --n 12 --raters A B C                 # blinded samples
    python human_eval.py aggregate                                  # after raters fill outputs/ratings/*.csv

Pipeline check without internet / GPU: add `--demo` to prepare.py and `--epochs 1` to train.py.

Seeds: data 42, training 42, generation seeds 0 and 1 (Part 3/4), seed = prompt index (evaluate.py).
Defaults: emb 256, hid 512, 1 layer, dropout 0.3, Adam 1e-3, batch 64, clip 1.0, max story len 200, vocab 10k (min freq 3).
