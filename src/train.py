#!/usr/bin/env python3
"""Continue pretraining one model on one corpus, and watch what it forgets.

A plain loop rather than a Trainer, for two reasons. The dry run has to take
the same path as the real run, and on this machine that path may be the CPU,
which most of a Trainer's machinery is not built for. And the thing this
experiment has to see is forgetting, which means evaluating on a corpus the
run is not training on, at every checkpoint, rather than at the end.

That second point is the whole design. An earlier attempt at this on a 270M
instruction-tuned model ran 816 steps of full-weight CPT and scored 0.4% on a
task it had scored 94.2% on before, because the format it had been tuned to
produce was gone. The loss on the training corpus looked fine throughout.

    # dry run: a few steps on a few documents, on whatever hardware is here
    python3 src/train.py --model google/gemma-3-270m-it \
        --corpus data/ja-dry.bin --control data/en-dry.bin \
        --out runs/dry --max-steps 5 --batch-size 1 --block-size 256

    # the real thing
    python3 src/train.py --model google/gemma-3-270m-it \
        --corpus data/ja.bin --control data/en.bin --out runs/ja --epochs 1
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np


def load_tokens(path):
    """Memory-map the corpus. Never read it into memory; it does not fit."""
    return np.memmap(path, dtype=np.uint32, mode="r")


def split_holdout(n, holdout, block):
    """Where training stops and the held-out tail begins.

    A tail rather than a random sample, because windows overlap: a randomly
    chosen evaluation window shares tokens with the training windows either
    side of it, and reports a loss the model has already seen.
    """
    tail = max(block + 1, int(n * holdout))
    return max(0, n - tail)


def chance_loss(vocab_size):
    """The loss of a model that has learnt nothing: a uniform draw."""
    return math.log(vocab_size)


def looks_like_chance(loss, vocab_size, margin=1.0):
    """Whether a loss is close enough to a uniform draw to be suspicious.

    A pretrained model reading its own language lands far below this. Sitting
    near it means the model is being asked the wrong question, which is what a
    shifting mistake looks like from the outside.
    """
    return loss > chance_loss(vocab_size) - margin


def batch(tokens, lo, hi, batch_size, block, rng, device):
    """One batch of windows drawn from tokens[lo:hi].

    A window, not a pair: Transformers shifts labels itself, so the caller
    passes the same window as input_ids and as labels. Handing it a window
    already shifted by one asks for the token after next, and the loss sits at
    chance while the run completes normally.
    """
    import torch

    top = hi - block - 1
    if top <= lo:
        raise SystemExit(f"corpus slice {lo}:{hi} is shorter than a block of "
                         f"{block}; use --block-size below {hi - lo - 1}")
    starts = rng.integers(lo, top, size=batch_size)
    x = np.stack([tokens[s:s + block] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device, non_blocking=True)


def evaluate(model, tokens, lo, hi, batch_size, block, iters, device, seed):
    """Mean loss over a fixed set of windows, so two runs are comparable."""
    import torch

    rng = np.random.default_rng(seed)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x = batch(tokens, lo, hi, batch_size, block, rng, device)
            total += float(model(input_ids=x, labels=x).loss)
    model.train()
    return total / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="what to train on")
    ap.add_argument("--control", default=None,
                    help="a corpus NOT trained on, evaluated at every "
                         "checkpoint. Without it forgetting is invisible")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop early; 0 means run the epochs")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--holdout", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=[None, "bf16", "fp16", "fp32"])
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.dtype:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[a.dtype]
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(a.seed)
    tokens = load_tokens(a.corpus)
    cut = split_holdout(len(tokens), a.holdout, a.block_size)
    control = load_tokens(a.control) if a.control else None

    per_step = a.batch_size * a.grad_accum * a.block_size
    steps = a.max_steps or max(1, int(a.epochs * cut / per_step))
    print(f"device {device} {dtype}")
    print(f"corpus  {len(tokens):,} tokens, training on {cut:,}, "
          f"holding out {len(tokens) - cut:,}")
    if control is not None:
        print(f"control {len(control):,} tokens, never trained on")
    print(f"{steps:,} steps of {per_step:,} tokens "
          f"({a.batch_size} x {a.grad_accum} x {a.block_size})", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, min(a.warmup, steps // 10 + 1), steps)

    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "log.jsonl")
    rng = np.random.default_rng(a.seed)
    began = time.time()

    def checkpoint(step):
        row = {"step": step,
               "held_out": evaluate(model, tokens, cut, len(tokens),
                                    a.batch_size, a.block_size, a.eval_iters,
                                    device, a.seed)}
        if control is not None:
            row["control"] = evaluate(model, control, 0, len(control),
                                      a.batch_size, a.block_size, a.eval_iters,
                                      device, a.seed)
        row["elapsed_s"] = round(time.time() - began, 1)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        parts = " ".join(f"{k} {v:.4f}" for k, v in row.items()
                         if k in ("held_out", "control"))
        print(f"  step {step:>6} {parts}", flush=True)
        return row

    first = checkpoint(0)
    vocab = getattr(model.config, "vocab_size", None)
    if vocab and looks_like_chance(first["held_out"], vocab):
        raise SystemExit(
            f"the starting loss is {first['held_out']:.2f} against "
            f"{chance_loss(vocab):.2f} for a uniform draw over {vocab:,} "
            "tokens. A pretrained model reading its own language does far "
            "better than that, so something is wrong before any training has "
            "happened: the wrong tokenizer for this corpus, or labels shifted "
            "twice. Fix it rather than watching this number come down.")
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            x = batch(tokens, 0, cut, a.batch_size, a.block_size, rng, device)
            loss = model(input_ids=x, labels=x).loss / a.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == steps:
            checkpoint(step)

    last = checkpoint(steps) if steps % a.eval_every else None
    model.save_pretrained(a.out)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    print(f"saved to {a.out}")

    # Forgetting is the thing to shout about, so say it rather than leave it
    # in the log for somebody to notice.
    rows = [json.loads(l) for l in open(log_path, encoding="utf-8")]
    if control is not None and len(rows) > 1:
        rise = rows[-1]["control"] - rows[0]["control"]
        drop = rows[0]["held_out"] - rows[-1]["held_out"]
        def moved(x):
            return f"fell by {x:.4f}" if x > 0 else f"rose by {-x:.4f}"
        print(f"held-out loss {moved(drop)}; control loss {moved(-rise)}")
        if rise > 0.1:
            print("  the control corpus got worse. This is what catastrophic "
                  "forgetting looks like before it reaches a benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
