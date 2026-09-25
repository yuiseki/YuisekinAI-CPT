#!/usr/bin/env python3
"""What continued pretraining will cost in VRAM, before renting any.

Worth computing rather than guessing, because for a small model with a large
vocabulary the dominant term is not the model. gemma-3-270m has 262,144
entries and a hidden size of 640: the embedding is 62.6% of the parameters,
and the logits tensor the loss is computed over is batch x block x 262,144,
which passes the whole model in size at a batch of one and a block of 512.

    python3 src/budget.py --model google/gemma-3-270m
    python3 src/budget.py --model google/gemma-3-270m --batch 8 --block 1024
"""
import argparse
import sys

GB = 1024 ** 3


def parameters(cfg):
    """Count them from the architecture rather than by loading the weights."""
    v, h, L = cfg["vocab_size"], cfg["hidden_size"], cfg["layers"]
    heads, kv, hd = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    ff = cfg["intermediate"]
    embedding = v * h                      # tied, so counted once
    attn = h * heads * hd + h * kv * hd * 2 + heads * hd * h
    mlp = 3 * h * ff
    norms = 4 * h
    return {"embedding": embedding, "layers": L * (attn + mlp + norms),
            "total": embedding + L * (attn + mlp + norms)}


def budget(cfg, batch, block, dtype_bytes=2, optimizer="adamw",
           master_fp32=False, checkpointing=True):
    """Bytes, by what holds them.

    master_fp32 says which of two recipes is being costed, and the difference
    is 2.0 GB for this model: an fp32 master copy, and moments that are fp32
    rather than bf16. torch.optim.AdamW on a bf16 model keeps its two
    moments in bf16 and holds no master copy, which is what src/train.py does
    and what the measurement on an RTX 3060 matched: 2.54 GB at 1 x 512 where
    the mixed-precision figure is 4.75. It is also numerically the worse of
    the two, so the cheap answer is not automatically the right one.
    """
    p = parameters(cfg)["total"]
    out = {"parameters": p * dtype_bytes,
           "gradients": p * dtype_bytes}
    state_bytes = 4 if master_fp32 else dtype_bytes
    if optimizer == "adamw":
        out["optimizer"] = 2 * p * state_bytes
    elif optimizer == "sgd":
        out["optimizer"] = 0
    if master_fp32 and dtype_bytes < 4:
        out["master weights"] = p * 4

    # The loss is computed over every position, so the logits are the full
    # vocabulary wide. Transformers upcasts them to fp32 for cross-entropy,
    # and the upcast copy lives alongside the original.
    tokens = batch * block
    out["logits"] = tokens * cfg["vocab_size"] * dtype_bytes
    out["logits upcast"] = tokens * cfg["vocab_size"] * 4

    # Everything else. With checkpointing only one layer's activations are
    # live at a time; without it, all of them.
    per_layer = tokens * cfg["hidden_size"] * dtype_bytes * 12
    out["activations"] = per_layer * (1 if checkpointing else cfg["layers"])
    return out


CONFIGS = {
    "google/gemma-3-270m": dict(vocab_size=262144, hidden_size=640, layers=18,
                                heads=4, kv_heads=1, head_dim=256,
                                intermediate=2048),
    "google/gemma-3-270m-it": dict(vocab_size=262144, hidden_size=640, layers=18,
                                   heads=4, kv_heads=1, head_dim=256,
                                   intermediate=2048),
    "EleutherAI/pythia-160m": dict(vocab_size=50304, hidden_size=768, layers=12,
                                   heads=12, kv_heads=12, head_dim=64,
                                   intermediate=3072),
    "allenai/OLMo-2-0425-1B": dict(vocab_size=100352, hidden_size=2048, layers=16,
                                   heads=16, kv_heads=16, head_dim=128,
                                   intermediate=8192),
}


def fetch(model):
    """Read the architecture from the Hub when it is not one of the few here."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    t = getattr(c, "text_config", c)
    return dict(vocab_size=t.vocab_size, hidden_size=t.hidden_size,
                layers=t.num_hidden_layers, heads=t.num_attention_heads,
                kv_heads=getattr(t, "num_key_value_heads", t.num_attention_heads),
                head_dim=getattr(t, "head_dim", t.hidden_size // t.num_attention_heads),
                intermediate=t.intermediate_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-270m")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--master-fp32", action="store_true",
                    help="cost a mixed-precision recipe instead: fp32 master "
                         "weights and fp32 optimizer moments. 3 GB more for "
                         "this model, and the numerically sounder choice")
    ap.add_argument("--table", action="store_true",
                    help="sweep batch x block instead of one configuration")
    a = ap.parse_args()

    cfg = CONFIGS.get(a.model) or fetch(a.model)
    p = parameters(cfg)
    print(f"{a.model}")
    print(f"  parameters {p['total']:,}   embedding {p['embedding']:,} "
          f"({p['embedding']/p['total']:.1%})")
    db = 2 if a.dtype == "bf16" else 4

    if a.table:
        print(f"\n  peak GB, {a.dtype}, {a.optimizer}, "
              f"{'no ' if a.no_checkpointing else ''}gradient checkpointing")
        blocks = [512, 1024, 2048, 4096]
        print("    batch " + "".join(f"{b:>10}" for b in blocks))
        for bs in (1, 2, 4, 8, 16, 32):
            row = []
            for bl in blocks:
                b = budget(cfg, bs, bl, db, a.optimizer,
                           a.master_fp32, not a.no_checkpointing)
                row.append(sum(b.values()) / GB)
            print(f"    {bs:>5} " + "".join(f"{v:>10.2f}" for v in row))
        return 0

    b = budget(cfg, a.batch, a.block, db, a.optimizer,
               a.master_fp32, not a.no_checkpointing)
    print(f"\n  batch {a.batch} x block {a.block} = {a.batch * a.block:,} tokens, "
          f"{a.dtype}, {a.optimizer}"
          f"{', fp32 master' if a.master_fp32 else ''}")
    for k, v in sorted(b.items(), key=lambda kv: -kv[1]):
        print(f"    {k:18} {v/GB:8.3f} GB")
    print(f"    {'total':18} {sum(b.values())/GB:8.3f} GB")
    fixed = sum(v for k, v in b.items()
                if k in ("parameters", "gradients", "optimizer", "master weights"))
    print(f"\n  fixed (model and optimizer) {fixed/GB:.3f} GB")
    print(f"  per token of batch x block  "
          f"{(sum(b.values()) - fixed) / (a.batch * a.block) / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
