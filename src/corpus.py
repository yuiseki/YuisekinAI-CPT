#!/usr/bin/env python3
"""Turn a published dataset into a flat array of token ids.

One file of uint32, documents separated by the tokenizer's end-of-text id, plus
a small JSON manifest saying what went in. Training then memory-maps the file
and samples windows from it, so a corpus larger than memory costs nothing to
train on and the dry run and the real run read the same way.

Not a datasets.Dataset held in memory: 311 million Japanese tokens is 1.2 GB as
uint32 and several times that as Python objects, and the shape training wants
is a contiguous array anyway.

    python3 src/corpus.py --dataset yuiseki/wikipedia-geotagged \
        --config 20260901.ja --model google/gemma-3-270m-it \
        --out data/ja.bin --limit 200          # dry run
"""
import argparse
import json
import os
import sys

import numpy as np

# uint32 because the vocabularies in play do not fit in 16 bits: gemma-3 has
# 262,144 entries and OLMo 2 has 100,352.
DTYPE = np.uint32
BATCH = 256


def manifest_path(out):
    return os.path.splitext(out)[0] + ".json"


def write(tokens_iter, out, dtype=DTYPE, flush_every=1_000_000):
    """Append batches of ids to one file, and report how many were written."""
    total = 0
    buf = []
    with open(out, "wb") as f:
        for ids in tokens_iter:
            buf.extend(ids)
            if len(buf) >= flush_every:
                np.asarray(buf, dtype=dtype).tofile(f)
                total += len(buf)
                buf = []
        if buf:
            np.asarray(buf, dtype=dtype).tofile(f)
            total += len(buf)
    return total


def documents(dataset, config, split, text_field, limit):
    """Stream the text of a published dataset, without downloading all of it."""
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            return
        text = row.get(text_field)
        if text:
            yield text


def encode(docs, tok, eos_id, batch=BATCH):
    """Tokenise in batches, separating documents with the end-of-text id.

    The separator is what tells the model a document ended. Without it a window
    that straddles two articles reads as one run-on article, and the model
    learns to continue from Kyoto into Kagoshima.
    """
    pending = []
    for text in docs:
        pending.append(text)
        if len(pending) >= batch:
            for ids in tok(pending, add_special_tokens=False)["input_ids"]:
                yield ids + [eos_id]
            pending = []
    if pending:
        for ids in tok(pending, add_special_tokens=False)["input_ids"]:
            yield ids + [eos_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yuiseki/wikipedia-geotagged")
    ap.add_argument("--config", required=True, help="a subset, like 20260901.ja")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--model", required=True, help="whose tokenizer to use")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many documents; 0 means all")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit(f"{a.model} has no eos token; nothing to separate "
                         "documents with")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    docs = documents(a.dataset, a.config, a.split, a.text_field, a.limit)
    n = write(encode(docs, tok, eos), a.out)

    meta = {
        "dataset": a.dataset, "config": a.config, "split": a.split,
        "model": a.model, "vocab_size": len(tok), "eos_token_id": eos,
        "dtype": np.dtype(DTYPE).name, "tokens": n,
        "documents_limit": a.limit or None,
        "bytes": os.path.getsize(a.out),
    }
    with open(manifest_path(a.out), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"{n:,} tokens -> {a.out} ({meta['bytes']/1e6:.1f} MB)")
    print(f"manifest    -> {manifest_path(a.out)}")

    # Everything is written and closed. Leave now rather than through the
    # interpreter's own exit: a streaming dataset stopped early by --limit
    # leaves an HTTP thread mid-retry, and finalising with that thread alive
    # aborts with PyGILState_Release. It looks like a failure and is not one,
    # which is worse than a failure.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
