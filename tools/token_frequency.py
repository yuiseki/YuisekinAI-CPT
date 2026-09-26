"""Frequency, or dedication? Two hypotheses that make opposite predictions.

If a fact is hard to attach because the token's embedding is undertrained,
what matters is how often the token appears at all. If it is hard because the
token already means something else, what matters is what share of its
appearances are this place name. The two are not the same number and this
counts both.

The name's own occurrences are counted as its token sequence inside the
tokenised text, so 萩市 in running text counts and 萩 alone does not.
"""
import collections, json, sys
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
rows = json.load(open(sys.argv[1]))["rows"]

SUFFIX = {tok(s, add_special_tokens=False)["input_ids"][-1] for s in "市町村区"}
seqs = {}
for r in rows:
    ids = tuple(tok(r["child"], add_special_tokens=False)["input_ids"])
    seqs[r["child"]] = ids
first = collections.defaultdict(list)
for name, ids in seqs.items():
    first[ids[0]].append((ids, name))

freq = collections.Counter()
hits = collections.Counter()
ds = load_dataset("yuiseki/wikipedia-geotagged", "20260901.ja", split="train",
                  streaming=True)
buf, n = [], 0
def flush(buf):
    for ids in tok(buf, add_special_tokens=False)["input_ids"]:
        freq.update(ids)
        for i, t in enumerate(ids):
            for seq, name in first.get(t, ()):
                if ids[i:i + len(seq)] == list(seq):
                    hits[name] += 1
for i, doc in enumerate(ds):
    buf.append((doc.get("text") or "")[:4000])
    if len(buf) >= 200:
        flush(buf); buf = []
    if i + 1 >= 20000:
        break
if buf:
    flush(buf)
print(f"{sum(freq.values()):,} tokens counted", flush=True)

out = []
for r in rows:
    ids = seqs[r["child"]]
    body = [x for x in ids if x not in SUFFIX] or list(ids)
    # The piece the fact has to hang on: the rarest non-suffix token, since
    # that is the one that makes the name this name rather than another.
    key = min(body, key=lambda x: freq.get(x, 0))
    kf = freq.get(key, 0)
    out.append({**r, "key_freq": kf, "name_hits": hits[r["child"]],
                "dedication": (hits[r["child"]] / kf) if kf else None})
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False)


def table(title, rows_, key, buckets):
    print(f"\n== {title}")
    b = collections.defaultdict(lambda: [0, 0])
    for r in rows_:
        v = r[key]
        if v is None:
            continue
        for lo, hi, lab in buckets:
            if lo <= v < hi:
                b[lab][1] += 1
                b[lab][0] += (not r["ok"])
                break
    for _, _, lab in buckets:
        e, n_ = b[lab]
        if n_:
            print(f"   {lab:<12} {e:>4}/{n_:<5} {e/n_:6.1%}")


table("by frequency of the name's rarest non-suffix token", out, "key_freq",
      [(0, 100, "0-99"), (100, 1000, "100-999"), (1000, 10000, "1k-9k"),
       (10000, 10**9, "10k+")])
table("by dedication: share of that token's uses that are this place",
      out, "dedication",
      [(0, .01, "under 1%"), (.01, .1, "1-10%"), (.1, .5, "10-50%"),
       (.5, 10, "over 50%")])
print("\n== dedication, holding the token's frequency to 1k-9k")
mid = [r for r in out if r["dedication"] is not None and 1000 <= r["key_freq"] < 10000]
table("", mid, "dedication",
      [(0, .1, "under 10%"), (.1, .5, "10-50%"), (.5, 10, "over 50%")])
