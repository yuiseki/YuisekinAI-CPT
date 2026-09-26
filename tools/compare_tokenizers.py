"""How do two tokenizers cut the same 1,632 place names?

    python3 tools/compare_tokenizers.py Qwen/Qwen3-0.6B-Base llm-jp/llm-jp-3-440m

Run 2 showed that a name of two Qwen tokens does not learn, however many times
it is written, while a name of three does. The obvious next question is whether
a tokenizer built for Japanese cuts those same names differently, and whether
"two tokens" even names the same set of places under both.

This has to be answered before any training, because if llm-jp gives most of
these names one token each, the bands are not comparable and a run comparing
the two models would be comparing two different experiments.
"""
import collections
import json
import sys

from datasets import load_dataset
from transformers import AutoTokenizer


def counts(tok, names):
    return {n: len(tok(n, add_special_tokens=False)["input_ids"]) for n in names}


def main():
    a_name, b_name = sys.argv[1], sys.argv[2]
    rows = [dict(r) for r in
            load_dataset("yuiseki/geo-triples-jp-gov", "probe", split="train")]
    names = sorted({r["child_ja"] for r in rows if r["split"] == "train"})
    prefs = sorted({r["parent_ja"] for r in rows})
    print(f"{len(names)} municipality names, {len(prefs)} prefecture names\n")

    ta = AutoTokenizer.from_pretrained(a_name)
    tb = AutoTokenizer.from_pretrained(b_name)
    print(f"{'':34} {'vocab':>8}")
    for name, t in ((a_name, ta), (b_name, tb)):
        print(f"{name:34} {len(t):>8,}")

    a, b = counts(ta, names), counts(tb, names)
    print(f"\nmean tokens per municipality name: "
          f"{a_name} {sum(a.values())/len(a):.2f}, "
          f"{b_name} {sum(b.values())/len(b):.2f}")
    pa, pb = counts(ta, prefs), counts(tb, prefs)
    print(f"mean tokens per prefecture name:   "
          f"{a_name} {sum(pa.values())/len(pa):.2f}, "
          f"{b_name} {sum(pb.values())/len(pb):.2f}")

    print(f"\nwhere the {a_name} bands land under {b_name}")
    cross = collections.defaultdict(collections.Counter)
    for n in names:
        cross[a[n]][b[n]] += 1
    widths = sorted({b[n] for n in names})
    print("        " + "".join(f"{w:>6}" for w in widths) + "     n")
    for w in sorted(cross):
        row = cross[w]
        print(f"  {w:>3} -> " + "".join(f"{row[x] or '':>6}" for x in widths)
              + f" {sum(row.values()):>6}")

    print(f"\nthe 2-token names under {a_name}, cut by {b_name}")
    for n in sorted(n for n in names if a[n] == 2)[:12]:
        piece = tb.convert_ids_to_tokens(tb(n, add_special_tokens=False)["input_ids"])
        print(f"  {n:8} {b[n]}  {'|'.join(piece)}")

    json.dump({"a": a_name, "b": b_name, "names": {n: [a[n], b[n]] for n in names}},
              open(sys.argv[3], "w"), ensure_ascii=False, indent=1) \
        if len(sys.argv) > 3 else None


if __name__ == "__main__":
    main()
