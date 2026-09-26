"""The 35B on the cloze form, through the raw completion endpoint.

The chat endpoint cannot be asked this question: a cloze is a continuation,
and an instruction-tuned model handed 「当別町は」 in a chat turn answers
about it rather than finishing it. /v1/completions is the same model without
the template.
"""
import json, random, sys, collections
import httpx
from datasets import load_dataset

# The endpoint and the name it serves, because neither is guessable. Any
# OpenAI-compatible completion endpoint will do; this was measured against a
# llama.cpp server on a local network.
#
#     python3 tools/ceiling_cloze.py out.json http://localhost:8080 my-model
if len(sys.argv) != 4:
    raise SystemExit("usage: ceiling_cloze.py OUT.json URL MODEL\n"
                     "  URL   an OpenAI-compatible endpoint, for instance "
                     "http://localhost:8080\n"
                     "  MODEL the name that endpoint serves; ask it with "
                     "curl $URL/v1/models")
OUT, URL, MODEL = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]
N = 400


def bare(n):
    return n[:-1] if n and n[-1] in "県府都道" else n


rows = [dict(r) for r in load_dataset("yuiseki/geo-triples-jp-gov", "probe",
                                      split="train")]
train = sorted([r for r in rows if r["split"] == "train"],
               key=lambda r: r["child_id"])
held = sorted([r for r in rows if r["split"] == "eval"],
              key=lambda r: r["child_id"])
random.Random(3).shuffle(train)

out = {}
with httpx.Client(timeout=120.0) as c:
    for label, group in (("train", train[:N]), ("eval", held)):
        hits, wrong = 0, []
        for i, r in enumerate(group):
            try:
                got = c.post(f"{URL}/v1/completions",
                             json={"model": MODEL, "temperature": 0,
                                        "max_tokens": 16,
                                        "prompt": r["child_ja"] + "は"}
                             ).json()["choices"][0]["text"]
            except Exception as e:
                got = f"ERROR {type(e).__name__}"
            first = (got or "").strip().split("\n")[0]
            if first and bare(r["parent_ja"]) in first:
                hits += 1
            elif len(wrong) < 5:
                wrong.append(f"{r['child_ja']} -> {r['parent_ja']} | {first[:40]}")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(group)}", flush=True)
        out[f"cloze {label}"] = {"correct": hits, "n": len(group),
                                 "accuracy": hits / len(group), "wrong": wrong}
        print(f"35B cloze {label}  {hits}/{len(group)}  {hits / len(group):.1%}",
              flush=True)
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
