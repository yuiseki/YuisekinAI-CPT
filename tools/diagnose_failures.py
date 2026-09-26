"""Which facts did it not learn, and is there a pattern?"""
import collections, json, sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "yuiseki/qwen3-0.6b-jp-gov-v0.1"
rows = [dict(r) for r in load_dataset("yuiseki/geo-triples-jp-gov", "probe", split="train")]
train = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["child_id"])
# How many municipalities each prefecture has in the corpus: the majority answer
size = collections.Counter(r["parent_ja"] for r in train)

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()


def bare(n):
    return n[:-1] if n and n[-1] in "県府都道" else n


out = []
for i, r in enumerate(train):
    ids = tok(r["child_ja"] + "は", return_tensors="pt").to("cuda")
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=14, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    got = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    first = got.strip().split("\n")[0]
    ok = bool(first) and bare(r["parent_ja"]) in first
    said = next((p for p in size if bare(p) in first), None)
    out.append({"child": r["child_ja"], "want": r["parent_ja"], "got": first,
                "said": said, "ok": ok,
                "name_tokens": len(tok(r["child_ja"])["input_ids"])})
    if (i + 1) % 400 == 0:
        print(f"  {i+1}/{len(train)}", flush=True)

json.dump({"size": dict(size), "rows": out},
          open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
hit = sum(r["ok"] for r in out)
print(f"{hit}/{len(out)} {hit/len(out):.1%}")
