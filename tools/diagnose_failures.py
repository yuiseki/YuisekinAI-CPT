"""Which facts did it not learn, and is there a pattern?

    python3 tools/diagnose_failures.py out.json [model]

The model defaults to the published v0.1. Pass a local directory to
diagnose a later run: the point of the tool is to compare two runs band by
band, which means it has to load weights that are not on the Hub yet.
"""
import collections, json, sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

M = sys.argv[2] if len(sys.argv) > 2 else "yuiseki/qwen3-0.6b-jp-gov-v0.1"
rows = [dict(r) for r in load_dataset("yuiseki/geo-triples-jp-gov", "probe", split="train")]
train = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["child_id"])
# How many municipalities each prefecture has in the corpus: the majority answer
size = collections.Counter(r["parent_ja"] for r in train)

tok = AutoTokenizer.from_pretrained(M)
# The resident 35B holds most of both cards, so take a card only when what is
# free there exceeds what the weights need, and fall back to the CPU quietly.
# A diagnosis that evicts the machine's working model is not worth three minutes.
def device_with_room(need_gb=2.0):
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free / 1e9 > need_gb:
            return f"cuda:{i}"
    return "cpu"


DEV = device_with_room()
print(f"running on {DEV}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.float32 if DEV == "cpu" else torch.bfloat16).to(DEV).eval()


def bare(n):
    return n[:-1] if n and n[-1] in "県府都道" else n


out = []
for i, r in enumerate(train):
    ids = tok(r["child_ja"] + "は", return_tensors="pt").to(DEV)
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=14, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    got = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    first = got.strip().split("\n")[0]
    ok = bool(first) and bare(r["parent_ja"]) in first
    said = next((p for p in size if bare(p) in first), None)
    out.append({"child": r["child_ja"], "want": r["parent_ja"], "got": first,
                "said": said, "ok": ok,
                # Without the special tokens. llm-jp puts <s> in front and
                # Qwen puts nothing, so counting them made every llm-jp name
                # one token longer than every Qwen name and the bands of two
                # tokenizers stopped meaning the same thing.
                "name_tokens": len(tok(r["child_ja"],
                                       add_special_tokens=False)["input_ids"])})
    if (i + 1) % 400 == 0:
        print(f"  {i+1}/{len(train)}", flush=True)

json.dump({"size": dict(size), "rows": out},
          open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
hit = sum(r["ok"] for r in out)
print(f"{hit}/{len(out)} {hit/len(out):.1%}")
