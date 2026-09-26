"""Is this model worth publishing? Four questions it has to answer."""
import json, random, sys, collections
from datasets import load_dataset

# The published model, not the directory it was uploaded from. A local path
# works too, as the first argument, which is how a run is measured before it
# is published.
#
#     python3 tools/battery.py                      # the published weights
#     python3 tools/battery.py tmp/<run> out.json   # a run not yet published
TRAINED = sys.argv[1] if len(sys.argv) > 1 else "yuiseki/qwen3-0.6b-jp-gov-v0.1"
BASE = "Qwen/Qwen3-0.6B-Base"
OUT = sys.argv[2] if len(sys.argv) > 2 else "battery.json"
N = 200


def bare(name):
    return name[:-1] if name and name[-1] in "県府都道" else name


def ok(got, want):
    first = (got or "").strip().split("\n")[0]
    if first.startswith("Q:") or first.startswith("Q："):
        return False
    return bool(first) and bare(want) in first


rows = [dict(r) for r in load_dataset("yuiseki/geo-triples-jp-gov", "probe",
                                      split="train")]
train = sorted([r for r in rows if r["split"] == "train"],
               key=lambda r: r["child_id"])
held = sorted([r for r in rows if r["split"] == "eval"],
              key=lambda r: r["child_id"])
random.Random(5).shuffle(train)
sample = train[:N]

# Every prefecture and the municipalities in it, for the reverse test. From
# the whole probe rather than the train half: a named municipality is right
# if it is in that prefecture, whether or not the corpus said so.
inside = collections.defaultdict(set)
for r in rows:
    inside[r["parent_ja"]].add(r["child_ja"])

# Four ways of asking that appear nowhere in the corpus. The eight templates
# it trained on are in src/phrasings.py; none of these is one of them.
UNSEEN = {
    "unseen cloze A": lambda r: f"{r['child_ja']}が属する都道府県は",
    "unseen cloze B": lambda r: f"{r['child_ja']}の位置する都道府県名は",
    "unseen qa": lambda r: "".join(
        f"Q: {d['child_ja']}は何県にありますか。\nA: {d['parent_ja']}\n\n"
        for d in train[N:N + 3]) + f"Q: {r['child_ja']}は何県にありますか。\nA:",
    "unseen romaji": lambda r: f"{r['child_en']}は",
}

PROSE = [
    "日本の首都は",
    "吾輩は猫である。名前は",
    "昨日の夜、友人と一緒に",
    "機械学習とは、",
    "この関数は引数を二つ取り、",
]


def run(path, label, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def ask(text, n=16):
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=n, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return tok.decode(g[0][ids["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    for name, build in UNSEEN.items():
        hits, wrong = 0, []
        for r in sample:
            got = ask(build(r))
            if ok(got, r["parent_ja"]):
                hits += 1
            elif len(wrong) < 4:
                wrong.append(f"{r['child_ja']} -> {r['parent_ja']} | "
                             f"{got.strip().splitlines()[0][:40] if got.strip() else ''}")
        out[f"{label} {name}"] = {"correct": hits, "n": len(sample),
                                  "accuracy": hits / len(sample),
                                  "wrong": wrong}
        print(f"{label:8} {name:16} {hits:4}/{len(sample)} "
              f"{hits / len(sample):6.1%}", flush=True)

    # Reverse: name a municipality of a prefecture. The corpus taught this
    # exact frame, so it is a fair test of whether the mapping is usable in
    # the other direction rather than only as a completion of a name.
    hits, said = 0, []
    for pref in sorted(inside):
        got = ask(f"{pref}の市区町村のひとつが", 12).strip().splitlines()
        got = got[0] if got else ""
        hit = any(m in got for m in inside[pref])
        hits += hit
        if len(said) < 6:
            said.append(f"{pref} -> {got[:24]} {'ok' if hit else 'no'}")
    out[f"{label} reverse"] = {"correct": hits, "n": len(inside),
                               "accuracy": hits / len(inside), "said": said}
    print(f"{label:8} {'reverse':16} {hits:4}/{len(inside)} "
          f"{hits / len(inside):6.1%}", flush=True)

    out[f"{label} prose"] = {p: ask(p, 24) for p in PROSE}
    del model
    torch.cuda.empty_cache()


out = {}
run(TRAINED, "trained", out)
run(BASE, "base", out)
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
print("wrote", OUT)
