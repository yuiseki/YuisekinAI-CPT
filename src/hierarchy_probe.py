#!/usr/bin/env python3
"""Which parent does this place have? Asked of a model, at two levels.

The questions are not built here. They are the frozen `probe` subset of
geo-triples-tokyo23, derived from the same triples as the training corpus and
pinned by the same digest, so a score names a dataset revision rather than
whatever answer key happened to be on disk. This file only asks them.

Which means the score is recall, not generalisation: every answer it asks for
is stated somewhere in the corpus the model was trained on. That is the
question being asked here, and the dataset card says so in the same words.

The municipality probe cannot be used against that corpus: 1 of its 1,134
places appears in it.

Chance is not one number here. A state's country is one of 258 and a ward's
prefecture is one of 47, so the two levels are reported apart and each with
its own chance. Reporting one figure over both would let the easier level
carry the harder one.

    python3 src/hierarchy_probe.py --model google/gemma-3-270m --n 200 \
        --set /path/to/probe.parquet     # or a local copy
    python3 src/hierarchy_probe.py --url http://10.108.45.102:8080 --model-name gvt-llm
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The published dataset, not a sibling checkout. A path next to this
# repository works on the machine it was written on and nowhere else, and a
# rented GPU is the case this has to work on.
DEFAULT_SET = "yuiseki/geo-triples-tokyo23"
DEFAULT_CONFIG = "probe"

# How many things the answer could be, per level. Used only to report chance
# beside the score, because 2% reads as knowledge without it. The dataset's
# manifest carries these; they are repeated here so that a score printed from
# a Parquet alone still says what luck would give.
CANDIDATES = {"state-in-country": 258, "ward-in-state": 47,
              "place-in-ward": 23}

QUESTION = {
    ("state-in-country", "en"): "Which country is {child} in? Answer with the country only.",
    ("state-in-country", "ja"): "{child}はどの国にありますか。国名だけ答えてください。",
    ("ward-in-state", "en"): "Which prefecture is {child} in? Answer with the prefecture only.",
    ("ward-in-state", "ja"): "{child}はどの都道府県にありますか。都道府県名だけ答えてください。",
    ("place-in-ward", "en"): "Which ward of Tokyo is {child} in? Answer with the ward only.",
    ("place-in-ward", "ja"): "{child}は東京都のどの区にありますか。区名だけ答えてください。",
}


def bare(name):
    """愛媛県 -> 愛媛, so an answer in either form counts.

    Not 区: 港区 and 港 are not interchangeable the way 愛媛県 and 愛媛 are,
    and dropping it would let 北区 match 北千住.
    """
    return name[:-1] if name and name[-1] in "県府都道" else name


def correct(answer, expected):
    return bool(answer) and bare(expected) in answer


def read_rows(path, config=DEFAULT_CONFIG, revision=None):
    """The questions, from the Hub or from a file.

    A repository id is anything without a path separator that is not on disk.
    Reading from the Hub by default is what makes a score name a dataset
    revision rather than whatever was in a directory beside this one.
    """
    if os.path.exists(path):
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            return pq.read_table(path).to_pylist()
        return json.load(open(path, encoding="utf-8"))
    from datasets import load_dataset
    return list(load_dataset(path, config, split="train", revision=revision))


def load(path, n, lang, seed=3, config=DEFAULT_CONFIG, revision=None):
    """n questions from each level, not n from the whole set.

    There are 2,896 state questions and 17 ward questions, so a sample drawn
    over both is a sample of the first: 150 drew one ward. The ward level is
    the one closest to the use this was built for, where a fine-tuned model
    writes a well-formed area with an invented parent.
    """
    rows = read_rows(path, config, revision)
    for r in rows:
        # The frozen subset names the English columns explicitly. The older
        # JSON called them child and parent; both are read so that a run
        # against a file kept from before the freeze still works.
        r.setdefault("child", r.get("child_en"))
        r.setdefault("parent", r.get("parent_en"))
    if lang == "ja":
        rows = [r for r in rows if r.get("child_ja") and r.get("parent_ja")]
    rows.sort(key=lambda r: (r["level"], r["child_id"]))
    by_level = collections.defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
    out = []
    for level in sorted(by_level):
        group = by_level[level]
        random.Random(seed).shuffle(group)
        out.extend(group[:n] if n else group)
    out.sort(key=lambda r: (r["level"], r["child_id"]))
    return out


def prompt(row, lang):
    child = row["child_ja"] if lang == "ja" else row["child"]
    return QUESTION[(row["level"], lang)].format(child=child)


def expected(row, lang):
    return row["parent_ja"] if lang == "ja" else row["parent"]


def report(label, answers):
    per = collections.defaultdict(lambda: [0, 0])
    wrong = []
    for row, lang, got in answers:
        ok = correct(got, expected(row, lang))
        per[row["level"]][1] += 1
        per[row["level"]][0] += ok
        if not ok and len(wrong) < 8:
            wrong.append((row, lang, got))
    print(f"\n{label}")
    for level, (hit, n) in sorted(per.items()):
        chance = 1 / CANDIDATES[level]
        print(f"  {level:18} {hit:5}/{n:<5} {hit/n:6.1%}   (chance {chance:.1%})")
    if wrong:
        print("  wrong:")
        for row, lang, got in wrong:
            child = row["child_ja"] if lang == "ja" else row["child"]
            print(f"    {child} -> {expected(row, lang)} | "
                  f"{got.strip().replace(chr(10), ' ')[:46]}")
    return {level: {"correct": h, "n": n, "accuracy": h / n,
                    "chance": 1 / CANDIDATES[level]}
            for level, (h, n) in per.items()}


def ask_local(model_path, rows, lang, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    # A base model has no chat template, and a model continued from one still
    # has none after training. Asking for a template that is not there raises;
    # asking the question as plain text is what a base model can answer at
    # all, so that is the fallback rather than a failure.
    templated = bool(getattr(tok, "chat_template", None))
    if not templated:
        print("  no chat template; asking as plain text", flush=True)
    out = []
    for i, row in enumerate(rows):
        text = (tok.apply_chat_template(
            [{"role": "user", "content": prompt(row, lang)}],
            tokenize=False, add_generation_prompt=True) if templated
            else prompt(row, lang) + "\n")
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        out.append((row, lang,
                    tok.decode(gen[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    return out


def ask_endpoint(url, model_name, rows, lang, timeout=120.0):
    import httpx

    out = []
    with httpx.Client(timeout=timeout) as c:
        for i, row in enumerate(rows):
            try:
                r = c.post(f"{url.rstrip('/')}/v1/chat/completions", json={
                    "model": model_name, "temperature": 0, "max_tokens": 24,
                    "messages": [{"role": "user",
                                  "content": prompt(row, lang) + " /no_think"}]})
                got = r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                got = f"ERROR {type(e).__name__}"
            out.append((row, lang, got))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--url")
    ap.add_argument("--model-name", default="gvt-llm")
    ap.add_argument("--set", default=DEFAULT_SET,
                    help="a Hugging Face dataset, or a local parquet or json")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--revision", default=None,
                    help="pin the dataset. Without it the score names the "
                         "dataset as it is today")
    ap.add_argument("--n", type=int, default=200, help="per language; 0 for all")
    ap.add_argument("--langs", nargs="*", default=["en", "ja"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not (a.model or a.url):
        ap.error("give --model or --url")

    scores = {}
    for lang in a.langs:
        rows = load(a.set, a.n, lang, config=a.config, revision=a.revision)
        print(f"{len(rows)} questions in {lang}")
        answers = (ask_endpoint(a.url, a.model_name, rows, lang) if a.url
                   else ask_local(a.model, rows, lang))
        scores[lang] = report(f"{a.label or a.model or a.model_name}  [{lang}]",
                              answers)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"model": a.label or a.model or a.model_name, "scores": scores},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
