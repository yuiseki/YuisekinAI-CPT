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
              "place-in-ward": 23,
              # geo-triples-japan, whose parent layers are the whole country:
              # 1,740 municipalities and 47 prefectures.
              "place-in-municipality": 1740,
              "municipality-in-prefecture": 47}

# Worked examples put in front of each question. Three is enough to show the
# shape; they come out of the pool so none of them is scored.
SHOTS = 3

QUESTION = {
    ("state-in-country", "en"): "Which country is {child} in?",
    ("state-in-country", "ja"): "{child}はどの国にありますか。",
    ("ward-in-state", "en"): "Which prefecture is {child} in?",
    ("ward-in-state", "ja"): "{child}はどの都道府県にありますか。",
    ("place-in-ward", "en"): "Which ward of Tokyo is {child} in?",
    ("place-in-ward", "ja"): "{child}は東京都のどの区にありますか。",
    ("place-in-municipality", "en"): "Which municipality of Japan is {child} in?",
    ("place-in-municipality", "ja"): "{child}はどの市区町村にありますか。",
    # The question this whole line of work started from. Until
    # geo-triples-japan it was the generalisation probe, asked about places
    # the corpus did not mention; now a corpus states it.
    ("municipality-in-prefecture", "en"): "Which prefecture is {child} in?",
    ("municipality-in-prefecture", "ja"): "{child}はどの都道府県にありますか。",
}

# Q and A on their own lines, and the last one left open. "Answer with the
# ward only" was in the question before; the format says it better, and a
# 0.6B base model needs the format. With the examples merely separated by a
# blank line it produced Chinese trivia, or repeated the first example's
# question back.
#
# The open turn ends at the colon with no space after it. A trailing space is
# its own token and the first token of the answer then arrives without its
# leading one: 文京区 came back as 京区, and every answer was wrong for a
# reason that had nothing to do with the model.
TURN = "Q: {q}\nA: {a}"
OPEN = "Q: {q}\nA:"


def bare(name):
    """愛媛県 -> 愛媛, so an answer in either form counts.

    Not 区: 港区 and 港 are not interchangeable the way 愛媛県 and 愛媛 are,
    and dropping it would let 北区 match 北千住.
    """
    return name[:-1] if name and name[-1] in "県府都道" else name


def correct(answer, expected):
    """Only the first line counts.

    The model is continuing a list of question-and-answer pairs, so after its
    answer it writes the next question, and that question names a place. A
    match anywhere in the completion would score the model for the example it
    was about to ask itself.
    """
    first = (answer or "").strip().split("\n")[0]
    if first.startswith("Q:") or first.startswith("Q："):
        # The model wrote the next question instead of an answer, and that
        # question names a place. Scoring it would credit the model for what
        # it was about to ask itself.
        return False
    return bool(first) and bare(expected) in first


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


def load(path, n, lang, seed=3, config=DEFAULT_CONFIG, revision=None,
         shots=SHOTS, exclude_leaks=False):
    """n questions from each level, not n from the whole set.

    There are 2,896 state questions and 17 ward questions, so a sample drawn
    over both is a sample of the first: 150 drew one ward. The ward level is
    the one closest to the use this was built for, where a fine-tuned model
    writes a well-formed area with an invented parent.
    """
    rows = read_rows(path, config, revision)
    rows = [dict(r) for r in rows]
    for r in rows:
        # The frozen subset names the English columns explicitly. The older
        # JSON called them child and parent; both are read so that a run
        # against a file kept from before the freeze still works.
        r.setdefault("child", r.get("child_en"))
        r.setdefault("parent", r.get("parent_en"))
    # Both languages filter, not only Japanese. 3,449 places carry no
    # name:en, and asking "which ward of Tokyo is None in" scores the model on
    # a question nobody could answer.
    rows = [r for r in rows
            if r.get("child_" + lang) and r.get("parent_" + lang)]
    if exclude_leaks:
        # Questions whose subject already names the answer: "which
        # municipality is 市立福島第三小学校 in" carries 福島市 inside it.
        # Answering those is reading, not recall, and a model that cannot do
        # the rest still scores on them.
        rows = [r for r in rows if not r.get("answer_in_child_" + lang)]
    rows.sort(key=lambda r: (r["level"], r["child_id"]))
    prefixes, rows = few_shot(rows, lang, shots)
    by_level = collections.defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
    out = []
    for level in sorted(by_level):
        group = by_level[level]
        random.Random(seed).shuffle(group)
        out.extend(group[:n] if n else group)
    out.sort(key=lambda r: (r["level"], r["child_id"]))
    return prefixes, out


def few_shot(rows, lang, k=SHOTS):
    """The first k questions of each level, as worked examples.

    A base model has no idea that a question wants an answer. Asked one cold
    it continues the question, which scores zero for a reason that has nothing
    to do with what it knows. Three solved examples of the same shape are the
    ordinary way to ask, and the same three are used before and after
    training, so what moves is the model rather than the protocol.

    Spread across the level rather than taken from its front, and no two with
    the same answer. The first three by id at the country level were Aruba and
    two provinces of Afghanistan, and the model answered アフガニスタン to
    every question for the rest of the run.

    Removed from the pool, so a demonstration is never also scored.
    """
    by_level = collections.defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
    shots, rest = {}, []
    for level in sorted(by_level):
        group = sorted(by_level[level], key=lambda r: r["child_id"])
        order = list(range(len(group)))
        random.Random(11).shuffle(order)
        picked, parents = [], set()
        for i in order:
            if len(picked) >= k:
                break
            if group[i]["parent_id"] in parents:
                continue
            parents.add(group[i]["parent_id"])
            picked.append(i)
        used = [group[i] for i in sorted(picked)]
        group = [r for i, r in enumerate(group) if i not in set(picked)]
        shots[level] = "".join(
            TURN.format(q=QUESTION[(level, lang)].format(
                child=u["child_ja"] if lang == "ja" else u["child"]),
                a=u["parent_ja"] if lang == "ja" else u["parent"]) + "\n\n"
            for u in used)
        rest.extend(group)
    return shots, rest


def prompt(row, lang, shots=None):
    child = row["child_ja"] if lang == "ja" else row["child"]
    q = QUESTION[(row["level"], lang)].format(child=child)
    return (shots or {}).get(row["level"], "") + OPEN.format(q=q)


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
                  f"{got.strip().split(chr(10))[0][:46]}")
    return {level: {"correct": h, "n": n, "accuracy": h / n,
                    "chance": 1 / CANDIDATES[level]}
            for level, (h, n) in per.items()}


def ask_local(model_path, rows, lang, device=None, shots=None, chat=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    # Asked for, never detected. Qwen3-0.6B-Base ships a chat template
    # although it is a base model, and wrapping a few-shot block in ChatML
    # makes it answer by continuing the list of questions: every score was
    # zero, and none of it was about the model's geography.
    templated = chat and getattr(tok, "chat_template", None)
    print(f"  asking as {'chat' if templated else 'plain text'}", flush=True)
    out = []
    for i, row in enumerate(rows):
        text = (tok.apply_chat_template(
            [{"role": "user", "content": prompt(row, lang, shots)}],
            tokenize=False, add_generation_prompt=True) if templated
            else prompt(row, lang, shots))
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        out.append((row, lang,
                    tok.decode(gen[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    return out


def ask_endpoint(url, model_name, rows, lang, timeout=120.0, shots=None):
    import httpx

    out = []
    with httpx.Client(timeout=timeout) as c:
        for i, row in enumerate(rows):
            try:
                r = c.post(f"{url.rstrip('/')}/v1/chat/completions", json={
                    "model": model_name, "temperature": 0, "max_tokens": 16,
                    "messages": [{"role": "user",
                                  "content": prompt(row, lang, shots)
                                             + " /no_think"}]})
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
    ap.add_argument("--n", type=int, default=200,
                    help="per level per language; 0 for all")
    ap.add_argument("--exclude-leaks", action="store_true",
                    help="drop questions whose subject names the answer. The "
                         "dataset marks them; this is how to score without "
                         "them")
    ap.add_argument("--chat", action="store_true",
                    help="wrap the question in the tokenizer's chat template. "
                         "Off by default: this is a completion task, and a "
                         "base model that ships a template is not a model "
                         "that was tuned to follow one")
    ap.add_argument("--shots", type=int, default=SHOTS,
                    help="worked examples before each question. A base model "
                         "asked cold continues the question instead of "
                         "answering it")
    ap.add_argument("--langs", nargs="*", default=["en", "ja"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not (a.model or a.url):
        ap.error("give --model or --url")

    scores = {}
    for lang in a.langs:
        shots, rows = load(a.set, a.n, lang, config=a.config,
                           revision=a.revision, shots=a.shots,
                           exclude_leaks=a.exclude_leaks)
        print(f"{len(rows)} questions in {lang}, {a.shots} shots each")
        answers = (ask_endpoint(a.url, a.model_name, rows, lang, shots=shots)
                   if a.url
                   else ask_local(a.model, rows, lang, shots=shots,
                                  chat=a.chat))
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
