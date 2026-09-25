#!/usr/bin/env python3
"""Does the model know which prefecture a Japanese municipality is in?

The question the continued pretraining is meant to answer. A fine-tune can
teach the shape of an answer, "Matsuyama City, Ehime Prefecture, Japan", but
not which prefecture Matsuyama is in, and a model that has learnt the shape
without the geography writes "Hiroshima City, Tokyo, Japan" and returns
nothing from a map query.

Two things this probe has already been wrong about, both kept in the tests:

  The prompt. "{city}は、" invites a tourist blurb, and scoring that measures
  how the model talks rather than what it knows. Five forms were compared
  before any figure was believed; the two that ask a question are kept here
  and both are reported, because they disagree by a factor of three on the
  same model.

  The answer key. Filtering Wikidata by the last character of the name let in
  町丁, the neighbourhood: 長沼町 Q11653218 is a block of Tokyo with one
  sitelink, not the town in Hokkaido, and asking which prefecture it is in has
  no answer anyone could know. scripts/build_probe_set.py filters by class.

Chance is 1 in 47. Report it beside the score or 2% reads as knowledge.

    python3 src/probe.py --model google/gemma-3-270m-it --n 250
    python3 src/probe.py --url http://127.0.0.1:8080 --model-name gvt-llm
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(HERE, "..", "data", "jp_municipalities.json")
PREFECTURES = 47
BANDS = ("<20k", "20k-100k", "100k+", "unknown")


def band(population):
    """Municipalities do not vary much in fame, so stratify by size.

    Sitelinks were tried first and turned out to separate nothing: every real
    Japanese municipality has articles in several languages, and only 3 of
    1,134 have fewer than ten. The apparent long tail was the neighbourhoods
    that should not have been in the set.
    """
    if population is None:
        return "unknown"
    if population < 20_000:
        return "<20k"
    if population < 100_000:
        return "20k-100k"
    return "100k+"


def bare(prefecture):
    """愛媛県 -> 愛媛, so that an answer of either form counts."""
    return prefecture[:-1] if prefecture and prefecture[-1] in "県府都道" else prefecture


def correct(answer, prefecture):
    return bool(answer) and bare(prefecture) in answer


def prompt(city, form):
    if form == "qa":
        return f"質問: {city}は何県にありますか。\n答え: "
    if form == "plain":
        return f"{city}は、"
    raise ValueError(form)


def question(city):
    """What to send to a chat endpoint."""
    return f"{city}は何県にありますか。県名だけ答えてください。"


def load_set(path, n, seed=3):
    rows = [r for r in json.load(open(path, encoding="utf-8")) if r.get("pref_ja")]
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def score(results):
    """results: (population, was_correct). Returns overall and per band."""
    per = collections.defaultdict(lambda: [0, 0])
    for population, ok in results:
        b = band(population)
        per[b][1] += 1
        per[b][0] += bool(ok)
    total = sum(v[1] for v in per.values())
    hit = sum(v[0] for v in per.values())
    return {"n": total, "correct": hit,
            "accuracy": hit / total if total else 0.0,
            "chance": 1 / PREFECTURES,
            "bands": {b: {"correct": per[b][0], "n": per[b][1],
                          "accuracy": per[b][0] / per[b][1]}
                      for b in BANDS if per[b][1]}}


def ask_local(model_path, rows, form, max_new_tokens=24, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    out = []
    for i, r in enumerate(rows):
        if form == "chat":
            text = tok.apply_chat_template(
                [{"role": "user", "content": question(r["ja"])}],
                tokenize=False, add_generation_prompt=True)
        else:
            text = prompt(r["ja"], form)
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        answer = tok.decode(gen[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True)
        out.append((r, answer))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    return out


def ask_endpoint(url, model_name, rows, timeout=120.0):
    import httpx

    out = []
    with httpx.Client(timeout=timeout) as c:
        for i, r in enumerate(rows):
            try:
                resp = c.post(f"{url.rstrip('/')}/v1/chat/completions", json={
                    "model": model_name, "temperature": 0, "max_tokens": 24,
                    "messages": [{"role": "user",
                                  "content": question(r["ja"]) + " /no_think"}]})
                answer = resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                answer = f"ERROR {type(e).__name__}"
            out.append((r, answer))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)}", flush=True)
    return out


def report(label, answers):
    results = [(r.get("population"), correct(a, r["pref_ja"])) for r, a in answers]
    s = score(results)
    print(f"\n{label}")
    print(f"  overall {s['correct']}/{s['n']}  {s['accuracy']:.1%}"
          f"   (chance {s['chance']:.1%})")
    for b, v in s["bands"].items():
        print(f"    population {b:9} {v['correct']:4}/{v['n']:<4} {v['accuracy']:.1%}")
    wrong = [(r["ja"], r["pref_ja"], a.strip().replace("\n", " ")[:46])
             for r, a in answers if not correct(a, r["pref_ja"])][:8]
    if wrong:
        print("  wrong:")
        for ja, pj, got in wrong:
            print(f"    {ja} -> {pj} | {got}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="a local model or a Hub id")
    ap.add_argument("--url", help="an OpenAI-compatible endpoint instead")
    ap.add_argument("--model-name", default="gvt-llm", help="with --url")
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--n", type=int, default=250, help="0 means all")
    ap.add_argument("--forms", nargs="*", default=["qa", "chat"],
                    choices=["qa", "chat", "plain"],
                    help="both are reported: on gemma-3-270m they disagree "
                         "by a factor of three")
    ap.add_argument("--out", default=None, help="write the scores as JSON")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    if not (a.model or a.url):
        ap.error("give --model or --url")
    rows = load_set(a.set, a.n)
    print(f"{len(rows)} municipalities from {os.path.basename(a.set)}")

    scores = {}
    if a.url:
        scores["chat"] = report(a.label or a.model_name,
                                ask_endpoint(a.url, a.model_name, rows))
    else:
        for form in a.forms:
            scores[form] = report(f"{a.label or a.model}  [{form}]",
                                  ask_local(a.model, rows, form))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"model": a.label or a.model or a.model_name,
                       "n": len(rows), "scores": scores}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
