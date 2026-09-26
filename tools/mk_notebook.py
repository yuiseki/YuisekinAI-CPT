import argparse, json, os, sys

# Which model this notebook trains, and whose tokenizer decides how often a
# fact is written. The defaults reproduce the run that has already happened,
# so regenerating an existing notebook is a no-op rather than a rewrite of a
# record.
_ap = argparse.ArgumentParser()
_ap.add_argument("out", nargs="?")
_ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
_ap.add_argument("--schedule", default="Qwen/Qwen3-0.6B-Base",
                 help='"none" writes every fact exactly once')
_ap.add_argument("--epochs", type=float, default=60.0)
ARGS = _ap.parse_args()
MODEL_ID = ARGS.model
SCHEDULE_ID = None if ARGS.schedule.lower() == "none" else ARGS.schedule
EPOCHS_N = ARGS.epochs

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": list(lines)}

def src(text):
    """A cell body from a triple-quoted string, as nbformat wants it."""
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

cells = []

cells.append(md(*src(r'''
# Continued pretraining on geo-triples-jp-gov

One pass of continued pretraining on a corpus of spatial facts about Japan,
with the measurement on both sides of it. Runs top to bottom on a Colab A100
and needs nothing from disk.

The narrowest question that is still worth answering: can a model of well
under a billion parameters learn which prefecture each of Japan's
municipalities is in?

The facts come from the `probe` subset of
[`yuiseki/geo-triples-jp-gov`](https://huggingface.co/datasets/yuiseki/geo-triples-jp-gov),
computed from two frozen government registers by a Docker-pinned GeoSPARQL
endpoint, with every step from a DE-9IM matrix to a predicate proved by a
Lean development. The addresses come from
[`yuiseki/jp-admin-2026-09`](https://huggingface.co/datasets/yuiseki/jp-admin-2026-09),
which is where the registers were joined. Both are CC BY 4.0. Nothing in
either was written by a model.

1,632 facts, each said eight ways. 13,056 sentences, 185,832 characters,
172,304 tokens.

```
当別町は北海道に含まれる。
北海道は当別町を含む。
当別町は北海道にある。
当別町の所在する都道府県は北海道である。
北海道の市区町村のひとつが当別町である。
当別町は北海道の市区町村である。
住所：北海道石狩郡当別町
北海道石狩郡当別町
```

That is all of it. Nothing about places, nothing about borders, nothing about
the country.

## Why eight ways

Four runs came before this one, on the OpenStreetMap corpus and its own
wording alone.

The first trained on all three forms, so four fifths of the budget went on
N-Triples IRIs. Japanese scores fell and the control loss rose by 0.48.

The second trained on the Japanese administrative sentences and learnt what
they mostly say. Seven tenths of that text by character is "A and B do not
meet", because every composed disjointness is a sentence and most pairs do
not meet. Asked to continue 宗像市は, the model wrote 接していない。 for every
municipality in the country. It had learnt the corpus faithfully; the corpus
was mostly negation.

The third cut the corpus to one rung and worked, a little. On the cloze
probe, which is the question this is about, it went from 0.0% to 9.2% against
chance of 2.1%. Every answer had the right shape, 「〈県名〉に含まれる。」,
and the wrong prefecture: 宗像市 became 北海道. It had learnt the sentence and
a weak prior over prefecture names, not the mapping.

The fourth raised the exposure to a hundred passes and reached 23.3%, but the
control loss began to rise and every sentence still had the same shape. A
model shown one template a hundred times learns the template; the gradients
for 1,700 facts in one sentence pattern mostly cancel.

So this run varies the wording instead. A fact met in eight contexts is
remembered in far fewer passes than the same fact met eight times in one.
Four of the eight open with the child's name and a は, which is how the cloze
probe asks, so the question is not a shape the model has never seen. The
other four do not, which is what stops it learning the sentence instead of
the fact.

Two of the eight are addresses rather than sentences about geography, and
they are the reason the registers are worth using. An address register knows
that 篠栗町 is in 福岡県糟屋郡, which the municipality's own name leaves out:
770 of these 1,632 have a 郡, and a corpus built from names alone writes
福岡県篠栗町, which is how people write it rather than what it is.

## What this measures

Two ways of asking, reported apart.

cloze opens the corpus's own sentence: 松山市は . Whether the fact went in.

qa asks the question with three worked examples. Whether the fact can be
reached by being asked.

A rise in cloze with no rise in qa is the expected shape for continued
pretraining on declarative text, and it says instruction tuning is the next
step rather than that the corpus failed.

Two controls on top of that.

One municipality in ten is held out of the corpus entirely, chosen by the
sha256 of its own code. The eval half cannot be recalled and is not expected
to move. If it moves with the train half, the model has learnt to answer with
a plausible prefecture rather than the right one, which is exactly what the
third run did.

A Japanese Wikipedia corpus is never trained on and its loss is watched.
Training Japanese templates while losing Japanese prose is the specific risk
here, so the control is in the same language. REPLAY mixes a share of it back
into each batch; it was what stopped the control loss rising last time.

Where this starts, measured on 2026-09-26 before any training:

| model | form | half | ja | en |
|---|---|---|---|---|
| Qwen3-0.6B-Base | cloze | train | 3.0% | |
| Qwen3-0.6B-Base | cloze | eval | 5.1% | |
| Qwen3-0.6B-Base | qa | train | 5.5% | 11.0% |
| Qwen3-0.6B-Base | qa | eval | 5.7% | 15.4% |
| Qwen3.6-35B-A3B | qa | train | 74.5% | 77.0% |
| Qwen3.6-35B-A3B | qa | eval | 61.1% | 77.1% |
| llm-jp-3-440m | cloze | train | 31.0% | |
| llm-jp-3-440m | cloze | eval | 33.7% | |
| llm-jp-3-440m | qa | train | 90.5% | |
| llm-jp-3-440m | qa | eval | 91.9% | |

Chance is 2.1%. The number to watch is cloze on the train half: every
sentence in the corpus is the shape that question opens, so if the facts are
in the model at all, that is where they show. The two halves agree before
training, which is what makes the eval half usable as a control afterwards.

The llm-jp row is a smaller model, freely licensed under the same Apache 2.0,
that answers nine questions in ten before anything is done to it, and
completes three sentences in ten. It was rescored three ways before being
believed. Whatever a run on it is for, it is not teaching it these facts.

Sixty passes over 172,000 tokens is a few minutes of A100 time. The probes
cost more than the training does at this size. Well under a compute unit for
the whole notebook.
''')))

cells.append(md("## 1. The machine, and what to install"))

cells.append(code(*src(r'''
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install -U "transformers>=4.57" datasets pyarrow
''')))

cells.append(md(*src(r'''
## 2. Configuration

The base model rather than the instruction-tuned one. Continued pretraining
undoes instruction tuning before it teaches anything, so the order that makes
sense is to continue from the base and instruction-tune afterwards.

Qwen rather than gemma, mostly because `google/gemma-3-270m` is a gated
repository and a notebook that needs a licence acceptance and a token does
not run as it stands. On this corpus the tokenizers go the other way: it is
all Japanese now, where gemma gets 1.66 characters per token against Qwen's
1.39, so gemma would read the same text in a fifth fewer tokens. Worth
revisiting if the licence is accepted; not worth blocking the notebook on.

To use gemma anyway, set `MODEL` and add a cell with `huggingface_hub.login()`
before section 4.
''')))

cells.append(code(*src(r'''
DATASET   = "yuiseki/geo-triples-jp-gov"
REVISION  = None          # pin a commit sha here to fix what is measured
REGISTER  = "yuiseki/jp-admin-2026-09"   # where the addresses are spelled
MODEL     = "Qwen/Qwen3-0.6B-Base"

# The corpus is built from the probe's own fact set rather than by filtering
# the sentence table, which is why the filters below are unused here and kept
# only for a run that sets FROM_PROBE to False. The probe is one row per
# municipality with its prefecture: no predicate to select, no rung to
# exclude, and exactly the facts the evaluation will ask about.
FORMS     = "ja"
TOPIC     = "admin"
TRAIN_ONLY = True
PAIRS      = "abr-muni>abr-pref,abr-pref>abr-muni"
PREDICATES = "sfWithin,sfContains"

# Say each fact eight ways rather than louder. Two earlier runs raised the
# exposure instead and the score followed, 9.2% at thirty passes and 23.3% at
# a hundred, but the control loss began to rise and every sentence still had
# the same shape. A model shown one template a hundred times learns the
# template; the gradients for 1,632 facts in one sentence pattern mostly
# cancel.
#
# 1,632 facts is about 9,000 bits. Neither model here is short of room.
FROM_PROBE = True     # build the corpus from the probe's train split
PHRASINGS  = 8        # how many of the eight to use

# Whose tokenizer the exposure schedule counts names with, or None to write
# every fact exactly once.
#
# This is deliberately not MODEL. The schedule writes a fact three times when
# its subject is short, and "short" is a property of a tokenizer, not of a
# name: 篠栗町 is three tokens to Qwen and two to llm-jp. Letting the schedule
# follow MODEL would mean a run that changes the model silently changes the
# corpus too, and the comparison would carry two variables. Pinning it names
# which corpus is being reused; setting it to None reproduces run 1's.
SCHEDULE   = "Qwen/Qwen3-0.6B-Base"

# Where the memory goes is the logits, not the model: batch x block x the
# vocabulary, upcast to fp32 for the loss. That is 151,669 entries for Qwen
# and 99,574 for llm-jp, so a smaller tokenizer is cheaper here as well as
# shorter. The first step prints the peak. If it runs out, halve BATCH before
# touching anything else; gradient accumulation makes that free, because
# 8 x 2 and 4 x 4 see the same tokens. Small, because the corpus is small.
BLOCK       = 512
BATCH       = 4
GRAD_ACCUM  = 1
LR          = 1e-4        # this corpus is a hundred thousand tokens or two,
                          # not thirty million
# How many times each fact is read. Sixty was right for a model that did not
# have these facts; run 3 put the same schedule on one that did, and its
# held-out loss bottomed at step 300 of 3,117, about six epochs, then climbed
# back past where it started. A model with something to lose needs a different
# number from a model with nothing.
#
# Epochs rather than steps is also what makes a run with a different tokenizer
# comparable. The corpus is written as text and segmented by whichever
# tokenizer is being trained, so the same sentences are a different number of
# tokens: 172,304 under Qwen and 106,948 under llm-jp. Holding epochs holds how
# often each fact is read and lets the step count differ; holding steps would
# hold the number of updates and let the reading differ. Neither is free of a
# confound, and the first one is the one this is asking about. Report the step
# count as a result, not as a nuisance.
EPOCHS      = 60.0
EVAL_EVERY  = 100
EVAL_ITERS  = 20
EVAL_BATCH  = 1           # eval builds the logits; training with liger does not
SEED        = 7
PROBE_N     = 400         # per condition, per half. There are eight
                          # conditions and the probe runs twice, so asking
                          # all 1,632 of the train half would cost more than
                          # the training. The eval half is 175 and is asked
                          # whole.
CONTROL      = "20260901.ja"   # Wikipedia, never trained on
CONTROL_DOCS = 200
REPLAY       = 0.3        # the control rose by 0.13 at 0.2 and 1e-4
OUT         = "runs/geo"

# For a smoke run before spending an hour: a few documents and a few steps
# take the same path as the real thing. 0 means the whole corpus.
LIMIT     = 0
MAX_STEPS = 0
''')))

cells.append(md(*src(r'''
## 3. The pipeline

A copy of `src/corpus.py`, `src/train.py` and `src/hierarchy_probe.py` from
the repository this notebook came from, flattened into one cell so the
notebook needs no checkout. Three things in it are load-bearing and easy to
get wrong, so they are commented where they happen: the labels are not
shifted by the caller, the held-out tail needs room for a window to start in,
and the probe is sampled per level.
''')))

cells.append(code(*src(r'''
import collections, json, math, os, random, time

# How long each stage took, filled in as the notebook runs and printed whole
# at the end. A run that has to be read back from a pasted log is much easier
# to reason about when the timings are in one place: whether the probe or the
# training dominated, and what a step cost, decide what to change next.
TIMES = {}


def stage(name):
    """with stage("training"): ... records the wall clock under that name."""
    import contextlib

    @contextlib.contextmanager
    def timed():
        t0 = time.time()
        try:
            yield
        finally:
            TIMES[name] = time.time() - t0

    return timed()
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    get_cosine_schedule_with_warmup

DTYPE = np.uint32   # 151,669 vocabulary entries for Qwen, 262,144 for gemma;
                    # neither fits in 16 bits


def build_corpus(out, dataset, config, model, where=None, limit=0,
                 text_field="text", revision=None):
    """A published dataset -> one flat array of token ids on disk.

    Documents are separated by the end-of-text id. Without it a window that
    straddles two documents reads as one run-on document, and the model learns
    to continue from one fact into an unrelated one.

    The limit counts documents kept, not documents read: the cpt table is
    sorted by form, so every ja row sits after the ntriples ones and counting
    rows read would make a small limit return nothing.

    A where value may list alternatives separated by commas. The two
    directions of a containment are two predicates saying one fact, and one
    flag per value would read as a conjunction and keep nothing.
    """
    tok = AutoTokenizer.from_pretrained(model)
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit(f"{model} has no end-of-text token; there is nothing "
                         "to separate documents with")
    ds = load_dataset(dataset, config, split="train", revision=revision)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    total, kept, buf = 0, 0, []
    pending = []

    def flush(f):
        nonlocal total, buf
        if buf:
            np.asarray(buf, dtype=DTYPE).tofile(f)
            total += len(buf)
            buf = []

    with open(out, "wb") as f:
        # A value may list alternatives: "sfWithin,sfContains" keeps both.
        want = {k: set(v.split(",")) for k, v in (where or {}).items()}
        for row in ds:
            if any(str(row.get(k)) not in v for k, v in want.items()):
                continue
            text = row.get(text_field)
            if not text:
                continue
            pending.append(text)
            kept += 1
            if len(pending) >= 256:
                for ids in tok(pending, add_special_tokens=False)["input_ids"]:
                    buf.extend(ids + [eos])
                pending = []
                if len(buf) >= 1_000_000:
                    flush(f)
            if limit and kept >= limit:
                break
        if pending:
            for ids in tok(pending, add_special_tokens=False)["input_ids"]:
                buf.extend(ids + [eos])
        flush(f)
    print(f"{total:,} tokens from {kept:,} documents -> {out}")
    return total


def split_holdout(n, holdout, block):
    """Where training stops and the held-out tail begins.

    A tail rather than a random sample, because windows overlap and a randomly
    chosen evaluation window shares tokens with the training windows either
    side of it.

    The floor is block + 2, not block + 1: a tail of exactly block + 1 holds
    one window and no choice of where to start it, and a short corpus then
    dies after the model has loaded.
    """
    return max(0, n - max(block + 2, int(n * holdout)))


def batch(tokens, lo, hi, batch_size, block, rng, device):
    """One batch of windows.

    A window, not a pair. Transformers shifts labels itself, so the caller
    passes the same window as input_ids and as labels. Handing it a window
    already shifted by one asks for the token after next, and the loss sits at
    chance while the run completes normally.
    """
    top = hi - block - 1
    starts = rng.integers(lo, top, size=batch_size)
    x = np.stack([tokens[s:s + block] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, tokens, lo, hi, block, iters, device, seed, bs=1):
    rng = np.random.default_rng(seed)
    model.eval()
    total = 0.0
    for _ in range(iters):
        x = batch(tokens, lo, hi, bs, block, rng, device)
        total += float(model(input_ids=x, labels=x).loss)
    model.train()
    return total / iters
''')))

cells.append(code(*src(r'''
# Eight ways to say one containment. The first is the dataset's own wording
# and the second its converse; the rest are ordinary Japanese for the same
# fact, and the last two are addresses rather than sentences about geography.
#
# Four open with 「{child}は」, which is how the cloze probe asks, so the
# question is not a form the model has never seen. The other four do not,
# which is what stops the model learning the sentence instead of the fact.
#
# The addresses are the reason the registers are worth using. 篠栗町 is in
# 福岡県糟屋郡, and its own name does not say so: 770 of these 1,632 have a
# 郡, and writing 福岡県篠栗町 is how people write it rather than what it is.
PHRASE = [
    "{child}は{parent}に含まれる。",
    "{parent}は{child}を含む。",
    "{child}は{parent}にある。",
    "{child}の所在する都道府県は{parent}である。",
    "{parent}の市区町村のひとつが{child}である。",
    "{child}は{parent}の市区町村である。",
    "住所：{address}",
    "{address}",
]


def address(pref, county, name):
    """福岡県糟屋郡篠栗町, and 北海道札幌市中央区 with no county to place.

    The name of a designated city's ward already carries its city, so the
    three parts are all there is: nothing else goes between a prefecture and
    a municipality in a Japanese address.
    """
    return f"{pref}{county or ''}{name}"


def phrasings(fact, n):
    return [t.format(child=fact["child"], parent=fact["parent"],
                     address=fact["address"])
            for t in PHRASE[:n]]


# How many times a fact is written, by how many tokens its subject takes.
#
# Run 1 got 1.7% of the six-token names wrong and 60.3% of the two-token ones,
# falling monotonically in between. It is not capacity: 1,632 facts is about
# 9,000 bits. A short name gives the model one or two places to hang a fact
# on, and one of them is the 市 or 町 it shares with 800 others.
#
# Two other things were measured and are deliberately not used. The frequency
# of the name's rarest token in Japanese Wikipedia predicts the same failures
# slightly less well, 131 of the 188 against 153, and would cost a frequency
# table. The size of the answering prefecture predicts them independently and
# much more weakly, and evening it out means repeating whole prefectures: 4.6
# times the corpus rather than 1.6.
def repeats(name_tokens):
    if name_tokens <= 2:
        return 3
    if name_tokens == 3:
        return 2
    return 1


CANDIDATES = {"municipality-in-prefecture": 47}
SHOTS = 3

QUESTION = {
    ("municipality-in-prefecture", "en"): "Which prefecture is {child} in?",
    ("municipality-in-prefecture", "ja"): "{child}はどの都道府県にありますか。",
}

# Q and A on their own lines, and the last one left open. A 0.6B base model
# needs the format: with the examples merely separated by a blank line it
# produced Chinese trivia, or repeated the first example's question back.
#
# The open turn ends at the colon with no space after it. A trailing space is
# its own token and the first token of the answer then arrives without its
# leading one: 文京区 came back as 京区, and every answer was wrong for a
# reason that had nothing to do with the model.
TURN = "Q: {q}\nA: {a}"
OPEN = "Q: {q}\nA:"

# The other way to ask, and for a model continued on declarative sentences it
# is the fair one. The corpus says "南伊勢町は三重県に含まれる。" and the
# question above asks for the same fact in a shape the corpus never uses. A
# first run of this notebook lost 6 points of question-and-answer score while
# its held-out loss fell by 0.60, which is what losing the format rather than
# the fact looks like. Both are measured now and reported apart.
CLOZE = "{child}は"


def bare(name):
    """愛媛県 -> 愛媛, so an answer in either form counts.

    Not 区: 港区 and 港 are not interchangeable the way 愛媛県 and 愛媛 are,
    and dropping it would let 北区 match 北千住.
    """
    return name[:-1] if name and name[-1] in "県府都道" else name


def correct(answer, expected, mode="qa"):
    """Only the first line counts, and it must not be another question.

    The model is continuing a list of question-and-answer pairs, so after its
    answer it writes the next question, and that question names a place.
    """
    first = (answer or "").strip().split("\n")[0]
    if mode == "cloze":
        # Continuing a sentence, so the answer sits inside the clause rather
        # than alone. One line still: the next sentence names other places.
        return bool(first) and bare(expected) in first
    if first.startswith("Q:") or first.startswith("Q："):
        return False
    return bool(first) and bare(expected) in first


def few_shot(rows, lang, k=SHOTS):
    """Worked examples, spread across the level, no two with the same answer.

    A base model asked cold continues the question instead of answering it.
    The first three by id at the country level were Aruba and two provinces of
    Afghanistan, and the model then answered アフガニスタン to everything, so
    they are spread rather than taken from the front. They come out of the
    pool, so a demonstration is never also scored, and the same three are used
    before and after training.
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
                child=u["child_ja"] if lang == "ja" else u["child_en"]),
                a=u["parent_ja"] if lang == "ja" else u["parent_en"]) + "\n\n"
            for u in used)
        rest.extend(group)
    return shots, rest


def probe_rows(n, lang, seed=3, dataset=DATASET, revision=None,
               exclude_leaks=False, split=None):
    """n questions from each level, not n from the whole set.

    There are 66,541 place questions and 1,484 municipality ones, so a sample
    drawn over both is a sample of the first. Rows without a label in this
    language are dropped: most places carry no name:en, and asking which
    municipality None is in scores the model on a question nobody could
    answer.

    exclude_leaks drops the questions whose subject already names the answer.
    39% of the place questions do, and answering those is reading rather than
    recall.
    """
    rows = [dict(r) for r in load_dataset(dataset, "probe", split="train",
                                          revision=revision)]
    rows = [r for r in rows
            if r.get("child_" + lang) and r.get("parent_" + lang)]
    if exclude_leaks:
        rows = [r for r in rows if not r.get("answer_in_child_" + lang)]
    if split:
        rows = [r for r in rows if r.get("split") == split]
    rows.sort(key=lambda r: (r["level"], r["child_id"]))
    shots, rows = few_shot(rows, lang)
    by_level = collections.defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
    out = []
    for level in sorted(by_level):
        group = by_level[level]
        random.Random(seed).shuffle(group)
        out.extend(group[:n] if n else group)
    out.sort(key=lambda r: (r["level"], r["child_id"]))
    return shots, out


def ask(model, tok, shots, rows, lang, device, chat=False, mode="qa"):
    """The questions, as a completion.

    Asked as plain text unless chat is on. Qwen3-0.6B-Base ships a chat
    template although it is a base model, and wrapping a few-shot block in
    ChatML makes it answer by continuing the list of questions: every score
    was zero, and none of it was about the model's geography. Leave chat off
    for any base model, whether or not it carries a template.
    """
    templated = chat and getattr(tok, "chat_template", None)
    model.eval()
    out = []
    for i, row in enumerate(rows):
        child = row["child_ja"] if lang == "ja" else row["child_en"]
        if mode == "cloze":
            text = CLOZE.format(child=child)
        else:
            q = QUESTION[(row["level"], lang)].format(child=child)
            text = shots.get(row["level"], "") + OPEN.format(q=q)
        if templated:
            text = tok.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=False,
                                           add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        out.append((row, lang, tok.decode(gen[0][ids["input_ids"].shape[1]:],
                                          skip_special_tokens=True)))
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(rows)}", flush=True)
    model.train()
    return out


def score(label, answers, mode="qa"):
    per = collections.defaultdict(lambda: [0, 0])
    wrong = []
    for row, lang, got in answers:
        expected = row["parent_ja"] if lang == "ja" else row["parent_en"]
        ok = correct(got, expected, mode)
        per[row["level"]][1] += 1
        per[row["level"]][0] += ok
        if not ok and len(wrong) < 6:
            wrong.append((row, lang, got, expected))
    print(f"\n{label}")
    for level, (hit, n) in sorted(per.items()):
        print(f"  {level:18} {hit:5}/{n:<5} {hit / n:6.1%}   "
              f"(chance {1 / CANDIDATES[level]:.1%})")
    for row, lang, got, expected in wrong:
        child = row["child_ja"] if lang == "ja" else row["child_en"]
        print(f"    {child} -> {expected} | "
              f"{got.strip().split(chr(10))[0][:46]}")
    return {level: {"correct": h, "n": n, "accuracy": h / n,
                    "chance": 1 / CANDIDATES[level]}
            for level, (h, n) in per.items()}
''')))

cells.append(md("## 4. The corpus, and the control it is measured against"))

cells.append(code(*src(r'''
os.makedirs("data", exist_ok=True)


def build_from_probe(out, dataset, model, n_phrasings, revision=None,
                     register=REGISTER, schedule=SCHEDULE):
    """Every fact on the train side of the split, said n ways.

    The probe subset is the fact set, one row per child with its parent, so
    it is a cleaner source for this than filtering the sentence table: no
    predicate to select, no rung to exclude, and exactly the facts the probe
    will ask about. The eval side is not written, so the held-out
    municipalities stay held out.

    The register is joined on the local government code only to spell the
    address. It adds no facts, and it cannot add a municipality the probe
    does not have, so the split stays the probe's split. A code it does not
    have stops the build: a silently smaller corpus would leave the probe
    still asking about the missing facts, and that reads afterwards as a
    model that failed to learn them.
    """
    tok = AutoTokenizer.from_pretrained(model)
    eos = tok.eos_token_id
    rows = [r for r in load_dataset(dataset, "probe", split="train",
                                    revision=revision)
            if r["split"] == "train" and r["child_ja"] and r["parent_ja"]]
    reg = {m["lg_code"]: m for m in load_dataset(
        register, "municipalities", split="train").remove_columns(["geometry"])}
    facts = []
    for r in rows:
        m = reg.get(r["child_id"].rsplit("-", 1)[-1])
        if m is None:
            raise SystemExit(f"{r['child_id']} is not in {register}; the two "
                             f"are not the same vintage")
        facts.append({"child": r["child_ja"], "parent": r["parent_ja"],
                      "address": address(m["pref"], m["county"], m["name"])})
    # The schedule counts with its own tokenizer, which is usually not the one
    # being trained. See SCHEDULE in the configuration cell.
    counter = (AutoTokenizer.from_pretrained(schedule) if schedule else None)
    texts = []
    for f in facts:
        said = phrasings(f, n_phrasings)
        n = (repeats(len(counter(f["child"],
                                 add_special_tokens=False)["input_ids"]))
             if counter else 1)
        texts.extend(said * n)
    # Sorted, then shuffled with a fixed seed. Sorting first makes the order
    # independent of how the rows arrived; shuffling stops the eight
    # phrasings of one fact from always landing in the same window.
    texts.sort()
    random.Random(17).shuffle(texts)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    total, buf = 0, []
    with open(out, "wb") as f:
        for i in range(0, len(texts), 256):
            for ids in tok(texts[i:i + 256],
                           add_special_tokens=False)["input_ids"]:
                buf.extend(ids + [eos])
            if len(buf) >= 1_000_000:
                np.asarray(buf, dtype=DTYPE).tofile(f)
                total += len(buf)
                buf = []
        if buf:
            np.asarray(buf, dtype=DTYPE).tofile(f)
            total += len(buf)
    print(f"{total:,} tokens from {len(facts):,} facts x {n_phrasings} "
          f"phrasings, said {len(texts) / (len(facts) * n_phrasings):.2f} "
          f"times each on average -> {out}")
    for line in texts[:3]:
        print(f"    {line}")
    return total


where = {}
if FORMS:
    where["form"] = FORMS
if TOPIC:
    where["topic"] = TOPIC
if PAIRS:
    where["pair"] = PAIRS
if PREDICATES:
    where["predicate"] = PREDICATES
if TRAIN_ONLY:
    # The held-out places, dropped here rather than filtered later, so that
    # nothing downstream has to remember to.
    where["holdout"] = "False"
with stage("corpus"):
    if FROM_PROBE:
        n_corpus = build_from_probe("data/geo.bin", DATASET, MODEL, PHRASINGS,
                                    revision=REVISION, register=REGISTER)
    else:
        n_corpus = build_corpus("data/geo.bin", DATASET, "cpt", MODEL,
                                where=where or None, limit=LIMIT,
                                revision=REVISION)
    n_control = build_corpus("data/control.bin", "yuiseki/wikipedia-geotagged",
                             CONTROL, MODEL, limit=CONTROL_DOCS)
''')))

cells.append(md(*src(r'''
## 5. Before

Two languages, three levels, sampled per level. Takes a few minutes: the
questions are answered one at a time so that the decode is the same before and
after.
''')))

cells.append(code(*src(r'''
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)

# Which conditions to ask under. Every combination is 8 runs of PROBE_N
# questions on each side of training, which is most of the probe's cost, so
# the defaults ask what this run can answer: Japanese, both halves of the
# split, with and without the questions that name their own answer.
#
# The eval half is expected to sit still. It is the control that says a rise
# in the train half is the facts going in rather than the model learning to
# answer with a plausible municipality.
CONDITIONS = [(mode, leaks, split)
              for mode in ("cloze", "qa")
              for leaks in (True, False)
              for split in ("train", "eval")]


def run_probe(model, tok, label):
    out = {}
    for mode, leaks, split in CONDITIONS:
        shots, rows = probe_rows(PROBE_N, "ja", revision=REVISION,
                                 exclude_leaks=not leaks, split=split)
        if not rows:
            continue
        tag = f"{mode} {split}" + ("" if leaks else " no-leak")
        print(f"{len(rows)} questions, {tag}")
        out[tag] = score(f"{label} [{tag}]",
                         ask(model, tok, shots, rows, "ja", device, mode=mode),
                         mode)
    return out


with stage("probe before"):
    before = run_probe(model, tok, f"{MODEL} before")
''')))

cells.append(md("## 6. The run"))

cells.append(code(*src(r'''
torch.manual_seed(SEED)
tokens = np.memmap("data/geo.bin", dtype=DTYPE, mode="r")
control = np.memmap("data/control.bin", dtype=DTYPE, mode="r")
cut = split_holdout(len(tokens), 0.005, BLOCK)

# With replay on, half the control corpus is trained on and the other half is
# not. A corpus that is mixed into training is no longer a control, and
# measuring forgetting against text the model has just seen would report that
# there is none.
replay_cut = len(control) // 2 if REPLAY else 0
print(f"replay {REPLAY:.0%}"
      + (f", drawn from the first {replay_cut:,} control tokens" if REPLAY
         else "; the control is never trained on"))

per_step = BATCH * GRAD_ACCUM * BLOCK
steps = MAX_STEPS or max(1, int(EPOCHS * cut / per_step))
print(f"{len(tokens):,} tokens, training on {cut:,}, "
      f"holding out {len(tokens) - cut:,}")
print(f"{steps:,} steps of {per_step:,} tokens "
      f"({BATCH} x {GRAD_ACCUM} x {BLOCK})")

model.gradient_checkpointing_enable()
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                        betas=(0.9, 0.95))
sched = get_cosine_schedule_with_warmup(opt, min(100, steps // 10 + 1), steps)

os.makedirs(OUT, exist_ok=True)
log = []
rng = np.random.default_rng(SEED)
began = time.time()


def checkpoint(step):
    row = {"step": step,
           "held_out": evaluate(model, tokens, cut, len(tokens), BLOCK,
                                EVAL_ITERS, device, SEED, EVAL_BATCH),
           "control": evaluate(model, control, replay_cut, len(control),
                               BLOCK, EVAL_ITERS, device, SEED, EVAL_BATCH),
           "elapsed_s": round(time.time() - began, 1)}
    log.append(row)
    print(f"  step {step:>6}  held_out {row['held_out']:.4f}  "
          f"control {row['control']:.4f}", flush=True)
    return row


first = checkpoint(0)
chance = math.log(model.config.vocab_size)
if first["held_out"] > chance - 1.0:
    # A pretrained model reading text sits far below a uniform draw. Sitting
    # near it means the model is being asked the wrong question, which is what
    # a shifting mistake looks like from the outside.
    raise SystemExit(
        f"the starting loss is {first['held_out']:.2f} against {chance:.2f} "
        "for a uniform draw. Something is wrong before any training has "
        "happened: the wrong tokenizer for this corpus, or labels shifted "
        "twice. Fix it rather than watching this number come down.")

for step in range(1, steps + 1):
    opt.zero_grad(set_to_none=True)
    for _ in range(GRAD_ACCUM):
        if REPLAY and rng.random() < REPLAY:
            x = batch(control, 0, replay_cut, BATCH, BLOCK, rng, device)
        else:
            x = batch(tokens, 0, cut, BATCH, BLOCK, rng, device)
        (model(input_ids=x, labels=x).loss / GRAD_ACCUM).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step == 1 and device == "cuda":
        print(f"  peak VRAM {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
    if step % EVAL_EVERY == 0 or step == steps:
        checkpoint(step)

seen = steps * per_step
TIMES["training"] = time.time() - began
print(f"{seen:,} tokens in {TIMES['training']:.0f}s")
with stage("saving"):
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
''')))

cells.append(md("## 7. After"))

cells.append(code(*src(r'''
with stage("probe after"):
    after = run_probe(model, tok, f"{MODEL} after")

print("\n=== recall, on places the corpus states outright")
for tag in sorted(before):
    for level in sorted(before[tag]):
        b, a = before[tag][level], after[tag][level]
        print(f"  [{tag:10}] {level:28} {b['accuracy']:6.1%} -> "
              f"{a['accuracy']:6.1%}   on {b['n']}  "
              f"(chance {b['chance']:.2%})")

rise = log[-1]["control"] - log[0]["control"]
drop = log[0]["held_out"] - log[-1]["held_out"]
print(f"\nheld-out loss fell by {drop:.4f}")
print(f"control loss {'rose' if rise > 0 else 'fell'} by {abs(rise):.4f}")
if rise > 0.1:
    print("  the control corpus got worse. This is what catastrophic "
          "forgetting looks like before it reaches a benchmark.")
    if not REPLAY:
        print("  set REPLAY = 0.2 and run again: mixing general text back in "
              "is the ordinary remedy, and it was left off so that this run "
              "measured one thing.")

json.dump({"before": before, "after": after, "log": log},
          open(os.path.join(OUT, "result.json"), "w"), ensure_ascii=False, indent=2)
print(f"\nwrote {OUT}/result.json")
''')))

cells.append(md(*src(r'''
## 8. The run in one block

Everything needed to compare this run with another, in a form that survives
being pasted into a chat window: what was configured, what it scored, what it
cost, and how fast it went. Read the timings before deciding what to change.
A run where the probe costs more than the training is a run whose PROBE_N is
the wrong size, and a step time that drifts upward is a machine throttling
rather than a model learning.
''')))

cells.append(code(*src(r'''
def hms(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


print("=" * 64)
print(f"{'model':16} {MODEL}")
print(f"{'dataset':16} {DATASET} @ {REVISION or 'main'}")
print(f"{'schedule':16} {SCHEDULE or 'off, every fact written once'}")
print(f"{'corpus':16} {len(tokens):,} tokens, {PHRASINGS} phrasings, "
      f"{EPOCHS:g} epochs")
print(f"{'optimiser':16} lr {LR:g}, block {BLOCK}, batch {BATCH} x {GRAD_ACCUM}"
      f", replay {REPLAY:.0%}")
print(f"{'steps':16} {steps:,} of {per_step:,} tokens, {seen:,} seen")
if device == "cuda":
    print(f"{'device':16} {torch.cuda.get_device_name(0)}, peak "
          f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
else:
    print(f"{'device':16} {device}")

print("\n--- scores, before -> after")
print(f"  {'condition':22} {'before':>8} {'after':>8} {'diff':>8}")
for tag in sorted(before):
    for level in sorted(before[tag]):
        b, a = before[tag][level], after[tag][level]
        print(f"  {tag:22} {b['accuracy']:8.1%} {a['accuracy']:8.1%} "
              f"{a['accuracy'] - b['accuracy']:+8.1%}  on {b['n']}")

print("\n--- loss")
print(f"  {'held-out':22} {log[0]['held_out']:8.4f} {log[-1]['held_out']:8.4f} "
      f"{log[-1]['held_out'] - log[0]['held_out']:+8.4f}")
print(f"  {'control':22} {log[0]['control']:8.4f} {log[-1]['control']:8.4f} "
      f"{log[-1]['control'] - log[0]['control']:+8.4f}")
best = min(log, key=lambda r: r["held_out"])
print(f"  held-out was lowest at step {best['step']:,}: {best['held_out']:.4f}")

print("\n--- time")
for name, seconds in TIMES.items():
    print(f"  {name:22} {hms(seconds):>10}")
print(f"  {'total':22} {hms(sum(TIMES.values())):>10}")

# Per-step cost from the checkpoints rather than from the total, so that the
# evaluations and the saving do not get charged to the training loop.
marks = [(r["step"], r["elapsed_s"]) for r in log]
if len(marks) > 1:
    per = [(s2 - s1, t2 - t1) for (s1, t1), (s2, t2) in zip(marks, marks[1:])
           if s2 > s1]
    rates = [t / n for n, t in per]
    print(f"  {'per step':22} {sum(rates) / len(rates) * 1000:>7.0f} ms"
          f"   (first block {rates[0] * 1000:.0f}, last {rates[-1] * 1000:.0f})")
    print(f"  {'tokens per second':22} {per_step / (sum(rates) / len(rates)):>10,.0f}")
print("=" * 64)
''')))


def rebind(name, value):
    """Rewrite one assignment in the configuration cell.

    The cell bodies are raw strings full of {child} and {parent}, so they
    cannot be run through str.format, and the two settings that vary between
    runs are not worth a templating language. This edits the built cell and
    fails loudly if the line it is looking for has moved.
    """
    for cell in cells:
        for i, line in enumerate(cell["source"]):
            if line.startswith(name):
                # A line whose comment runs on to the next one cannot be
                # replaced without orphaning the rest of it. This happened
                # once: EPOCHS lost its first comment line and kept four
                # dangling continuations, which Python accepts and nobody can
                # read. Put the comment above the assignment instead.
                rest = cell["source"][i + 1] if i + 1 < len(cell["source"]) else ""
                if rest.strip().startswith("#") and rest.startswith(" "):
                    raise SystemExit(
                        f"{name.strip()} carries a comment that continues on "
                        "the next line; move it above the assignment")
                keep = line[len(line.rstrip("\n")):]
                cell["source"][i] = f"{name}= {value!r}{keep}"
                return
    raise SystemExit(f"no {name.strip()} assignment to rebind")


# What this particular run is for, inserted after the introduction. Keyed on
# the model, because the reason to run a notebook is rarely the same twice and
# a notebook that does not say why it exists is hard to read a month later.
RUN_NOTES = {
    ("llm-jp/llm-jp-3-440m", 6.0): r"""
## Why this run exists

Run 3 is this notebook with `EPOCHS` at 60, and it overshot. Everything else
is the same: the same corpus, every fact written once, the same learning rate,
the same replay. One number changed, and it was chosen from run 3's own
measurement rather than guessed.

Run 3 went like this.

    cloze train no-leak    26.0% -> 97.5%
    cloze eval  no-leak    29.1% -> 61.4%
    qa    train no-leak    91.5% -> 19.8%
    qa    eval  no-leak    90.5% -> 11.4%
    control loss          2.8477 -> 3.9062
    held-out loss         lowest 1.3916 at step 300 of 3,117, ended 2.5815

Two things in that are worth having and one is not.

Worth having: the held-out half rose by 32 points although none of those place
names is written anywhere in the corpus. This model knew the facts before the
run, answering 91% when asked as a question, and could not state them in the
corpus's sentence form. It learnt the form, and a form carries to every fact
already held. That is what continued pretraining did here, and it is not what
it did to Qwen, whose held-out half moved 1.3 points.

Not worth having: qa fell by 71 points and the control loss rose by 1.0585,
against 0.1556 in run 1. The facts are still there, since the same held-out
places score 11.4% asked and 61.4% completed, so what closed is a way in
rather than the knowledge. It is few-shot pattern following, which this model
has from pretraining alone and which 3,117 steps of one sentence shape
flattened. Base model, note: the instruction-tuned llm-jp checkpoints are
separate repositories and neither was used.

The held-out loss says where to stop: step 300, about six epochs.

## What to watch

Whether the 61.4% survives. If the format transfer needed sixty epochs then
this run will not show it, and the trade was real rather than an overshoot.

Whether qa comes back. If a shorter run keeps both, the recipe was simply too
long for a model that already knew the answers, and nothing more subtle is
going on. If qa still collapses at six epochs, the next run lowers the
learning rate, and that is the point at which the trade looks structural.

The control loss, which at 0.1 is ordinary and at 1.0 is not.
""",
    "llm-jp/llm-jp-3-440m": r"""
## Why this run exists

Not to teach this model these facts. It already answers 90.5% of the
questions on the train half and 91.9% on the eval half, which is four and a
half times what sixty epochs of continued pretraining got Qwen3-0.6B-Base to
on the half it had never read. Teaching it would be measuring nothing.

It exists to settle a question left over from those runs, and it can only be
settled inside one model.

Qwen failed on short names. Names of two Qwen tokens came back wrong 60.3% of
the time against 1.7% for names of six, and writing the short ones three times
as often did not move them: 39.7% to 36.2% over 58 names, four fixed and six
broken. Two readings survive that. Either a short name is hard because the
tokenizer gave it too little to hang a fact on, in which case a tokenizer
built for Japanese would not have the problem, or a short name is hard for
some reason of its own that no tokenizer design reaches.

Comparing the two models does not separate these. llm-jp was pretrained on
Japanese and Qwen was not, so any difference between them carries both causes
at once. The question has to be asked within llm-jp: after this run, band the
1,632 facts by how many tokens **llm-jp's own tokenizer** gives each name, and
look at the short bands. If one- and two-token names fail here the way they
failed on Qwen, the difficulty is in the names. If they do not, it is in the
tokenizer, and that is a finding about how to build one.

The two tokenizers do not agree about which names are short, which is the
whole reason this is worth doing. llm-jp gives a prefecture name one token
where Qwen gives 3.19, and a municipality name 2.47 against 3.66. Of the 58
names Qwen cuts into two tokens, llm-jp cuts 12 into one and 7 into three.

## What is held fixed

`SCHEDULE` is off, so every fact is written exactly once. This is deliberate
and it is not the corpus run 2 used.

Run 2 wrote short names more often, and the result was that the untouched
bands got worse although their own exposure had not changed: what fell was
their share of the corpus. Leaving the schedule on here would mean that a
short name failing could always be answered with "it was not written enough
times", which is the argument run 1 and run 2 already spent themselves on.
Equal exposure closes it. If a short name fails when every fact was written
the same number of times, the exposure reading is finished.

`EPOCHS` stays at 60. The corpus is the same sentences as run 1, and llm-jp
segments them into fewer tokens than Qwen does, so this run is shorter in
steps while reading each fact exactly as often. Report the step count as a
property of the tokenizer, not as something to correct for.

## What to watch, and what it would cost

cloze on the train half is the number this run is about: it starts at 31.0%
against a qa score of 90.5%, so this model knows the facts and cannot state
them in the corpus's own sentence form. That gap is the opposite of the one
the trained Qwen ended with, 93.5% cloze against 75.2% qa.

qa is what there is to lose. Qwen had nothing to forget here and this model
has a great deal, so a rise in cloze bought with a fall in qa is a real
result and not a rounding error. Watch both, and watch the control loss.
""",
}

# Keyed on the model and, where a run turns on one setting, on that setting
# too: two runs of the same model at different epochs are different
# experiments and should not carry the same note.
note = RUN_NOTES.get((MODEL_ID, EPOCHS_N)) or RUN_NOTES.get(MODEL_ID)
if note:
    cells.insert(1, md(*src(note)))

rebind("MODEL     ", MODEL_ID)
rebind("SCHEDULE   ", SCHEDULE_ID)
rebind("EPOCHS      ", EPOCHS_N)

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

# One notebook per model, named after the model it produced, so that a run
# can be read back long after it happened. The generator overwrites it, which
# is only safe while the notebook has not been run and edited by hand: what
# is checked in here is the file that was uploaded to Colab, not a copy
# brought back from it.
path = ARGS.out or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks", "qwen3-0.6b-base-jp-gov-v0.1.ipynb")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(path, len(cells), "cells")
