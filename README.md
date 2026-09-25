# YuisekinAI-CPT

Continued pretraining on geographic text, and the measurements that say whether
it worked.

The question is narrow. A small model does not know which prefecture a Japanese
municipality is in, and a fine-tune cannot teach it: fine-tuning teaches the
shape of an answer, and a model that has learnt the shape without the geography
writes `Hiroshima City, Tokyo, Japan`, which is well formed and returns nothing
from a map query. Does feeding it geographic text fix that, and what does it
break on the way?

## The floor and the ceiling, measured before any training

250 Japanese municipalities, one prefecture each, drawn from
[`yuiseki/wikidata-gazetteer`](https://huggingface.co/datasets/yuiseki/wikidata-gazetteer).
Chance is 1 in 47.

| model | form | overall | <20k | 20k-100k | 100k+ |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | chat | 83.6% | 68.4% | 88.2% | 98.0% |
| gemma-3-270m-it | qa | 6.4% | 2.5% | 5.0% | 16.3% |
| gemma-3-270m-it | chat | 2.0% | 0.0% | 0.0% | 10.2% |

The gap is real and it is knowledge, not formatting: under the chat template
the 270M model answers with the city's own name. The 35B knows the same facts
and loses a third of the small towns, which says the knowledge scales with
exposure rather than with capacity. 1,134 facts fit in 270M parameters many
times over.

## The corpora

The corpus being learnt is the `cpt` subset of
[`yuiseki/geo-triples-tokyo23`](https://huggingface.co/datasets/yuiseki/geo-triples-tokyo23),
which states 126,208 true spatial facts three times over: as an N-Triples
line, as an English sentence and as a Japanese one. Nothing in it was written
by a model, and every statement is traceable to a DE-9IM matrix computed from
frozen geometry.

| `form` | rows | characters |
|---|---:|---:|
| `ntriples` | 126,208 | 37,696,172 |
| `en` | 89,500 | 6,567,573 |
| `ja` | 89,500 | 4,030,260 |

The three sit in one table with a `form` column, so `--where form=ja` picks
one and the weighting is a choice made at corpus time rather than at build
time.

The control is `20260901.en` of
[`yuiseki/wikipedia-geotagged`](https://huggingface.co/datasets/yuiseki/wikipedia-geotagged),
4,331,110,851 characters, never trained on. It is there to make forgetting
visible at every checkpoint rather than at the end.

| subset | characters | tokens (gemma) | tokens (OLMo) |
|---|---|---|---|
| `20260901.ja` | 435,046,691 | 311 M | 473 M |
| `20260901.en` | 4,331,110,851 | 1,062 M | 1,017 M |

Japanese Wikipedia is measured here because it was the first corpus tried and
the tokenizer comparison below still rests on it.

## The two probes, which measure opposite things

| | asks about | in the corpus | measures |
|---|---|---|---|
| `src/hierarchy_probe.py` | 2,913 states and wards | yes, every one | recall |
| `src/probe.py` | 1,134 Japanese municipalities | 1 of them | generalisation |

The first is the frozen `probe` subset of the same dataset, pinned by the same
digest, so a score names a dataset revision. Its answers are all stated in the
training corpus by construction, which is why it is labelled recall and not
anything stronger.

The second is the original question and the corpus says almost nothing about
it. A rise in the first with none in the second is the expected result rather
than a disappointment, and both are run before and after so that one cannot be
reported as the other.

| model | level | en | ja |
|---|---|---:|---:|
| Qwen3.6-35B-A3B | `place-in-ward` | 37.5% | 45.8% |
| Qwen3.6-35B-A3B | `state-in-country` | 86.7% | 63.3% |
| Qwen3-0.6B-Base | `place-in-ward` | 10.0% | 22.5% |
| Qwen3-0.6B-Base | `state-in-country` | 46.7% | 7.5% |

120 questions per level with three worked examples in front of each, asked as
a completion. Chance is 4.3% and 0.4%. `ward-in-state` is left out of the
table because both models answer all 16 of its questions and every answer is
東京都.

`place-in-ward` is the level with room in it: 37.5% at 35B. The language gap
reverses between the two levels, which is the `name:en` coverage of
OpenStreetMap seen from the other side.

The protocol took three attempts and none of the failures were about
geography. Asked cold, a base model continues the question. Wrapped in the
chat template that `Qwen3-0.6B-Base` ships with although it is a base model,
it does the same. And a prompt ending in `A: ` with a trailing space loses the
first token of the answer, so 文京区 comes back as 京区 and every row is
wrong. All three read as 0%.

## What is here

    tmp/main.ipynb    the whole experiment as one Colab notebook
    src/corpus.py     a published dataset -> one flat array of token ids
    src/train.py      continued pretraining, with a control corpus
    src/probe.py      which prefecture is this municipality in
    src/hierarchy_probe.py       which parent does this place have
    scripts/build_probe_set.py   rebuilds the answer key from the gazetteer
    scripts/dry_run.sh           every stage, small enough to finish here

## On Colab

`tmp/main.ipynb` runs top to bottom on an A100 and needs nothing from disk:
the corpus, the control and the probe all come from the Hub. It is a flattened
copy of `src/corpus.py`, `src/train.py` and `src/hierarchy_probe.py`, with
`LIMIT` and `MAX_STEPS` for a smoke run first.

It trains on the Japanese sentences of
[`yuiseki/geo-triples-japan`](https://huggingface.co/datasets/yuiseki/geo-triples-japan),
1,848,010 of them, and the defaults come from a run that failed. Training on
all three forms put four fifths of the budget into N-Triples IRIs: Japanese
`place-in-ward` fell from 22.5% to 11.7%, the control loss rose by 0.48, and
every answer collapsed onto one ward, with invented wards like 西武区 among
them. So the notebook takes one form and a third of the learning rate, and
`REPLAY` is there for the second run if the control still rises.

It continues `Qwen/Qwen3-0.6B-Base` rather than gemma. `google/gemma-3-270m`
is a gated repository, and a notebook that needs a licence acceptance and a
token does not run as it stands. The tokenizers also disagree about this
corpus in Qwen's favour, because four fifths of it is N-Triples:

| | N-Triples | ja | en |
|---|---:|---:|---:|
| gemma-3-270m | 2.23 | 1.66 | 4.71 |
| Qwen3-0.6B | 2.71 | 1.39 | 4.25 |

characters per token. 31 million tokens rather than 39 for the same text,
against a model twice the size: about half an hour of A100 time for one pass,
3 to 4 compute units. The generalisation probe is the one thing it
cannot do by itself, because its answer key is `data/jp_municipalities.json`
here rather than on the Hub; the last cell takes it as an upload.

## Run it here before renting anything

    bash scripts/dry_run.sh

4,000 statements, five steps, on whatever hardware is present. It teaches the
model nothing; it shows that each stage reads its input and writes its output,
and it costs no compute units.

## Two things the dry run found

**The loss sat at chance.** 11.94 against `ln(262144) = 12.48`, on a pretrained
model reading its own language, because `model(input_ids=x, labels=y)` was
given a window already shifted by one and Transformers shifts labels itself.
The run completed and the loss came down from 11.94 to 8.11, which looks like
learning. `train.py` now refuses to start when the first loss is near chance,
because reporting it was not enough.

**Both corpora improved together.** Held-out Japanese fell 7.44 to 5.29 and the
English control fell 6.40 to 4.76, from 1,280 tokens of training. Nothing
learns geography from 1,280 tokens. What moved is the model's register: an
instruction-tuned model asked to continue raw text is out of its own
distribution, and the first thing continued pretraining does is undo the
instruction tuning.

That is the earlier failure in slow motion. A previous attempt ran 816 steps of
full-weight CPT on this same model and scored 0.4% on a task it had scored
94.2% on, because the format it had been tuned to produce was gone, while the
loss on the training corpus looked healthy throughout.

Which suggests continuing from `google/gemma-3-270m` rather than
`google/gemma-3-270m-it`, and instruction-tuning afterwards rather than
through. The control corpus is in the loop so that this is visible at every
checkpoint rather than at the end.

## What the memory goes on

Not the model. gemma-3-270m has 262,144 vocabulary entries against a hidden
size of 640, so the embedding is 62.6% of its parameters and the loss is
computed over logits that are batch x block x 262,144, upcast to fp32.

    python3 src/budget.py --table

The model and its optimizer cost 2.0 GB and never change; everything else is
1.5 MB per token of batch x block. Measured against an RTX 3060 the estimate
is good to about 1%: 2.75 GB predicted and 2.53 measured at 1 x 512, 3.51 and
3.54 at 1 x 1024. Gradient accumulation is therefore free, because 1 x 2048
and 2 x 1024 cost the same.

`--liger` fuses the final projection into the loss so the logits are never
built. It works, and on this hardware it is a bad trade:

| | tokens/s | peak at 1 x 512 |
|---|---|---|
| plain | 3,026 | 3.03 GB |
| RoPE, RMSNorm, GeGLU only | 3,437 | 3.03 GB |
| fused cross entropy only | 420 | 2.54 GB |
| everything | 425 | 2.53 GB |

The loss is bit-identical, and the memory saving is total: 8 x 4096 costs
2.75 GB where the plain path would need over 100. But the fused loss alone is
what costs the speed, sevenfold, on an RTX 3060. A 640-wide hidden state
against a 262,144-wide vocabulary makes a very tall, thin matrix to chunk, and
Liger's published benchmarks are A100 and H100. Whether the trade reverses
there is a question for an A100, which is why the flag exists and is off.

One more thing to know: the fusion applies in training mode only. The same
1 x 1024 forward costs 1.34 GB in `train()` with no logits built and 3.55 GB
in `eval()` with them built, so evaluation has its own `--eval-batch-size`,
defaulting to 1. Without that a training batch the fusion makes affordable
fails at the first checkpoint.

## Compute

Google AI Pro includes 200 Colab compute units a month; an A100 40GB draws
about 5.37 an hour, so roughly 37 hours. At 6ND and 40-80 TFLOPS effective:

| model | corpus | hours |
|---|---|---|
| 270M | ja, 311 M tokens | 1.7-3.5 |
| 270M | ja+en, 1,373 M tokens | 7.7-15 |
| 1.5B | ja, 473 M tokens | 10-15 |

Measured here for comparison: 3,399 tokens/s on one RTX 3060 sharing the card
with a resident llama-server, which is 25 hours for one pass over the Japanese
subset. This machine is for the dry run and for short ablations.

Background execution is an Ultra benefit, so a Pro run holds a browser tab
open and the idle threshold is unpublished. Two or three hours is a reasonable
bet; fifteen is not.

## On OSAID

The Open Source AI Definition wants training data a skilled person could
recreate. A model continued from a base whose corpus is undisclosed inherits
that, so these runs are instruments for answering a question, not steps toward
the model itself. Worth restating whenever a result here looks like a product.
