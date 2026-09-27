# YuisekinAI-CPT

Continued pretraining on geographic text, and the measurements that say whether
it worked.

The question is narrow. A small model does not know which prefecture a Japanese
municipality is in, and a fine-tune cannot teach it: fine-tuning teaches the
shape of an answer, and a model that has learnt the shape without the geography
writes `Hiroshima City, Tokyo, Japan`, which is well formed and returns nothing
from a map query. Does feeding it geographic text fix that, and what does it
break on the way?

It does, for the facts it is fed. And then, on a second base model, the same
corpus showed that most of what continued pretraining was doing is not what
the question assumed.

## What came out of it

[`yuiseki/qwen3-0.6b-jp-gov-v0.1`](https://huggingface.co/yuiseki/qwen3-0.6b-jp-gov-v0.1):
Qwen3-0.6B-Base continued on 1,632 facts about where Japan's municipalities
are, each said eight ways. Japanese, greedy decoding, the same prompt to every
model, chance 2.1%.

| asked | Qwen3-0.6B-Base | after | Qwen3.6-35B-A3B |
|---|---|---|---|
| 「当別町は」, taught half | 2.2% | 86.5% | 69.8% |
| 「当別町は」, held-out half | 3.8% | 22.8% | 72.0% |
| 「当別町が属する都道府県は」 | 2.0% | 88.5% | |
| 「Q: 当別町は何県にありますか。A:」 | 5.0% | 49.5% | |
| 「北海道の市区町村のひとつが」 | 14.9% | 93.6% | |

It holds the table it was given better than a model fifty-eight times its
size, and it does not know the municipalities it was not given: 22.8% on the
held-out half against the 35B's 72.0%. That is the honest summary, and the
model card says it in the same words.

The 35B answering 69.8% of one half and 72.0% of the other is the control the
split needed. It was shown neither, so the halves are equally hard, and the
63.7-point gap in the small model is training rather than difficulty.

[`yuiseki/llm-jp-3-440m-jp-gov-v0.2`](https://huggingface.co/yuiseki/llm-jp-3-440m-jp-gov-v0.2):
the same corpus on a smaller Japanese base, for six passes and 41 seconds.

| asked | llm-jp-3-440m | after |
|---|---|---|
| 「当別町は」, taught half | 26.0% | 97.2% |
| 「当別町は」, held-out half | 29.1% | 71.5% |
| 「当別町が属する都道府県は」 | 72.5% | 96.5% |
| 「Q: 当別町は何県にありますか。A:」, taught | 91.5% | 98.0% |
| 「Q: 当別町は何県にありますか。A:」, held-out | 90.5% | 83.5% |
| 「北海道の市区町村のひとつが」 | 95.7% | 100.0% |

Read the second row against the Qwen table's second row. The held-out half
rises 42 points here and 19 there, on place names written nowhere in the
corpus. This model already knew where they were, answering 91% when asked as a
question; what it lacked was a reading of 「Xは」 that treats it as that
question. Continued pretraining supplied the reading, and a reading carries to
every fact already held.

Which is the finding, and it was not the question anyone set out with:
continued pretraining on declarative text teaches a way in more than it teaches
facts, and how much of each depends on what the model already has. Sweeping the
base model both ways before training says which it will be. For llm-jp on this
corpus, 116 of 1,525 facts were absent in both forms; the other 93% of the work
was phrasing.

That ratio sets the epoch count. Sixty passes, which was right for Qwen, took
llm-jp's question form from 91.5% to 19.8%: it stopped applying its three
worked examples and started repeating the nearest one, while emitting a clean
single prefecture every time and so looking fine. Six passes cost nothing and
gave more. On the 116 genuinely new facts, though, sixty beat six, p = 0.033.
Facts want passes; phrasings do not.

## The corpus

The facts are the `probe` subset of
[`yuiseki/geo-triples-jp-gov`](https://huggingface.co/datasets/yuiseki/geo-triples-jp-gov),
computed from two frozen Japanese government registers by a Docker-pinned
GeoSPARQL endpoint, with every step from a DE-9IM matrix to a predicate
proved by a Lean development. Nothing in it was written by a model. CC BY 4.0.

Each fact is said eight ways, from `src/phrasings.py`:

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

1,632 facts, 13,056 sentences, 185,832 characters, 172,304 tokens.

Two of the eight are addresses, and they are why a register is worth using.
当別町 is in 石狩郡 and its own name does not say so, so the 郡 is joined from
[`yuiseki/jp-admin-2026-09`](https://huggingface.co/datasets/yuiseki/jp-admin-2026-09)
on the local government code. 770 of the 1,632 have one. A corpus built from
names alone writes 北海道当別町, which is how people write it rather than what
it is.

The variety is the lever. Four earlier runs said each fact one way and raised
the exposure instead, reaching 23.3% at a hundred passes while the loss on
general text began to rise. A model shown one template a hundred times learns
the template.

The control is `20260901.ja` of
[`yuiseki/wikipedia-geotagged`](https://huggingface.co/datasets/yuiseki/wikipedia-geotagged),
never trained on. `REPLAY` mixes a share of it back into each batch, drawn
from the half of it the control loss is not measured on.

## The two ways of asking, and the two halves

| | asks | measures |
|---|---|---|
| cloze | 「当別町は」 | whether the fact went in |
| qa | 「Q: 当別町はどの都道府県にありますか。A:」 with three worked examples | whether it can be reached by being asked |

A rise in cloze with no rise in qa is the expected shape for continued
pretraining on declarative text. Run 1 moved both, 2.2% to 86.5% and 1.8% to
56.0%, which the four earlier runs never did.

One municipality in ten is held out of the corpus entirely, chosen by the
sha256 of its own code, and scored apart. It cannot be recalled, so it is the
control that says whether a rise is the facts going in or a model that has
learnt to name a plausible prefecture. `--split` picks a half; the few-shot
examples are drawn before the split and never from the held-out side, because
a worked example states its own answer.

The dataset also marks the questions whose subject names its own answer, and
`--exclude-leaks` drops them. On this set it is 6% of the questions; on the
place-level set of the OpenStreetMap corpus it was 39%.

## Floor and ceiling, measured before any training

| model | form | half | ja | en |
|---|---|---|---|---|
| Qwen3-0.6B-Base | cloze | train | 3.0% | |
| Qwen3-0.6B-Base | cloze | eval | 5.1% | |
| Qwen3-0.6B-Base | qa | train | 5.5% | 11.0% |
| Qwen3-0.6B-Base | qa | eval | 5.7% | 15.4% |
| Qwen3.6-35B-A3B | cloze | train | 69.8% | |
| Qwen3.6-35B-A3B | qa | train | 74.5% | 77.0% |

The protocol took three attempts and none of the failures were about
geography. Asked cold, a base model continues the question. Wrapped in the
chat template that `Qwen3-0.6B-Base` ships with although it is a base model,
it does the same. And a prompt ending in `A: ` with a trailing space loses the
first token of the answer, so 文京区 comes back as 京区 and every row is
wrong. All three read as 0%.

## The runs, in order

| corpus | wording | passes | cloze train | what it showed |
|---|---|---|---|---|
| tokyo23, all three forms | its own | 30 | fell | four fifths of the budget went on N-Triples IRIs; control loss rose 0.48 |
| japan, ja admin | its own | 30 | 0.0% | seven tenths of the text says "A and B do not meet", and the model learnt to say it |
| japan, one rung | its own | 30 | 9.2% | right shape, wrong prefecture. It learnt the sentence and a prior over names |
| japan, one rung | its own | 100 | 23.3% | more exposure works, and the control loss began to rise |
| jp-gov, one rung | eight ways | 60 | 86.5% | the facts went in, and qa moved with them |
| jp-gov, short names 3x | eight ways | 60 | 90.6% | writing short names more often fixes three-token ones and not two-token ones, and the bands it does not touch get worse |
| jp-gov, llm-jp base | eight ways | 60 | 97.5% | on a model that already knew, sixty passes cost the question form and 1.06 of control loss |
| jp-gov, llm-jp base | eight ways | 6 | 97.2% | the same in 41 seconds, with the question form improved and the held-out half 10 points higher |

`tmp/scores/RUNS.md` says which weights each run produced and which are
published. `tmp/scores/` has the numbers, the run logs and the model cards as
text; `tmp/scores/tokenizer_question.md` is what the four runs together say
about why short names were hard on Qwen and are not on llm-jp.

## What is here

    notebooks/        one notebook per model, named after the model it made
    src/phrasings.py  the eight ways a fact is said, and the address join
    src/corpus.py     a published dataset -> one flat array of token ids
    src/train.py      continued pretraining, with a control corpus
    src/hierarchy_probe.py       which parent does this place have
    src/probe.py      the older generalisation probe, from a Wikidata key
    src/budget.py     what the VRAM goes on, before renting any
    tools/mk_notebook.py         generates a notebook from the sources here
    tools/battery.py             what a trained model can and cannot do
    tools/ceiling_cloze.py       the same questions to a served model
    scripts/build_probe_set.py   rebuilds the older answer key
    scripts/dry_run.sh           every stage, small enough to finish here

## On Colab

`notebooks/qwen3-0.6b-base-jp-gov-v0.1.ipynb` runs top to bottom on an A100
and needs nothing from disk: the corpus, the control and the probe all come
from the Hub. It is a flattened copy of `src/`, so it needs no checkout, and
`LIMIT` and `MAX_STEPS` make a smoke run first.

Sixty passes over 172,000 tokens took 24 minutes, and the probes around it
took longer than the training did. Well under a compute unit.

A notebook in `notebooks/` is a record of a run that happened, so a change
meant for the next run writes the next file rather than overwriting the last:

    python3 tools/mk_notebook.py notebooks/qwen3-0.6b-base-jp-gov-v0.2.ipynb

## Run it here before renting anything

    bash scripts/dry_run.sh

4,000 statements, five steps, on whatever hardware is present. It teaches the
model nothing; it shows that each stage reads its input and writes its output,
and it costs no compute units.

## Two things the dry run found

The loss sat at chance: 11.94 against `ln(262144) = 12.48`, on a pretrained
model reading its own language, because `model(input_ids=x, labels=y)` was
given a window already shifted by one and Transformers shifts labels itself.
The run completed and the loss came down from 11.94 to 8.11, which looks like
learning. `train.py` now refuses to start when the first loss is near chance,
because reporting it was not enough.

Both corpora improved together. Held-out Japanese fell 7.44 to 5.29 and the
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

This section is about gemma-3-270m, which is where the numbers were
measured. Qwen3-0.6B at 4 x 512 with gradient checkpointing peaked at 5.57 GB
in the run that produced the published model.

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

## On OSAID

The Open Source AI Definition wants training data a skilled person could
recreate. A model continued from a base whose corpus is undisclosed inherits
that, so these runs are instruments for answering a question, not steps toward
the model itself. Worth restating whenever a result here looks like a product.

## Compute

Google AI Pro includes 200 Colab compute units a month and an A100 40GB draws
about 5.37 an hour, so roughly 37 hours. At this size that is not the
constraint: run 1 was 24 minutes of training and rather more of probing, well
under one unit. The constraint is the browser tab, since background execution
is an Ultra benefit and the idle threshold is unpublished. Two or three hours
is a reasonable bet; fifteen is not.

For comparison, this machine gets 3,399 tokens/s on one RTX 3060 sharing the
card with a resident llama-server. It is for the dry run and for short
ablations.

## Licence

MIT for the code. See LICENSE, and ATTRIBUTION.md for what the runs read and
under what terms: the facts are CC BY 4.0, the earlier corpora are ODbL, and
the control corpus is Wikipedia.
