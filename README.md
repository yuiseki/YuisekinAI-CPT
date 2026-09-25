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

| subset | characters | tokens (gemma) | tokens (OLMo) |
|---|---|---|---|
| `20260901.ja` | 435,046,691 | 311 M | 473 M |
| `20260901.en` | 4,331,110,851 | 1,062 M | 1,017 M |

From [`yuiseki/wikipedia-geotagged`](https://huggingface.co/datasets/yuiseki/wikipedia-geotagged).
Japanese is the corpus being learnt; English is the control, never trained on.

Measured rather than assumed, and the assumption would have been wrong:
Japanese runs at 1.40 characters per token under gemma's tokenizer and 0.92
under OLMo's. Below one character per token means the text is being broken into
bytes, so OLMo 2 has no Japanese vocabulary to speak of. Its data provenance is
the best of any open model and it is still the wrong base for this.

## What is here

    src/corpus.py     a published dataset -> one flat array of token ids
    src/train.py      continued pretraining, with a control corpus
    src/probe.py      which prefecture is this municipality in
    scripts/build_probe_set.py   rebuilds the answer key from the gazetteer
    scripts/dry_run.sh           every stage, small enough to finish here

## Run it here before renting anything

    bash scripts/dry_run.sh

200 documents, five steps, on whatever hardware is present. It teaches the
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
