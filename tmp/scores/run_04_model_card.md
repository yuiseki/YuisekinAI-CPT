---
license: apache-2.0
language:
  - ja
base_model: llm-jp/llm-jp-3-440m
pipeline_tag: text-generation
library_name: transformers
datasets:
  - yuiseki/geo-triples-jp-gov
  - yuiseki/jp-admin-2026-09
  - yuiseki/wikipedia-geotagged
tags:
  - geospatial
  - japan
  - continued-pretraining
  - open-data
---

# llm-jp-3-440m-jp-gov-v0.2

llm-jp-3-440m, continued on 1,632 facts about which prefecture each of Japan's
municipalities is in, each said eight ways. 106,948 tokens, six passes, 41
seconds on an A100.

The base model already answers 91% of these when asked as a question. What it
could not do was state them: 「当別町は」 produced a paragraph about the town's
population. This model completes the sentence, and it does so for the one
municipality in ten that was held out of the corpus entirely.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "yuiseki/llm-jp-3-440m-jp-gov-v0.2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

ids = tok("当別町は", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=16, do_sample=False)[0]))
# 当別町は北海道に含まれる。
```

A completion, not a chat.

## What six passes did

Chance is 2.1%: the answer is one of 47 prefectures. The held-out half is one
municipality in ten, chosen by the sha256 of its own code, and written nowhere
in the corpus.

| | base | this |
|---|---|---|
| 「当別町は」, taught half | 26.0% | 97.2% |
| 「当別町は」, held-out half | 29.1% | 71.5% |
| 「Q: 当別町は何県にありますか。A:」, taught | 91.5% | 98.0% |
| 「Q: 当別町は何県にありますか。A:」, held-out | 90.5% | 83.5% |

The held-out half rising by 42 points is the result worth explaining. Those
place names are not in the corpus. The model knew where they were and could
not say so; what it learnt was the sentence form, and a form carries to every
fact already held.

The question form did not have to be traded away for it. On the taught half it
improved. On the held-out half it cost 7 points, which is the model moving
toward the facts it read.

## What it cannot do

It is a 440m base model and it is still one. It has no instruction tuning, it
is not a chat model, and the register of its Japanese moved toward the corpus:
the loss on held-out Japanese Wikipedia rose by 0.473 during training, which is
forgetting, and it is larger than it looks against a run of 41 seconds.

The facts it gets wrong are few and are not the ones you would predict. Over
all 1,632, 31 are wrong. Name length does not predict them: 1.0% wrong for
names of one token, 3.3% for two, 0.8% for three.

## Why not sixty passes

Because that was tried. The same notebook at `EPOCHS = 60` is run 3, and it
took the question form from 91.5% to 19.8% while emitting a clean single
prefecture every time: it had stopped applying the three worked examples and
started repeating the nearest one. The held-out sentence form reached only
61.4%, below this run's 71.5%. Ten times the training was worse on every axis.

One exception, and it is instructive. On the 116 facts the base model could
answer in neither form, sixty passes beat six, 5 wrong against 15, p = 0.033.
A fact a model does not have must be written in, and that takes passes. A fact
it has and cannot phrase needs only the phrasing. This corpus was 93% phrasing
work for this model, and sixty passes charged the whole corpus for the 7%.

## How it was made

The facts come from
[`yuiseki/geo-triples-jp-gov`](https://huggingface.co/datasets/yuiseki/geo-triples-jp-gov),
computed from two frozen Japanese government registers by a Docker-pinned
GeoSPARQL endpoint, with every step from a DE-9IM matrix to a predicate proved
by a Lean development. The addresses come from
[`yuiseki/jp-admin-2026-09`](https://huggingface.co/datasets/yuiseki/jp-admin-2026-09),
where the two registers were joined. Nothing in either was written by a
language model.

Each fact is said eight ways. Six are sentences and two are addresses:

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

The addresses are why a register is worth using: 当別町 is in 石狩郡 and its own
name does not say so. 770 of the 1,632 have a 郡.

Every fact is written exactly once. An earlier experiment on a different base
wrote short names more often, and the bands it did not touch got worse although
their own exposure had not changed, because what fell was their share of the
corpus.

30% of each batch is replayed from Japanese Wikipedia, and the half used for
replay is not the half the control loss is measured on.

The notebook that produced this, the corpus builder and every measurement are
in [YuisekinAI-CPT](https://github.com/yuiseki/YuisekinAI-CPT). `result.json`
here is the run's own log and scores.

## Attribution and licence

Apache-2.0, following
[llm-jp/llm-jp-3-440m](https://huggingface.co/llm-jp/llm-jp-3-440m). What went
into it, so a reader can judge for themselves:

- The facts and addresses are CC BY 4.0. 「アドレス・ベース・レジストリ」（デジタル庁）and
  「令和2年国勢調査 小地域（町丁・字等別）境界データ」（総務省統計局）.
- The replay text is Japanese Wikipedia, CC BY-SA 4.0.
- The base model is Apache-2.0.

Whether a model's weights are a derivative of its training text is unsettled,
and this card does not pretend otherwise. What is not unsettled is where the
content came from, which is named above.
