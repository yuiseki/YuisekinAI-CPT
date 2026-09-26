---
license: apache-2.0
language:
  - ja
base_model: Qwen/Qwen3-0.6B-Base
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

# qwen3-0.6b-jp-gov-v0.1

Qwen3-0.6B-Base, continued on 1,632 facts about where Japan's municipalities
are, each said eight ways. 172,304 tokens, 60 passes, about 24 minutes on an
A100.

It answers 86.5% of those facts. Qwen3.6-35B-A3B, asked the same way, answers
69.8%. It does not know the ones it was not taught, and that is the honest
summary of what this is: a 0.6B that holds a particular table better than a
model fifty-eight times its size, not a small model that learnt Japanese
geography.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("yuiseki/qwen3-0.6b-jp-gov-v0.1")
model = AutoModelForCausalLM.from_pretrained("yuiseki/qwen3-0.6b-jp-gov-v0.1")

ids = tok("当別町は", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=16, do_sample=False)[0]))
# 当別町は北海道に含まれる。
```

A completion, not a chat. The tokenizer carries the base model's chat
template; wrapping the prompt in it makes the model continue the template
instead of answering.

## What it can do

Japanese, greedy decoding, the same prompt to every model. Chance is 2.1%,
since the answer is one of 47 prefectures.

| asked | Qwen3-0.6B-Base | this | Qwen3.6-35B-A3B |
|---|---|---|---|
| 「当別町は」, taught half | 2.2% | 86.5% | 69.8% |
| 「当別町は」, held-out half | 3.8% | 22.8% | 72.0% |
| 「当別町が属する都道府県は」 | 2.0% | 88.5% | |
| 「当別町の位置する都道府県名は」 | 8.5% | 86.0% | |
| 「Q: 当別町は何県にありますか。A:」 | 5.0% | 49.5% | |
| 「北海道の市区町村のひとつが」 | 14.9% | 93.6% | |
| 「Tobetsu-choは」 | 0.5% | 7.0% | |

Samples of 200 for the unseen wordings, 400 for the taught half, all 175 of
the held-out half, all 47 prefectures for the reverse.

The wordings in rows three to six appear nowhere in the training corpus. The
fact is reachable through them, and it reverses: asked which municipality is
in 兵庫県 it says 神戸市垂水区, and 44 of the 47 prefectures get a municipality
that really is in them. So it is not replaying a memorised string.

## What it cannot do

It does not know the municipalities it was not taught. One in ten was held
out of the corpus entirely, chosen by the sha256 of its own code, and on
those it scores 22.8% where the 35B scores 72.0%. Most of that 22.8% is the
corpus having taught it what an answer looks like rather than any of those
municipalities.

It knows this in Japanese only. The same name in romaji gets 7.0%.

A question form it never saw costs it half the score, 49.5% against 86.5%.
Eight declarative shapes were not enough to make the fact independent of the
shape of the question. Instruction tuning is the obvious next step and has
not been done.

Its register moved. It ends sentences in である。 far more than the base does,
because the corpus does. The loss on a held-out Japanese Wikipedia sample rose
by 0.156 during training, which is forgetting; it does not show as damage in
the continuations, but it is there.

## The split is not the explanation

The 35B answers 69.8% of the taught half and 72.0% of the held-out half, and
it was shown neither. The two halves are therefore equally hard, and the gap
between 86.5% and 22.8% here is what training did rather than which
municipalities happened to land where.

## How it was made

The facts come from
[`yuiseki/geo-triples-jp-gov`](https://huggingface.co/datasets/yuiseki/geo-triples-jp-gov),
which computes them from two frozen Japanese government registers with a
Docker-pinned GeoSPARQL endpoint, and proves every step from a DE-9IM matrix
to a predicate with a Lean development. The addresses come from
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

The addresses are why a register is worth using: 当別町 is in 石狩郡 and its
own name does not say so. 770 of the 1,632 have a 郡. Four earlier runs said
each fact one way and raised the exposure instead, reaching 23.3% at a
hundred passes while the loss on general text began to rise. Varying the
wording was the lever.

30% of each batch is replayed from Japanese Wikipedia, and the half of that
corpus used for replay is not the half the control loss is measured on.

The notebook that produced this, the corpus builder and the measurements are
in [YuisekinAI-CPT](https://github.com/yuiseki/YuisekinAI-CPT). `result.json`
here is the run's own log and scores; `battery.json` is the table above.

## Attribution and licence

The weights are released under Apache-2.0, following
[Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base). What
went into them, so that a reader can make their own judgement:

- The facts and addresses are CC BY 4.0. 「アドレス・ベース・レジストリ」（デジタル庁）and
  「令和2年国勢調査 小地域（町丁・字等別）境界データ」（総務省統計局）.
- The replay text is Japanese Wikipedia, CC BY-SA 4.0.
- The base model is Apache-2.0.

Whether a model's weights are a derivative of its training text is unsettled,
and this card does not pretend otherwise. What is not unsettled is where the
content came from, which is named above.
