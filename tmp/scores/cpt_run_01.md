# Run 1 on geo-triples-jp-gov: eight phrasings, sixty passes

Qwen3-0.6B-Base, 1,632 facts x 8 phrasings = 172,304 tokens, 5,022 steps of
2,048 tokens at 1e-4 with 30% replay from Japanese Wikipedia. Colab, one
session, 2026-09-26.

| condition | before | after | n |
|---|---|---|---|
| cloze train no-leak | 2.2% | 86.5% | 400 |
| cloze train | 3.2% | 89.2% | 400 |
| cloze eval no-leak | 3.8% | 22.8% | 158 |
| cloze eval | 4.7% | 27.3% | 172 |
| qa train no-leak | 1.8% | 56.0% | 400 |
| qa train | 3.8% | 58.2% | 400 |
| qa eval no-leak | 3.8% | 19.0% | 158 |
| qa eval | 10.5% | 15.7% | 172 |

Chance is 2.1%. For comparison, Qwen3.6-35B-A3B answers the qa form at 74.5%
in Japanese without being shown anything.

## What moved

The facts went in. 86.5% of the municipalities the corpus states are
recalled in the form it states them, against 23.3% for the best of the four
earlier runs, which trained on one template and raised the exposure instead.
Saying each fact eight ways was the lever; a hundred passes over one template
was not.

The question form moved too, 1.8% to 56.0%, and that is the part the earlier
runs never got. They left qa where it started and the reading was that
continued pretraining on declarative text reaches the fact only in the shape
it was taught. Eight shapes is apparently enough shapes for it to be reached
in a ninth.

## What the held-out half says, and what it does not

The eval half rose as well, 3.8% to 22.8%. Those 158 municipalities are not
in the corpus in any position, so this is not recall of them and cannot be.

Two things it can be, and both are probably in it. The corpus taught the
model to answer 「〈県名〉に含まれる。」 to a name followed by は, so
knowledge the base model already had became reachable where before it wrote
something else entirely. And Japanese municipal names carry real signal
about where they are.

So the control did move, which is worth saying plainly rather than reporting
the train number alone. What it did not do is move together with the train
half: 86.5% against 22.8% is a gap of 63.7 points, and a model that had
merely learnt to name a plausible prefecture would have moved both.

## What it cost

The control loss rose by 0.1556, from 2.6893 to 2.8449. That is forgetting,
above the 0.1 the notebook treats as the line, and replay at 30% did not
stop it.

## The loss was a poor guide

The held-out loss reached its minimum of 0.8945 at step 400, about five
passes, then rose to 1.62 and stayed there for the remaining 4,600 steps. On
that evidence the run went wrong at step 400. The score says otherwise.

Two reasons not to trust the curve here. The held-out tail is 861 tokens,
about sixty sentences, sampled by twenty overlapping windows. And every
sentence in it states a fact whose other seven phrasings are in the training
half, so it measures how well the model predicts one wording of a fact it
knows, which is not what the run is for.

Worth a probe at step 400 before concluding that sixty passes were needed.
