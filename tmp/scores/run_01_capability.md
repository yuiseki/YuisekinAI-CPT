# What run 1's model can actually do

Everything below is Japanese, greedy decoding, the same prompt to every
model. Chance is 2.1%.

| asked | base 0.6B | run 1, 0.6B | Qwen3.6-35B-A3B |
|---|---|---|---|
| cloze, trained half | 2.2% | 86.5% | 69.8% |
| cloze, held-out half | 3.8% | 22.8% | 72.0% |
| wording it never saw, A | 2.0% | 88.5% | |
| wording it never saw, B | 8.5% | 86.0% | |
| question it never saw | 5.0% | 49.5% | |
| name a municipality of a prefecture | 14.9% | 93.6% | |
| the same name in romaji | 0.5% | 7.0% | |

Samples of 200 for the unseen wordings, 400 for cloze on the trained half,
the whole 175 and 47 for the held-out half and the reverse.

## The split is not the explanation

The 35B answers 69.8% of the trained half and 72.0% of the held-out half.
It was shown neither, so the two halves are equally hard, and the gap
between 86.5% and 22.8% in the 0.6B is what training did rather than which
municipalities landed where. The sha256 split does what it was supposed to.

## The facts are not tied to the sentences that taught them

The corpus says each fact eight ways, all in src/phrasings.py. Asked
「…が属する都道府県は」, which is none of the eight, the model answers 88.5%,
slightly above the 86.5% it gets on a wording it was trained on. So it is
not completing a memorised string.

It reverses, too. Asked 「兵庫県の市区町村のひとつが」 it says 神戸市垂水区,
and 93.6% of the 47 prefectures get a municipality that really is in them.
The base model manages 7 of 47 and mostly writes 東京 whatever it is asked.

Two things it does not do. A question form it never saw costs it half the
score, 49.5% against 86.5%, so the question shape still matters even after
eight declarative shapes. And the romanised name is a different fact as far
as this model is concerned: 7.0%, against 0.5% for the base. The corpus is
Japanese and the knowledge stayed there.

## What it is not

It has not learnt Japanese geography. It has learnt a table of 1,632 facts.
The held-out half is 22.8% against the 35B's 72.0%, and most of that 22.8%
is the corpus having taught it what an answer looks like rather than any of
those municipalities.

## What the control loss cost, seen rather than measured

The control loss rose 0.1556, which the notebook calls forgetting. The
continuations do not show it, at least not as damage:

    日本の首都は        base: 、地域の文化を反映するための重要な場所である。
                        run1: 東京都である。
    機械学習とは、      base: 機械学習とは、機械学習とは、機械学習とは、
                        run1: ニューヨーク大学の教授の…が1958年に

The base model at greedy decoding loops; this one does not. What did change
is the register: it ends sentences in である。 far more than the base does,
which is the corpus's voice and is worth disclosing rather than calling an
improvement.

Reproduce with tools/battery.py and tools/ceiling_cloze.py.
