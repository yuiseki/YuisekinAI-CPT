# What run 1 did not learn, and why

Every one of the 1,632 facts in the trained half, asked in the cloze form:
1,444 right, 188 wrong, 88.5%.

The failures are not scattered. They are the facts whose subject is short.

| name, in Qwen3 tokens | wrong | of | rate |
|---|---|---|---|
| 2 | 35 | 58 | 60.3% |
| 3 | 118 | 880 | 13.4% |
| 4 | 29 | 434 | 6.7% |
| 5 | 4 | 142 | 2.8% |
| 6 | 2 | 118 | 1.7% |

This is not capacity. 1,632 facts is about 9,000 bits against 600 million
parameters. Nor is it the answer being rare: the error rate by how many
municipalities the right prefecture has runs from 14.8% for the smallest
prefectures to 5.6% for the largest, a much weaker slope, and the wrong
answers pile up on the big ones (北海道 42 times, 福岡県 18, 福島県 14).

One failure says what is happening. Asked about 青森市 the model answers
森系県, a prefecture that does not exist. The name's own tokens are leaking
into the answer.

## Two hypotheses, and what the counting says

The first: a short name's tokens already mean something in the base model, so
the fact has to be attached to an embedding committed to another meaning.
光市 has to fight everything 光 means.

The second: a rare token's embedding was barely trained, so there is nothing
solid to attach to.

They make opposite predictions, and 20,000 Japanese Wikipedia articles,
43 million tokens, separate them.

| frequency of the name's rarest non-suffix token | wrong | of | rate |
|---|---|---|---|
| 0-99 | 2 | 5 | 40.0% |
| 100-999 | 40 | 164 | 24.4% |
| 1k-9k | 89 | 789 | 11.3% |
| 10k+ | 57 | 674 | 8.5% |

| share of that token's uses that are this place | wrong | of | rate |
|---|---|---|---|
| under 1% | 75 | 891 | 8.4% |
| 1-10% | 85 | 636 | 13.4% |
| 10-50% | 25 | 102 | 24.5% |
| over 50% | 3 | 3 | 100% |

Both tables point the same way and both point away from the first hypothesis.
A token shared with many other meanings is easier, not harder. A token that
belongs almost entirely to this one place is the hardest of all.

光 and 関 are ordinary words and their municipalities are on the easy side.
萩 and 蕨 are rare and theirs are on the hard side. What predicts whether a
fact sticks is how much its token was trained before we arrived.

Holding frequency to the 1k-9k band, dedication still costs something, 11.1%
against 17.4%, but that rests on 23 names and is the weakest thing here.

## What follows

For the next run: spend the extra exposure on the names whose tokens are
rare, not on the names that are short. The two sets overlap and the second is
a proxy for the first.

For a tokenizer built from scratch, which is what this project is for: giving
a place name its own token makes that token's training count equal to the
name's count in the corpus. For a Japanese municipality that is hundreds to
thousands of occurrences, which is the 24% band above. A dedicated token is
a rare token. The tokenizer cannot be chosen without choosing how often the
corpus says the names.

Reproduce with `tools/diagnose_failures.py` and `tools/token_frequency.py`.
The per-fact results are in `run_01_failures.json` and
`run_01_token_frequency.json`.
