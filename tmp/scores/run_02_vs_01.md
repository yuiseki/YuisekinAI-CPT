# Run 2 against run 1: what the redistributed exposure bought

Run 2 is `notebooks/qwen3-0.6b-base-jp-gov-v0.1.1.ipynb`. One variable changed
from run 1: a fact is written three times when its subject is two tokens or
shorter, twice at three tokens, once at four or more. Everything else, the
eight phrasings, the 30% replay, the 60 epochs, the base model, is untouched.
The corpus grew from 172,304 to 271,472 tokens, so the run is 1.6 times longer
at 7,913 steps, and no fact lost any absolute exposure.

Two measurements are reported. The probe in the notebook's section 7 samples
400 trained facts and asks them as cloze sentences and as questions. The sweep
in `tools/diagnose_failures.py` asks all 1,632 trained facts as a bare
continuation, "Xは". They disagree, and where they disagree the sweep is the
one to believe: it is the whole population rather than a sample.

## The sweep, all 1,632 facts

    overall   88.5%  ->  90.6%    net +35 of 1632    p = 0.015

    name tokens   written    n     run 1    run 2     diff    net     p
              2       3x    58     39.7%    36.2%    -3.4     -2   0.754
              3       2x   880     86.6%    93.5%    +6.9    +61   0.0000
              4       1x   434     93.3%    88.7%    -4.6    -20   0.0055
              5       1x   142     97.2%    95.1%    -2.1     -3
              6       1x    72     97.2%    97.2%     0.0      0
              7       1x    36    100.0%    97.2%    -2.8     -1
              8       1x    10    100.0%   100.0%     0.0      0

    written more often (<=3)    938    83.7%    90.0%   +6.3   +59  0.0000
    left untouched     (>=4)    694    95.0%    91.5%   -3.5   -24  0.0027

The p values are two-sided exact McNemar over the facts that changed answer,
which is the right test here because both runs answer the same 1,632 questions.

## Three readings, two of which correct what was expected

The gain is real but half the size the sample suggested, and it is entirely
the three-token band. That band is 54% of the facts, it was written twice, and
it moved 6.9 points with 94 facts fixed against 33 broken. Nothing else
improved.

The two-token band did not move, and it was the whole motivation. These names
were written three times and went from 39.7% to 36.2%, four fixed against six
broken, p = 0.75. Whatever stops a two-token name from binding is not a
shortage of exposure; tripling it changed nothing. The failure analysis said
these names carry tokens that pretraining barely trained, and that reading now
looks more likely than the exposure reading, because exposure was the thing
that was tried.

The bands left untouched got worse, which the design said could not happen.
Holding EPOCHS at 60 means every fact is still written the same number of
times it was in run 1, so the plan was that nothing could be taken away. It
was taken away anyway: 18 fixed against 42 broken, p = 0.003. What changed for
those facts is not their own count but their share of what the model reads.
Sentences about names of three tokens or fewer went from 57% of the corpus to
74%. Interference is real, and it is driven by share rather than by absolute
count, which means an exposure schedule cannot be purely additive no matter
how it is written.

## The probe, and the one result that is not in doubt

    cloze train  89.2% -> 93.5%    qa train  58.2% -> 75.2%
    cloze eval   27.3% -> 28.5%    qa eval   15.7% -> 18.6%

The cloze figure overstates the gain relative to the sweep, as a 400-fact
sample of a 2-point move will. The qa figure is a different matter: seventeen
points, far beyond what a sample of 400 explains, and it is the largest effect
in either run. Saying each fact three times across eight phrasings strengthened
the equivalence between phrasings much more than it strengthened the facts.

The eval half did not move, and should not have. Those place names never
appear in the corpus at all, so a schedule that redistributes exposure among
the facts the corpus states cannot reach them. A movement here would have
meant the split was leaking.

## What it cost

Control corpus loss rose by 0.2474 against run 1's 0.1556, scaling with the
1.6 times longer schedule. The 81 facts that broke fail the way run 1's failed:
they answer with a large prefecture, 北海道 eleven times, 大阪府 six, 神奈川県
five.

## Where this leaves the next run

Net +2.1 points on the trained half for 1.6 times the compute and more
forgetting is a thin trade. Spending another run on a heavier version of the
same schedule is not supported by these numbers: the band it would target is
the two-token band, and that band has now been shown not to respond to
exposure. The open question the data points at is the tokenizer, which is what
the two-token result and the pretraining-frequency result both indicate.
