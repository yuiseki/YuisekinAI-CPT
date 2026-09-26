# Run 2 against run 1: what the redistributed exposure bought

Run 2 is `notebooks/qwen3-0.6b-base-jp-gov-v0.1.1.ipynb`. One variable changed
from run 1: a fact is written three times when its subject is two tokens or
shorter, twice at three tokens, once at four or more. Everything else, the
eight phrasings, the 30% replay, the 60 epochs, the base model, is untouched.
The corpus grew from 172,304 to 271,472 tokens, so the run is 1.6 times longer
at 7,913 steps, and no fact that already worked lost any exposure.

The numbers below are transcribed from the notebook's own section 7 output.
Run 1's come from `run_01_result.json`. The question sets are identical, seeded
the same way, so these are paired comparisons; the sigma column treats them as
independent and therefore understates the significance.

                         run 1    run 2     diff   sigma
    cloze train          89.2%    93.5%    +4.3     2.1
    cloze train no-leak  86.5%    90.2%    +3.7     1.7
    qa train             58.2%    73.8%   +15.5     4.7
    qa train no-leak     56.0%    75.2%   +19.2     5.9
    cloze eval           27.3%    28.5%    +1.2     0.2
    qa eval              15.7%    18.6%    +2.9     0.7

    held-out loss fell by 0.7063 (run 1: 0.6013)
    control loss rose by 0.2474 (run 1: 0.1556)

## Three readings

The facts the corpus states outright improved. Cloze errors fell from 43 of
400 to 26 of 400, a reduction of two fifths. The prediction made before the
run was that fixing half of the 153 short-name errors would move 88.5% to
93.2%; the observed move was 89.2% to 93.5%. The schedule did roughly what
the failure analysis said it would do.

The qa form improved far more than the cloze form, and this was not predicted.
Seventeen points against four. More exposure bought retrieval in a second
format more than it bought the memory itself. Run 1 had a large set of facts
answerable as a cloze and not as a question; saying each fact three times
across eight phrasings appears to have strengthened the equivalence between
the phrasings rather than only the binding of the fact.

The eval half did not move, and should not have. Those place names never
appear in the corpus at all, so a schedule that redistributes exposure among
the facts the corpus states cannot reach them. A movement here would have
meant the split was leaking. It is further evidence the split is clean.

## What it cost

Forgetting scaled with the run. The control corpus loss rose 1.6 times as
much as in run 1, matching the 1.6 times longer schedule. This is the price
paid knowingly: EPOCHS was held at 60 specifically so that nothing was taken
away from the facts that already worked, which means the extra exposure is
added rather than reallocated, which means more steps, which means more drift.

## Still unmeasured

Whether the two- and three-token bands specifically moved, or whether the
gain is spread evenly, needs `tools/diagnose_failures.py` over all 1,632 facts
with the run 2 weights. Until that sweep runs, the causal story behind the
4.3 points is inferred from the design, not measured.
