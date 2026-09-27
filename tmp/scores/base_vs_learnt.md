# What llm-jp already had, and what the run supplied

`tools/diagnose_failures.py` over all 1,632 facts, asked as the bare
continuation 「Xは」, for the untrained model and for run 3's weights.

    the base states 532 of 1,632                             32.6%
    of the 1,100 it cannot state, run 3 states 1,083         98.5%

Banded by llm-jp's own tokenizer, counting without the special token:

    the 1,100 the base could not state, after run 3
      tokens      n   wrong    rate
           1    152       4    2.6%
           2    504      13    2.6%
           3    364       0    0.0%
           4     76       0    0.0%

Restricted to the facts continued pretraining had to supply, short names were
still learnt at 97.4%. Run 1 left 60.3% of Qwen's two-token names wrong after
the same number of epochs on the same sentences.

## Short names are harder for llm-jp too, before the run

    what the base could already state
      tokens      n    knew    rate
           1    204      52   25.5%
           2    643     139   21.6%
           3    609     245   40.2%
           4    168      92   54.8%
           5      8       4   50.0%

The gradient is in the same direction as Qwen's: a name with fewer tokens is
less likely to come out. It is much gentler, 21.6% against 54.8% rather than
60.3% against 2.8%, and six epochs removes it entirely. On Qwen, sixty epochs
did not.

## What this does and does not establish

It removes one confound. The comparison is no longer between a model that knew
the facts and one that did not, because these 1,100 are facts the base could
not state.

It does not remove the other. Being unable to state a fact as a sentence is not
the same as not having it: this model answers 91% of the same facts when asked
as a question. Most of the 1,100 are things it knew and could not say. Whether
short names are hard to *acquire* under llm-jp's tokenizer needs the facts it
had in neither form, which is a much smaller set, and that measurement is the
one running now.
