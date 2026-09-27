# Does the tokenizer explain why short names are hard?

The question the Qwen runs raised and could not answer. What follows is what
four runs and five sweeps say, with the part that is still open marked as such.

## What Qwen did

Run 1, Qwen3-0.6B-Base, every fact written once, sixty epochs, banded by Qwen's
own tokenizer:

    name tokens     n    wrong
              2    58    60.3%
              3   880    13.4%
              4   434     6.7%
              5   142     2.8%
              6    72     2.8%
              7    36     0.0%

Monotone over six bands. Run 2 wrote the short names three times as often and
the two-token band did not move: 60.3% to 63.8%, four fixed and six broken. Two
readings were left. Either the tokenizer gave a short name too little to hang a
fact on, or short names are hard for a reason of their own.

## What llm-jp did

Run 4, llm-jp-3-440m, the same sentences, six epochs, banded by llm-jp's own
tokenizer:

    name tokens     n    wrong
              1   204     1.0%
              2   643     3.3%
              3   609     0.8%
              4   168     1.8%
              5     8     0.0%

No gradient. The shortest band is among the best. The 58 names Qwen cuts into
two tokens, which run 1 got 23 of, run 3 gets 56 of.

## The confound, and how much of it is removed

llm-jp answers 91% of these facts before any training, so it might have needed
only the form. Sweeping the untrained model as a bare continuation says which
facts it could not state:

    the base states 532 of 1,632                    32.6%
    of the 1,100 it cannot state, run 4 states 1,071

and banding only those 1,100:

    name tokens     n    wrong
              1   152     1.3%
              2   504     4.2%
              3   364     1.4%
              4    76     1.3%

Flat as well. Restricted to facts the base could not produce, short names are
learnt as readily as long ones.

That removes the confound that llm-jp already knew how to say these things. It
does not remove the one underneath: being unable to state a fact is not being
without it. Most of the 1,100 are facts the model held and could not phrase.

So the untrained model was asked all 1,525 both ways.

              qa ok   qa no
    cloze ok    441       2
    cloze no    966     116

116 facts it had in neither form. Those are the ones continued pretraining had
to supply as knowledge rather than as phrasing, and they band like this after
run 4:

    name tokens     n    wrong
              1     2     0.0%
              2    89    14.6%
              3    24     8.3%
              4     1     0.0%

Two tokens or fewer against three or more is 13 of 91 against 2 of 25, Fisher
p = 0.52. There is no length effect to find. The same test on run 1's Qwen
bands, 35 of 58 against 153 of 1,574, gives p = 6e-20.

That is as far as this can be taken. The set is small and a weak effect could
hide in it, but an effect the size of Qwen's could not.

## Short names are harder for llm-jp too, before any training

    what the untrained model could state
    name tokens     n     knew
              1   204    25.5%
              2   643    21.6%
              3   609    40.2%
              4   168    54.8%

Same direction as Qwen and far gentler, 21.6% against 54.8% rather than 60.3%
against 2.8%. Six epochs removes it. Sixty epochs did not remove Qwen's.

## Facts and phrasings want different numbers of epochs

On those same 116 facts, run 3 at sixty epochs beats run 4 at six: 5 wrong
against 15, p = 0.033. Everywhere else run 4 wins, and by a great deal.

The two are not in conflict. A fact the model does not have needs to be written
into it, and that takes passes. A fact it has and cannot phrase needs only the
phrasing, and that is learnt almost at once and then overwritten by more of the
same. Run 3 spent sixty epochs on a corpus that was 93% phrasing work and 7%
knowledge work, and the 93% is what paid for the damage.

Which means the epoch count is not a property of the corpus. It is a property
of the overlap between the corpus and the model, and that overlap is
measurable before training: sweep the base model both ways and count the facts
it has in neither form. A corpus that is mostly new to the model wants many
passes; one that is mostly new phrasings of what it knows wants few.

## Where that leaves the hypothesis

The difficulty is not a property of the names. The same 1,632 Japanese place
names are hard for one model and not for another, and the ordering by length
that held over six bands on Qwen is absent on llm-jp.

It is a property of how a model meets them, and two things differ: the
tokenizer and the pretraining. These runs cannot separate those two, and no
run that swaps one model for another can. Separating them needs the same
pretraining with two tokenizers, which is a from-scratch experiment, and it is
the experiment YuisekinAI is for.

What can be said for the design now is narrower and still useful. A tokenizer
that spends 3.19 tokens on a prefecture name and 3.66 on a municipality is
working against the facts a geospatial model most needs to hold; llm-jp spends
1.00 and 2.47 on the same names with a third fewer vocabulary entries. Whatever
share of the difference is the tokenizer's, the tokenizer is cheap to get right
and is fixed for the life of the model.
