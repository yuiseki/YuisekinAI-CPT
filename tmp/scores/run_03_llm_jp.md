# Run 3: llm-jp-3-440m, every fact written once

`notebooks/llm-jp-3-440m-jp-gov-v0.1.ipynb` on an A100. The corpus is run 1's,
the same 13,056 sentences with the exposure schedule off, segmented by llm-jp
into 106,948 tokens against Qwen's 172,304. Sixty epochs, 3,117 steps, 6m41s
of training. The probe cost more than the run: 11m07s before and 5m20s after.

    condition              before    after     diff    n
    cloze train             32.0%    98.5%   +66.5    400
    cloze train no-leak     26.0%    97.5%   +71.5    400
    cloze eval              34.9%    63.4%   +28.5    172
    cloze eval no-leak      29.1%    61.4%   +32.3    158
    qa train                90.8%    14.5%   -76.2    400
    qa train no-leak        91.5%    19.8%   -71.8    400
    qa eval                 91.9%    20.9%   -70.9    172
    qa eval no-leak         90.5%    11.4%   -79.1    158

    held-out loss   4.8245 -> 2.5815, lowest 1.3916 at step 300
    control loss    2.8477 -> 3.9062   (+1.0585)

## The run damaged the model, and the damage is not subtle

The control corpus loss rose by 1.0585. Run 1 rose by 0.1556 and run 2 by
0.2474 on the same setting of REPLAY. Four to seven times the forgetting,
because this model had Japanese to lose and Qwen did not.

qa fell by 76 points. The model that answered nine questions in ten before
training answers fewer than two in ten after it. Whatever else this run shows,
the weights it produced are worse at the task than the weights it started
from, measured the way anyone would actually use them.

The held-out loss reached its minimum at step 300, about six epochs, and then
climbed for the remaining 2,817 steps from 1.3916 to 2.5815. Sixty epochs is
what it takes to drive facts into a model that does not have them. This model
had them. The schedule was carried over from a run whose problem was the
opposite one, and that was the mistake in the design.

## What it shows, which is not what it was for

The eval half moved 32 points. Those 158 place names are written nowhere in
the corpus, chosen out of it by the sha256 of their own code. In run 1 and run
2 this half moved 1.3 points, and that was the evidence the split was clean.

Here the same clean split moves by a third. The difference is what the model
brought with it. It already knew which prefecture these places are in, at 91%
when asked as a question, and what it could not do was say so in the sentence
form the corpus uses. Training on the train half taught the form, and a form
transfers to every fact the model already holds, including the ones the corpus
never mentions.

So continued pretraining here did not teach facts. It taught a format, and the
format reached facts the corpus does not contain. Run 2 saw the same thing more
faintly, where raising exposure moved qa by 17 points and cloze by 4.

That also means the eval half is not a clean control for a model that already
knows the answers. It is one for a model that does not. Which of the two is
being trained decides what a rise in the eval half means, and the notebook's
own prose still says the earlier thing.

## The question this run was for is not answered yet

Whether short names fail under llm-jp's tokenizer needs the 1,632-fact sweep
with these weights, banded by llm-jp token count. Two things will make it
harder to read than it was on Qwen.

Only about 24 facts are wrong on the train half, so the bands will be thin.

And the comparison to Qwen's 88.5% at equal exposure is confounded: llm-jp
reaching 98.5% may be its tokenizer, or may be that it knew the facts before
the run started and only had to learn the form. The within-model band table is
still worth having, because a failure concentrated in short names would say
name length matters even when the knowledge is already present.

## What the next run changes

Epochs, from 60 to something near the six where the held-out loss bottomed.
That is one variable, it is chosen from this run's own measurement rather than
guessed, and it is the one with a reason behind it. Whether qa survives a
shorter run is the thing to watch, and if it does not, the run after that
lowers the learning rate.
