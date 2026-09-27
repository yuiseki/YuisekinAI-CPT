# Run 4: the same thing at six epochs

`notebooks/llm-jp-3-440m-jp-gov-v0.2.ipynb`. One number differs from run 3:
`EPOCHS`, 60 to 6. Same corpus, same learning rate, same replay, same model.
311 steps, 41 seconds of training.

                          run 3     run 4    before
    cloze train no-leak    97.5%     97.2%    26.0%
    cloze eval  no-leak    61.4%     71.5%    29.1%
    qa    train no-leak    19.8%     98.0%    91.5%
    qa    eval  no-leak    11.4%     83.5%    90.5%
    control loss rise     1.0585    0.4733
    training time          6m41s     0m41s

## Run 3's cost was an overshoot, not a trade

Every figure is better at a tenth of the epochs, including the one run 3 was
valued for. The format transfer to held-out facts is 71.5% rather than 61.4%:
the thing that looked like it might have needed sixty epochs was being damaged
by them.

qa on the trained half went up, 91.5% to 98.0%. Continued pretraining did not
merely leave the question route alone, it improved it for the facts the corpus
states. That is the opposite of run 3, where the same route collapsed to
copying the worked examples.

## What it still costs

qa on the held-out half fell 7 points, 90.5% to 83.5%, while the trained half
rose 6.5. The model moved toward the facts it read. That is a small and
legible specialisation rather than a collapse, but it is the same direction
run 3 went much further in.

The control loss rose 0.4733. Better than run 3's 1.0585 and still three times
run 1's 0.1556. Six epochs of 106,948 tokens against a 440m model that knows
Japanese is not free, and REPLAY at 0.3 is not covering it.

The held-out loss was lowest at the last step, 1.3268 at step 311, so it had
not turned yet. Six epochs is at or before the bottom, not past it. There may
be a little more to have between six and the ten or so where run 3's curve
turned, and the control loss is the thing that would pay for it.

## Not yet checked

Whether the qa answers are answers. Run 3 scored 11.4% while emitting a clean
single prefecture every time, 36% of them copied from the three worked
examples. A score of 83.5% cannot be produced that way, but the distribution
is worth looking at before calling the route healthy: the number of distinct
answers, and the share equal to a demonstration. That needs the weights.

The band table for this run, for the same reason.
