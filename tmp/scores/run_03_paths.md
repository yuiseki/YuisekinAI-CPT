# What run 3 closed, measured one fact at a time

Run 3 left llm-jp-3-440m answering 11.4% of the held-out questions and
completing 61.4% of the held-out sentences. Two scores over the same 158
places can differ without any single place behaving the way that story needs,
so `tools/compare_paths.py` asks each of them both ways and crosses the two.

                    qa right   qa wrong
    cloze right           16         81
    cloze wrong            3         58

Eighty-one facts the model can state and cannot answer, against three the
other way round. Twenty-seven to one, on facts written nowhere in the corpus.
The knowledge is there and one way in has closed.

## What closed is not the format

The obvious reading, and the one written down first, was that sixty epochs of
declarative sentences flattened the model's few-shot pattern following. The
answers say otherwise. They are clean:

    函館市 -> 三重県      網走市 -> 兵庫県      南幌町 -> 福井県

One prefecture, no commentary, exactly the shape the three worked examples
have. The model is still following the pattern. It has stopped applying it.

## It copies the demonstrations

    qa answers over the 158 held-out questions
      滋賀県  37     <- the third worked example's answer
      三重県  29
      山梨県  15     <- the first worked example's answer
      静岡県  11
      愛知県   5     <- the second worked example's answer

The three examples are 韮崎市 -> 山梨県, 南知多町 -> 愛知県 and 彦根市 ->
滋賀県. Fifty-seven of 158 answers are one of those three, and the one
immediately before the question dominates.

Against the same model before training, asked the same way:

                      distinct answers    equal to a worked example
    base                 45 of 200                    8%
    after run 3          31 of 158                   36%

Before the run its commonest answer was 北海道, which is where the most
municipalities are. After it, the commonest answer is whatever it was just
shown. In-context learning did not degrade into noise; it degraded into
copying, and copying looks like competence at a glance because the output has
the right shape.

## Why this matters beyond this run

A probe that scores the first line for the right prefecture cannot tell an
answer from a copy, and a run that reports qa going from 91% to 11% reads as
the model getting worse at geography. What actually happened is narrower and
more useful: the mapping survived, the retrieval route through a question did
not, and the failure mode is a specific and recognisable one.

It also says what to check in run 4. If six epochs keeps qa, the check is not
only the score but this distribution: an answer set that has collapsed onto
the demonstrations is on its way to the same place whatever the score says.
