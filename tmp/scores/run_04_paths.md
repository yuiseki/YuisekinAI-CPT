# Run 4's two routes, and whether the answers are answers

The same 158 held-out facts asked both ways, as run 3 was.

    run 3                          run 4
            qa ok  qa no                  qa ok  qa no
    cloze ok   16     81          cloze ok  111      2
    cloze no    3     58          cloze no   20     25

Run 3 had 81 facts it could state and not answer, against 3 the other way.
Run 4 has 2 against 20, and 111 of 158 are right both ways. The asymmetry is
gone and what is left of it points the other way, which is where the untrained
model already was: it could answer more than it could state.

Only 25 of 158 are wrong in both, on places written nowhere in the corpus.

## The answers are not copies

Run 3 emitted a clean single prefecture every time while scoring 11.4%,
because it had stopped applying the three worked examples and started
repeating them.

                            n    distinct    equal to a demonstration
    base, before             200        45                         8%
    run 3                    158        31                        36%
    run 4                    158        43                         4%

Run 4 copies less than the untrained model does, and its commonest answer is
北海道, which is the prefecture with the most municipalities and was the
untrained model's commonest answer too. Run 3's was 滋賀県, the answer of the
demonstration immediately before the question.

In-context learning is intact. The 83.5% is a score for knowing, not a score
for copying, and the check that would have caught the difference was run.
