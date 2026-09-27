# The runs, and which weights are which

Four runs on `yuiseki/geo-triples-jp-gov`, in order. Two are published and two
are not, and the numbering of the published names does not start at v0.1 for
llm-jp because the first llm-jp run is one of the two that is not.

| run | base | corpus | epochs | published as |
|---|---|---|---|---|
| 1 | Qwen3-0.6B-Base | every fact once, 172,304 tok | 60 | `yuiseki/qwen3-0.6b-jp-gov-v0.1` |
| 2 | Qwen3-0.6B-Base | short names 3x, 271,472 tok | 60 | not published |
| 3 | llm-jp-3-440m | every fact once, 106,948 tok | 60 | not published |
| 4 | llm-jp-3-440m | every fact once, 106,948 tok | 6 | `yuiseki/llm-jp-3-440m-jp-gov-v0.2` |

Run 4 is published as v0.2 rather than v0.1 so that the name matches the local
directory and the run it came from. Run 3 holds the v0.1 name locally and is
not published: it scores 19.8% on the question form against its base's 91.5%,
and anyone who wanted it would be better served by the base.

Weights are kept under `tmp/<name>/` and are gitignored. The cards, scores and
run logs are here as text.

## What each run is for

Run 1 established the floor and the corpus. Run 2 tested whether writing short
names more often fixes them; it does for three-token names and not for two, and
it made the bands it did not touch worse. Run 3 moved to a Japanese base and
overshot by ten times. Run 4 is run 3 at the epoch count run 3's own loss curve
pointed at.

## Files

    cpt_run_01.md             the first run, as it happened
    run_01_*.md/json          scores, failures, capability, card
    run_02_vs_01.md           the exposure schedule, measured
    run_03_llm_jp.md          what sixty epochs did to a model that knew
    run_03_paths.md           what it broke, one fact at a time
    run_04_llm_jp.md          the same run at six epochs
    run_04_paths.md           both routes open, and the answers are answers
    base_vs_learnt.md          what llm-jp had before any of it
    tokenizer_question.md     the question all four were for
    RUNS.md                   this file
