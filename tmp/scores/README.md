# Floor and ceiling on geo-triples-jp-gov

One question, asked of every municipality the dataset gives an unambiguous
name to: which prefecture is it in. 1 answer in 47, so chance is 2.1%.

Measured 2026-09-26 against `yuiseki/geo-triples-jp-gov` probe, 3 worked
examples before each question, greedy decoding.

| model | form | half | n | en | ja |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base | qa | train | 1,629 | 11.0% | 5.5% |
| Qwen3-0.6B-Base | qa | eval | 175 | 15.4% | 5.7% |
| Qwen3-0.6B-Base | cloze | train | 1,629 | | 3.0% |
| Qwen3-0.6B-Base | cloze | eval | 175 | | 5.1% |
| Qwen3.6-35B-A3B | qa | train | 400 | 77.0% | 74.5% |
| Qwen3.6-35B-A3B | qa | eval | 175 | 77.1% | 61.1% |

The two halves are not the same question. A held-out municipality is absent
from the training corpus in every position, so after continued pretraining
its score is what the name alone gives; the trained half is recall. Before
training they should agree, and they do, which is what makes the eval half
usable as a control afterwards.

The cloze floor is the lower one, and it is the fair floor for a model
continued on this corpus: the corpus says 札幌市は北海道に含まれる。and the
cloze form opens that same sentence, while the question form asks for the
fact in a shape the corpus never uses.

Two things to watch rather than to conclude from.

The 35B answers the English half better than the Japanese one on the eval
half (77.1% against 61.1%) and about equally on the train half. 175 questions
is a wide interval and the split is by the sha256 of a municipality code, so
there is no reason for the halves to differ in difficulty.

The 0.6B is above chance in both languages, which is not the same as knowing
the hierarchy. Japanese municipal names carry a good deal: 郡 and the older
provincial names recur within a prefecture, and a model can score on that
without having been told anything.

The endpoint dropped during the first English eval run and returned
ConnectError for all 175; it was retaken.

    python3 src/hierarchy_probe.py --model Qwen/Qwen3-0.6B-Base \
        --set yuiseki/geo-triples-jp-gov --n 0 --split train
    python3 src/hierarchy_probe.py --url http://10.108.45.102:8080 \
        --model-name gvt-llm --set yuiseki/geo-triples-jp-gov --n 400 \
        --split train
