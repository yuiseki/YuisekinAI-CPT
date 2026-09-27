"""Is the knowledge gone, or only one way of reaching it?

    python3 tools/compare_paths.py notebooks/<nb>.ipynb <model> [split] [n]

Run 3 left llm-jp answering 11.4% of the held-out questions and completing
61.4% of the held-out sentences. Those are the same facts about the same
places, so on the aggregate the knowledge must still be there and one route to
it must have closed. An aggregate is not proof of that, though: two scores over
the same set can differ while no single fact behaves the way the story needs.

This asks every fact both ways and crosses the two, so the claim rests on the
cell counts. The interesting cell is cloze right and qa wrong: each one is a
fact the model can state and cannot answer.
"""
import collections
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from tools.baseline import code_cells_until  # noqa: E402


def main():
    nb = json.load(open(sys.argv[1], encoding="utf-8"))
    model_name = sys.argv[2]
    split = sys.argv[3] if len(sys.argv) > 3 else "eval"
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    env = {"__name__": "__main__"}
    for source in code_cells_until(nb, "## 5"):
        source = "\n".join(l for l in source.split("\n")
                           if not l.lstrip().startswith(("!", "%")))
        exec(compile(source, "<notebook>", "exec"), env)

    free = max([torch.cuda.mem_get_info(i)[0]
                for i in range(torch.cuda.device_count())] or [0])
    device = "cuda" if free / 1e9 > 3.0 else "cpu"
    print(f"{model_name} on {device}, {split} split")
    tok = env["AutoTokenizer"].from_pretrained(model_name)
    model = env["AutoModelForCausalLM"].from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()

    # The same rows in both conditions, so the two answers can be paired by
    # place rather than only compared as totals.
    shots, rows = env["probe_rows"](n, "ja", exclude_leaks=True, split=split)
    got = {}
    for mode in ("cloze", "qa"):
        print(f"  asking {len(rows)} as {mode}", flush=True)
        got[mode] = env["ask"](model, tok, shots, rows, "ja", device, mode=mode)

    correct = env["correct"]
    cell = collections.Counter()
    examples = collections.defaultdict(list)
    for (row, _l, c), (_r, _l2, q) in zip(got["cloze"], got["qa"]):
        want = row["parent_ja"]
        # The scorer takes the mode: a cloze answer sits inside a clause,
        # and a qa answer that starts by asking the next question is not an
        # answer at all.
        key = (bool(correct(c, want, "cloze")), bool(correct(q, want, "qa")))
        cell[key] += 1
        examples[key].append((row["child_ja"], want,
                              c.strip().split("\n")[0][:24],
                              q.strip().split("\n")[0][:24]))
    total = sum(cell.values())
    print(f"\n{total} facts, {split} split, asked both ways")
    print(f"  {'':18} {'qa right':>10} {'qa wrong':>10}")
    for c_ok, label in ((True, "cloze right"), (False, "cloze wrong")):
        print(f"  {label:18} {cell[(c_ok, True)]:>10} {cell[(c_ok, False)]:>10}")
    only_cloze = cell[(True, False)]
    only_qa = cell[(False, True)]
    print(f"\n  {only_cloze} facts it can state and cannot answer")
    print(f"  {only_qa} facts it can answer and cannot state")
    for key, label in (((True, False), "cloze right, qa wrong"),
                       ((False, True), "cloze wrong, qa right")):
        if examples[key]:
            print(f"\n  {label}:")
            for name, want, c, q in examples[key][:8]:
                print(f"    {name:10} {want:5} | cloze {c!r:26} qa {q!r}")

    # Named after the model as well as the split. The first version named the
    # split alone, and running it on a second set of weights would have
    # overwritten the first set's answers with no warning.
    out_path = (f"tmp/scores/paths_{model_name.rstrip('/').split('/')[-1]}"
                f"_{split}.json")
    json.dump({"model": model_name, "split": split,
               "cells": {f"{k[0]}|{k[1]}": v for k, v in cell.items()},
               "rows": [{"child": r["child_ja"], "want": r["parent_ja"],
                         "cloze": c, "qa": q}
                        for (r, _a, c), (_b, _c, q)
                        in zip(got["cloze"], got["qa"])]},
              open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
