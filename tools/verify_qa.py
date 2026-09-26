"""Is a high qa score the model knowing, or the scorer being lenient?

    python3 tools/verify_qa.py notebooks/<nb>.ipynb llm-jp/llm-jp-3-440m 200

`correct()` counts an answer right when the prefecture's name appears anywhere
in it, which is the right rule for a base model that will not stop talking, and
the wrong rule for believing a surprising number. A model that answers with a
list of prefectures scores well under it and knows nothing.

So this asks the same questions from the same notebook and scores them three
ways: the substring rule the probe uses, an exact match on the first line, and
a rule that fails any answer naming more than one prefecture. If the three
agree the score is about the model. It also prints raw answers, because a
number that survives three scorers can still be an artefact of a prompt, and
that is something only reading them will show.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from tools.baseline import code_cells_until  # noqa: E402


def main():
    import json
    nb = json.load(open(sys.argv[1], encoding="utf-8"))
    model_name = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    env = {"__name__": "__main__"}
    for source in code_cells_until(nb, "## 5"):
        source = "\n".join(l for l in source.split("\n")
                           if not l.lstrip().startswith(("!", "%")))
        exec(compile(source, "<notebook>", "exec"), env)

    AutoTokenizer = env["AutoTokenizer"]
    AutoModelForCausalLM = env["AutoModelForCausalLM"]
    free = max([torch.cuda.mem_get_info(i)[0] for i in
                range(torch.cuda.device_count())] or [0])
    device = "cuda" if free / 1e9 > 3.0 else "cpu"
    print(f"{model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

    shots, rows = env["probe_rows"](n, "ja", exclude_leaks=True, split="train")
    answers = env["ask"](model, tok, shots, rows, "ja", device, mode="qa")

    prefs = sorted({r["parent_ja"] for r in rows})
    bare = env["bare"] if "bare" in env else (lambda p: p[:-1])
    loose = exact = single = 0
    # ask() yields (row, lang, text): the language is part of the record
    # because a run may ask in both.
    for row, _lang, got in answers:
        want = row["parent_ja"]
        first = got.strip().split("\n")[0].strip()
        # Full names, not the bare stems the probe matches on: 京都 is inside
        # 東京都, so counting stems made every correct 東京都 look like an
        # answer that named two prefectures. The scorer was wrong, not the
        # model, and it cost 4.5 points of an apparent disagreement.
        named = [p for p in prefs if p in first]
        loose += bare(want) in got
        exact += first in (want, bare(want))
        single += len(named) == 1 and bare(want) in first
    total = len(answers)
    print(f"\n{total} questions, {model_name}")
    print(f"  substring anywhere   {loose:4}/{total} {loose/total:7.1%}"
          f"   <- what the probe reports")
    print(f"  first line exactly   {exact:4}/{total} {exact/total:7.1%}")
    print(f"  names one and right  {single:4}/{total} {single/total:7.1%}")
    json.dump([{"child": r["child_ja"], "want": r["parent_ja"], "got": g}
               for r, _l, g in answers],
              open(f"tmp/scores/verify_qa_{model_name.split('/')[-1]}.json", "w"),
              ensure_ascii=False, indent=1)
    print("\nraw answers, first 20:")
    for row, _lang, got in answers[:20]:
        shown = got.strip().replace("\n", " \\n ")[:70]
        print(f"  {row['child_ja']:10} want {row['parent_ja']:5} | {shown}")


if __name__ == "__main__":
    main()
