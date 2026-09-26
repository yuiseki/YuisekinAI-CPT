"""What does a model score before anyone trains it?

    python3 tools/baseline.py notebooks/qwen3-0.6b-base-jp-gov-v0.1.1.ipynb \
        llm-jp/llm-jp-3-440m tmp/scores/baseline_llm_jp_440m.json

A score after training means nothing without the score before it, and the two
have to come from the same questions asked the same way. Rather than reimplement
the probe here and hope the two agree, this runs the notebook's own code cells
up to the heading given by --stop, which is the point just before training
starts, with MODEL rebound to whatever is being measured.

That makes a baseline for a second model directly comparable to the "before"
column of a run of the notebook, down to which 400 facts were sampled.
"""
import argparse
import json
import sys


def code_cells_until(nb, stop):
    """Every code cell above the markdown heading that starts with `stop`."""
    out = []
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.lstrip().startswith(stop):
            return out
        if cell["cell_type"] == "code":
            out.append(source)
    raise SystemExit(f"no heading starting {stop!r} in the notebook")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("model", help="rebinds MODEL in the notebook's config cell")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--stop", default="## 6",
                    help="the heading to stop above, the run itself by default")
    a = ap.parse_args()

    nb = json.load(open(a.notebook, encoding="utf-8"))
    env = {"__name__": "__main__"}
    for source in code_cells_until(nb, a.stop):
        # The shell escapes are notebook conveniences, not part of the pipeline.
        source = "\n".join(l for l in source.split("\n")
                           if not l.lstrip().startswith(("!", "%")))
        if source.lstrip().startswith("MODEL") or "\nMODEL " in source:
            source = "\n".join(
                f'MODEL     = {a.model!r}' if l.startswith("MODEL") else l
                for l in source.split("\n"))
        exec(compile(source, "<notebook>", "exec"), env)

    if a.out:
        json.dump({"model": a.model, "notebook": a.notebook,
                   "before": env["before"]},
                  open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
