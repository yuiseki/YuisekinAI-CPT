#!/bin/bash
# Take every step of the experiment on this machine, small enough to finish.
#
# The point is not that the numbers mean anything: a few hundred statements and
# five steps teach a model nothing. The point is that the same code runs end to
# end, on whatever hardware is here, before any of it is trusted to a rented
# A100 with a compute-unit meter running.
#
# Two probes run, and they measure opposite things. The hierarchy probe asks
# about places the corpus states outright, so it measures recall. The
# municipality probe asks about 1,134 Japanese places, one of which appears in
# the corpus, so it measures whether anything generalised. A rise in the first
# with no rise in the second is the expected result, not a disappointment, and
# the pair is here so that it cannot be mistaken for the other one.
#
#   bash scripts/dry_run.sh                 # CPU or GPU, whatever is present
#   MODEL=google/gemma-3-270m bash scripts/dry_run.sh
set -euo pipefail

MODEL="${MODEL:-google/gemma-3-270m-it}"
OUT="${OUT:-runs/dry}"
DATA="${DATA:-data/dry}"
DOCS="${DOCS:-4000}"
STEPS="${STEPS:-5}"
BLOCK="${BLOCK:-256}"
N_PROBE="${N_PROBE:-20}"
FORM="${FORM:-ja}"
PROBE_SET="${PROBE_SET:-../../_research/geo-triples-tokyo23/data/probe.parquet}"

cd "$(dirname "$0")/.."
mkdir -p "$DATA" "$OUT"

echo "=== 1/5  corpus: the subset being learnt"
python3 src/corpus.py --dataset yuiseki/geo-triples-tokyo23 --config cpt \
    --where "form=$FORM" --model "$MODEL" \
    --out "$DATA/geo.bin" --limit "$DOCS"

echo
echo "=== 2/5  corpus: the control, never trained on"
python3 src/corpus.py --dataset yuiseki/wikipedia-geotagged \
    --config 20260901.en --model "$MODEL" \
    --out "$DATA/en.bin" --limit "$DOCS"

echo
echo "=== 3/5  probes before"
python3 src/hierarchy_probe.py --model "$MODEL" --n "$N_PROBE" \
    --set "$PROBE_SET" --label "$MODEL before" \
    --out "$OUT/hierarchy-before.json"
python3 src/probe.py --model "$MODEL" --n "$N_PROBE" --forms qa \
    --label "$MODEL before" --out "$OUT/probe-before.json"

echo
echo "=== 4/5  train"
python3 src/train.py --model "$MODEL" \
    --corpus "$DATA/geo.bin" --control "$DATA/en.bin" \
    --out "$OUT/model" --max-steps "$STEPS" --batch-size 1 --grad-accum 1 \
    --block-size "$BLOCK" --eval-every "$STEPS" --eval-iters 2

echo
echo "=== 5/5  probes after"
python3 src/hierarchy_probe.py --model "$OUT/model" --n "$N_PROBE" \
    --set "$PROBE_SET" --label "$MODEL after" \
    --out "$OUT/hierarchy-after.json"
python3 src/probe.py --model "$OUT/model" --n "$N_PROBE" --forms qa \
    --label "$MODEL after" --out "$OUT/probe-after.json"

echo
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]


def load(name):
    return json.load(open(f"{out}/{name}.json"))["scores"]


hb, ha = load("hierarchy-before"), load("hierarchy-after")
pb, pa = load("probe-before")["qa"], load("probe-after")["qa"]
log = [json.loads(l) for l in open(f"{out}/model/log.jsonl")]

print("DRY RUN COMPLETE")
print("  recall, on places the corpus states:")
for lang in sorted(hb):
    for level in sorted(hb[lang]):
        b, a = hb[lang][level], ha[lang][level]
        print(f"    [{lang}] {level:18} {b['accuracy']:6.1%} -> "
              f"{a['accuracy']:6.1%}  on {b['n']} "
              f"(chance {b['chance']:.1%})")
print("  generalisation, on 1,134 places it does not:")
print(f"    {pb['accuracy']:6.1%} -> {pa['accuracy']:6.1%}  on {pb['n']} "
      f"municipalities (chance {pb['chance']:.1%})")
print(f"  held-out loss {log[0]['held_out']:.4f} -> {log[-1]['held_out']:.4f}")
if "control" in log[0]:
    print(f"  control  loss {log[0]['control']:.4f} -> {log[-1]['control']:.4f}")
print("\nFive steps teach nothing. What this shows is that every stage runs,")
print("reads its input, writes its output, and that the numbers move at all.")
PY
