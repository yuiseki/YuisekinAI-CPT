#!/bin/bash
# Take every step of the experiment on this machine, small enough to finish.
#
# The point is not that the numbers mean anything: 200 documents and 5 steps
# teach a model nothing. The point is that the same code runs end to end, on
# whatever hardware is here, before any of it is trusted to a rented A100 with
# a compute-unit meter running.
#
#   bash scripts/dry_run.sh                 # CPU or GPU, whatever is present
#   MODEL=EleutherAI/pythia-160m bash scripts/dry_run.sh
set -euo pipefail

MODEL="${MODEL:-google/gemma-3-270m-it}"
OUT="${OUT:-runs/dry}"
DATA="${DATA:-data/dry}"
DOCS="${DOCS:-200}"
STEPS="${STEPS:-5}"
BLOCK="${BLOCK:-256}"
N_PROBE="${N_PROBE:-20}"

cd "$(dirname "$0")/.."
mkdir -p "$DATA" "$OUT"

echo "=== 1/4  corpus: the subset being learnt"
python3 src/corpus.py --config 20260901.ja --model "$MODEL" \
    --out "$DATA/ja.bin" --limit "$DOCS"

echo
echo "=== 2/4  corpus: the control, never trained on"
python3 src/corpus.py --config 20260901.en --model "$MODEL" \
    --out "$DATA/en.bin" --limit "$DOCS"

echo
echo "=== 3/4  probe before"
python3 src/probe.py --model "$MODEL" --n "$N_PROBE" --forms qa \
    --label "$MODEL before" --out "$OUT/probe-before.json"

echo
echo "=== 4/4  train"
python3 src/train.py --model "$MODEL" \
    --corpus "$DATA/ja.bin" --control "$DATA/en.bin" \
    --out "$OUT/model" --max-steps "$STEPS" --batch-size 1 --grad-accum 1 \
    --block-size "$BLOCK" --eval-every "$STEPS" --eval-iters 2

echo
echo "=== probe after"
python3 src/probe.py --model "$OUT/model" --n "$N_PROBE" --forms qa \
    --label "$MODEL after" --out "$OUT/probe-after.json"

echo
python3 - "$OUT" <<'PY'
import json, sys, os
out = sys.argv[1]
before = json.load(open(f"{out}/probe-before.json"))["scores"]["qa"]
after = json.load(open(f"{out}/probe-after.json"))["scores"]["qa"]
log = [json.loads(l) for l in open(f"{out}/model/log.jsonl")]
print("DRY RUN COMPLETE")
print(f"  probe   {before['accuracy']:.1%} -> {after['accuracy']:.1%} "
      f"on {before['n']} municipalities (chance {before['chance']:.1%})")
print(f"  held-out loss {log[0]['held_out']:.4f} -> {log[-1]['held_out']:.4f}")
if "control" in log[0]:
    print(f"  control  loss {log[0]['control']:.4f} -> {log[-1]['control']:.4f}")
print("\nFive steps teach nothing. What this shows is that every stage runs,")
print("reads its input, writes its output, and that the numbers move at all.")
PY
