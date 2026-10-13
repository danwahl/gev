#!/usr/bin/env bash
# Gev's recipe at each seed: Granite 4.1 8B base, instruction layout, decision-v7 replayed whole plus the ETHICS-Animals
# training partition, trained with deterministic kernels (a rerun gives the same weights), then every read: decision-v7,
# transfer-v4, ETHICS, ETHICS-Animals, SpeciesismBench, AISI, and SpeciEval in every language (plus its reversed
# statements) through a local kev.serve.
#
#   scripts/gev.sh NAME SEED [SEED...]
#
# Writes runs/gev/NAME-sSEED/{ckpt,bench-*} and runs/gev/specieval/NAME-sSEED{,-rev}. Needs a SpeciEval checkout in
# SPECIEVAL (default ../specieval) and the SpeciesismBench and AISI partitions, which are not in git: build them first with
# scripts/build_gev_speciesismbench.py and scripts/build_gev_aisi_{animal,harm}.py.
# DATA replaces the ETHICS-Animals training file; with WANDB_API_KEY set, training logs to Weights & Biases as
# gev-NAME-sSEED. The server listens on PORT (default 8019). A rerun skips a finished checkpoint, a read with a report.json and a
# SpeciEval read whose log ends in "done"; training continues from its last resume point (written when the job is
# preempted) and anything else unfinished starts again.
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
specieval=$(realpath "${SPECIEVAL:-../specieval}")
port=${PORT:-8019}
data=${DATA:-evals/gev/ethics-animals-v1/train.jsonl}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pid=
trap '[ -n "$pid" ] && kill $pid 2> /dev/null' EXIT
trap 'exit 143' INT TERM
for seed in "$@"; do
  run=$name-s$seed; dir=runs/gev/$run; mkdir -p "$dir"; failed=
  if [ ! -f "$dir/ckpt/head.pt" ]; then
    WANDB_NAME=gev-$run uv run --no-sync python -m kev.train --suite evals/v7/decision-v7 \
      --data "$data" --replay 1000000 \
      --base ibm-granite/granite-4.1-8b-base --base_revision cbc32f94fbf8179ca4b22b664831f6a73965bb16 \
      --layout instruction --weights_dtype bf16 --epochs 2 --lr 5e-5 --batch 2 --accum 4 --dtype bf16 --checkpointing 1 \
      --p_none_pair 0.25 --seed "$seed" --deterministic 1 --device cuda --save_on_sigterm 1 --resume 1 --out "$dir/ckpt" || { echo "$run FAILED train"; continue; }
  fi
  for suite in evals/v7/decision-v7 evals/v4/transfer-v4 evals/gev/ethics-v1 evals/gev/ethics-animals-v1 \
               evals/gev/speciesismbench-v1 evals/gev/aisi-animal-v1 evals/gev/aisi-harm-v1; do
    out=$dir/bench-$(basename "$suite")
    [ -f "$out/report.json" ] && continue
    rm -rf "$out"
    KEV_TEMPERATURE=1.0 uv run --no-sync python -m kev.benchmark --run "$dir/ckpt" --suite "$suite" --out "$out" \
      > "$out.log" 2>&1 || { echo "$run FAILED bench $suite"; failed=1; }
  done
  [ "$(tail -n 1 "$dir/specieval.log" 2> /dev/null)" = "$run en done" ] && { [ -z "$failed" ] && echo "$run DONE"; continue; }
  curl -s "localhost:$port" > /dev/null && { echo "$run FAILED serve: port $port is in use"; continue; }
  KEV_TEMPERATURE=1.0 uv run --no-sync --extra serve python -m kev.serve --run "$dir/ckpt" --port "$port" > "$dir/serve.log" 2>&1 &
  pid=$!
  until curl -sf -H "Authorization: Bearer ${KEV_API_KEY:-local}" "localhost:$port/v1/models" > /dev/null; do
    kill -0 $pid 2> /dev/null || { echo "$run FAILED serve"; continue 2; }; sleep 3
  done
  rm -rf "runs/gev/specieval/$run" "runs/gev/specieval/$run-rev"
  scripts/specieval_decisions.sh "$specieval" "$run" "http://127.0.0.1:$port/v1/systemone" > "$dir/specieval.log" 2>&1 &&
    scripts/specieval_decisions.sh "$specieval" "$run" "http://127.0.0.1:$port/v1/systemone" reverse >> "$dir/specieval.log" 2>&1 ||
    { echo "$run FAILED specieval"; failed=1; }
  kill $pid; wait $pid 2> /dev/null; pid=
  [ -z "$failed" ] && echo "$run DONE"
done
