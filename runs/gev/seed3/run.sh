#!/usr/bin/env bash
# Seed 3 of the AISI arm and the harm-pairs arm (otherwise identical to runs/gev/aw/train.sh and runs/gev/harm/run.sh), then every read.
SPEC=/home/dan/specieval
cd /home/dan/gev
export HF_HOME=/data/cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . ./.env.local; set +a
for arm in "aisi:evals/gev/aisi-animal-v1/train.jsonl" "harm:runs/gev/harm/train-mix.jsonl"; do
  name=${arm%%:*}; data=${arm##*:}; out=runs/gev/seed3/kev4b-v7-$name
  uv run --no-sync python -m kev.train --suite evals/v7/decision-v7 --data $data --replay 1000000 \
    --base Qwen/Qwen3.5-4B-Base --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b --epochs 2 --lr 5e-5 --batch 2 --accum 4 \
    --dtype bf16 --checkpointing 1 --p_none_pair 0.25 --seed 3 --device cuda --out $out || { echo "$name TRAIN FAILED"; continue; }
  echo "$name TRAINDONE"
  for suite in evals/gev/aisi-harm-v1 evals/v7/decision-v7 evals/v4/transfer-v4 evals/gev/aisi-animal-v1 evals/gev/speciesismbench-v1 evals/gev/tac-v1; do
    KEV_TEMPERATURE=1.0 uv run --no-sync python -m kev.benchmark --run $out --suite $suite --out runs/gev/seed3/bench-$name-$(basename $suite)
  done
  KEV_TEMPERATURE=1.0 uv run --no-sync --extra serve python -m kev.serve --run $out --port 8009 > runs/gev/seed3/serve-$name.log 2>&1 &
  pid=$!
  until curl -sf localhost:8009/v1/models >/dev/null; do kill -0 $pid 2>/dev/null || { echo "$name SERVER DIED"; continue 2; }; sleep 3; done
  scripts/specieval_decisions.sh $SPEC gev-$name-s3 2>&1 | grep -E "Error|Traceback" | head -3
  scripts/specieval_decisions.sh $SPEC gev-$name-s3 http://127.0.0.1:8009/v1/systemone reverse 2>&1 | grep -E "Error|Traceback" | head -3
  kill $pid; wait $pid 2>/dev/null
  echo "$name ARMDONE"
done
echo ALLDONE
