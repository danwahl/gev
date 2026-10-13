#!/usr/bin/env bash
# Run the four SpeciEval tasks in every language (or those in SPECIEVAL_LANGUAGES) against a System One endpoint (a
# local kev.serve by default).
# Decision models are deterministic, so one epoch stands in for SpeciEval's ten.
#
#   scripts/specieval_decisions.sh <specieval checkout> <name> [endpoint] [reverse]
#
# Logs go to runs/gev/specieval/<name>, or <name>-rev with a fourth argument `reverse` (SpeciEval's negated statements,
# scored 8 - answer, English only since only English has them); score them with scripts/specieval_score.py. The
# endpoint gets KEV_API_KEY as its bearer token ("local" when unset), never the caller's OpenRouter key.
set -euo pipefail

specieval=$1
name=$2
endpoint=${3:-http://127.0.0.1:8009/v1/systemone}
reverse=$([ "${4:-}" = reverse ] && echo true || echo false)
out=$(pwd)/runs/gev/specieval/$name$([ "$reverse" = true ] && echo -rev || true)
languages=${SPECIEVAL_LANGUAGES:-"en de fr es zh ja pl pt nl ru it id ko ms th"}
[ "$reverse" = true ] && languages=en

cd "$specieval"
for lang in $languages; do
  OPENROUTER_API_KEY=${KEV_API_KEY:-local} uv run inspect eval \
    specieval/speciesism specieval/sentience specieval/attitude_meat specieval/attitude_seafood \
    --model "openrouter-decisions/$name" --model-base-url "$endpoint" \
    -T epochs=1 -T language="$lang" -T reverse="$reverse" --log-dir "$out" --display none
  echo "$name $lang done"
done
