#!/usr/bin/env bash
# Step 2. Build nine pushbacks per turn 1 input and generate them.
# Needs results/generation/rewrites.csv from run_inputs.sh.
#
#   GPUS="4,5,6,7" MODEL=Qwen/Qwen3-32B nohup bash run_counterfactuals.sh > logs/run_cf.out 2>&1 &

set -e

PYTHON="${PYTHON:-python}"
IFS=',' read -r -a GPU <<< "${GPUS:-0}"
N=${#GPU[@]}
export HF_HOME="${HF_HOME:-/mnt/data1/workspace/kes298/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # weights are cached, skip the hub check
export TOKENIZERS_PARALLELISM=false

mkdir -p logs results

started=$(date +%s)
say() { printf '[%5ds] %s\n' "$(( $(date +%s) - started ))" "$*"; }

if [ ! -f results/generation/rewrites.csv ]; then
  say "FATAL: results/generation/rewrites.csv is missing, run run_inputs.sh first"
  exit 1
fi

say "counterfactuals | gpus ${GPU[*]} ($N) | model ${MODEL:-unset}"

"$PYTHON" generation/02_counterfactual.py

export INPUT=results/prompts/pushback_prompts.csv
export OUTPUT=results/generation/pushbacks.jsonl
export COLUMN=pushback
export TEMPERATURE=0.7
export MAX_NEW=300

for i in $(seq 0 $((N - 1))); do
  say "  start  gpu${GPU[$i]}  shard $i/$N"
  ALLOW="${GPU[$i]}" SHARD="$i/$N" "$PYTHON" generation/run_generation.py \
    > "logs/counterfactuals_s${i}of${N}.out" 2>&1 &
done
wait

CSV=results/generation/pushbacks.csv "$PYTHON" generation/merge_generation.py
say "counterfactuals done -> results/generation/pushbacks.csv"