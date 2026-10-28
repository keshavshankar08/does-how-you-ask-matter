#!/usr/bin/env bash
# Step 1. Build the five turn 1 inputs per question and generate the four rewrites.
#
#   GPUS="4,5,6,7" MODEL=Qwen/Qwen3-32B nohup bash run_inputs.sh > logs/run_inputs.out 2>&1 &
#
# One process per GPU, each taking every Nth row. Each writes its own shard file
# and skips ids it already wrote, so rerunning the same command resumes.
#
#   PYTHON  interpreter, default python
#   GPUS    comma separated, nvidia-smi numbering, default "0"
#   MODEL   hugging face model id
#   LIMIT   only the first N rows, for a quick check

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

say "inputs | gpus ${GPU[*]} ($N) | model ${MODEL:-unset}"

"$PYTHON" generation/01_framing_rewrite.py results/prompts/speaker_prompts.csv

export INPUT=results/prompts/speaker_prompts.csv
export OUTPUT=results/generation/rewrites.jsonl
export COLUMN=rewritten
export TEMPERATURE=0
export MAX_NEW=800

for i in $(seq 0 $((N - 1))); do
  say "  start  gpu${GPU[$i]}  shard $i/$N"
  ALLOW="${GPU[$i]}" SHARD="$i/$N" "$PYTHON" generation/run_generation.py \
    > "logs/inputs_s${i}of${N}.out" 2>&1 &
done
wait

CSV=results/generation/rewrites.csv "$PYTHON" generation/merge_generation.py
say "inputs done -> results/generation/rewrites.csv"