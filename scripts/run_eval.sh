#!/usr/bin/env bash
# Run turn 1 and turn 2 for every target model.
#
#   GPUS="2,3,4,5" nohup bash scripts/run_eval.sh > logs/run_eval.out 2>&1 &
#
# Each model in MODELS may carry a GPU count after a colon, e.g. "...-70B:2".
# Default is 1. Models are run in waves, smallest GPU need first, so the small
# models finish before anything waits on a 70B. Within a wave the GPUs are split
# into groups and one model runs per group, with device_map spreading a model
# across its group.
#
# Every stage appends to its own shard file and skips ids it has already written,
# so rerunning the same command after an interruption continues where it stopped.
# Reassigning GPUs between runs is safe for the same reason.
#
#   PYTHON  interpreter, default python
#   GPUS    comma separated, nvidia-smi numbering, at least 2
#   MODELS  comma separated model ids, optionally model:gpus
#   LIMIT   only the first N questions per model, for a quick check

set -e

PYTHON="${PYTHON:-python}"
IFS=',' read -r -a GPU <<< "${GPUS:-2,3}"
NGPU=${#GPU[@]}

MODELS="${MODELS:-\
google/gemma-3-4b-it,\
google/gemma-3-12b-it,\
google/gemma-3-27b-it,\
google/medgemma-27b-text-it,\
meta-llama/Llama-3.2-3B-Instruct,\
meta-llama/Llama-3.1-8B-Instruct,\
mistralai/Mistral-7B-Instruct-v0.3,\
BioMistral/BioMistral-7B,\
m42-health/Llama3-Med42-8B,\
meta-llama/Llama-3.3-70B-Instruct:2,\
m42-health/Llama3-Med42-70B:2}"

export HF_HOME="${HF_HOME:-/mnt/data1/workspace/kes298/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # set 1 once every model is cached
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results

started=$(date +%s)
say() { printf '[%6ds] %s\n' "$(( $(date +%s) - started ))" "$*"; }

if [ "$NGPU" -lt 2 ]; then
  say "FATAL: need at least 2 GPUs, got $NGPU"
  exit 1
fi

# model:gpus -> two arrays
declare -a NAME NEED
IFS=',' read -r -a ENTRIES <<< "$MODELS"
for entry in "${ENTRIES[@]}"; do
  NAME+=("${entry%%:*}")
  if [[ "$entry" == *:* ]]; then NEED+=("${entry##*:}"); else NEED+=(1); fi
done

say "gpus ${GPU[*]} ($NGPU) | ${#NAME[@]} models"

# run_model <model> <gpus for this model, comma separated> <group index> <groups>
run_model() {
  local model=$1 allow=$2 index=$3 groups=$4
  local tag; tag=$(basename "$model" | tr '[:upper:]' '[:lower:]' | tr '.' '-')
  say "  start  gpu${allow}  $tag"
  (
    export MODEL="$model" ALLOW="$allow" SHARD="${index}/${groups}"
    if "$PYTHON" evaluation/01_turn1.py > "logs/turn1_${tag}_s${index}of${groups}.out" 2>&1 \
    && "$PYTHON" evaluation/02_conversation.py > "logs/conv_${tag}_s${index}of${groups}.out" 2>&1; then
      printf '[%6ds]   done    gpu%s  %s\n' "$(( $(date +%s) - started ))" "$allow" "$tag"
    else
      printf '[%6ds]   FAILED  gpu%s  %s  -> %s\n' "$(( $(date +%s) - started ))" \
        "$allow" "$tag" "$(tail -3 "logs/conv_${tag}_s${index}of${groups}.out" 2>/dev/null | tr '\n' ' ')"
    fi
  ) &
}

# one wave per distinct GPU requirement, ascending
for size in $(printf '%s\n' "${NEED[@]}" | sort -un); do
  groups=$(( NGPU / size ))
  if [ "$groups" -lt 1 ]; then
    say "skipping ${size}-GPU models, only $NGPU GPUs available"
    continue
  fi
  say "=== wave: models needing $size gpu(s), $groups running at a time"

  # the GPU list for each group in this wave
  declare -a ALLOWS=()
  for ((g = 0; g < groups; g++)); do
    chunk=""
    for ((k = 0; k < size; k++)); do
      chunk="${chunk}${chunk:+,}${GPU[$((g * size + k))]}"
    done
    ALLOWS+=("$chunk")
  done

  # fill the groups, waiting for the whole batch before starting the next
  slot=0
  for i in "${!NAME[@]}"; do
    [ "${NEED[$i]}" = "$size" ] || continue
    run_model "${NAME[$i]}" "${ALLOWS[$slot]}" 0 1
    slot=$(( slot + 1 ))
    if [ "$slot" -ge "$groups" ]; then
      wait
      slot=0
    fi
  done
  wait
  say "=== wave for $size gpu(s) done"
done

say "all models done"
