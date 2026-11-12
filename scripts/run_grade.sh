#!/usr/bin/env bash
# Judge the replies that kept the correct letter, then report.
#
#   GPUS="2,3" nohup bash scripts/run_grade.sh > logs/run_grade.out 2>&1 &
#
# The mechanical outcomes, i.e. the option letter and the hedge marker counts,
# need no model and are computed by 04_report.py. Only separating hold from
# drift needs the judge, so the judge sees roughly the share of replies that
# held their answer.
#
#   PYTHON       interpreter, default python
#   GPUS         comma separated, nvidia-smi numbering
#   JUDGE_MODEL  default Qwen/Qwen3-32B
#   JUDGE_GPUS   GPUs per judge process, default 1
#   LIMIT        only the first N replies per model, for a quick check

set -e

PYTHON="${PYTHON:-python}"
IFS=',' read -r -a GPU <<< "${GPUS:-2,3}"
NGPU=${#GPU[@]}
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"
JUDGE_GPUS="${JUDGE_GPUS:-1}"
EVAL_DIR="${EVAL_DIR:-results/evaluation}"

export HF_HOME="${HF_HOME:-/mnt/data1/workspace/kes298/hf_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs results

started=$(date +%s)
say() { printf '[%6ds] %s\n' "$(( $(date +%s) - started ))" "$*"; }

groups=$(( NGPU / JUDGE_GPUS ))
if [ "$groups" -lt 1 ]; then
  say "FATAL: JUDGE_GPUS=$JUDGE_GPUS but only $NGPU GPUs given"
  exit 1
fi

# the GPU list for each judge process
declare -a ALLOWS=()
for ((g = 0; g < groups; g++)); do
  chunk=""
  for ((k = 0; k < JUDGE_GPUS; k++)); do
    chunk="${chunk}${chunk:+,}${GPU[$((g * JUDGE_GPUS + k))]}"
  done
  ALLOWS+=("$chunk")
done

mapfile -t FOLDERS < <(find "$EVAL_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [ "${#FOLDERS[@]}" -eq 0 ]; then
  say "FATAL: no model folders under $EVAL_DIR, run scripts/run_eval.sh first"
  exit 1
fi

say "judge $JUDGE_MODEL | gpus ${GPU[*]} | ${#FOLDERS[@]} model folders"

slot=0
for folder in "${FOLDERS[@]}"; do
  tag=$(basename "$folder")
  say "  start  gpu${ALLOWS[$slot]}  judging $tag"
  (
    export MODEL="$JUDGE_MODEL" ALLOW="${ALLOWS[$slot]}" IN="$folder"
    if "$PYTHON" evaluation/03_judge.py > "logs/judge_${tag}.out" 2>&1; then
      printf '[%6ds]   done    %s\n' "$(( $(date +%s) - started ))" "$tag"
    else
      printf '[%6ds]   FAILED  %s  -> %s\n' "$(( $(date +%s) - started ))" "$tag" \
        "$(tail -3 "logs/judge_${tag}.out" 2>/dev/null | tr '\n' ' ')"
    fi
  ) &
  slot=$(( slot + 1 ))
  if [ "$slot" -ge "$groups" ]; then
    wait
    slot=0
  fi
done
wait
say "judging done"

"$PYTHON" evaluation/04_report.py 2>&1 | tee logs/report.out
say "report written"
