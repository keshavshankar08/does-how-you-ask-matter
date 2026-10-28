# does-how-you-ask-matter: generation

Builds the two-turn prompts. Evaluation is separate and not here yet.

## Setup

    conda create -n dhyam python=3.11 -y
    conda activate dhyam
    pip install -r requirements.txt

Run from the repo root.

## Step 1: turn 1 inputs

Five per question: neutral plus patient, doctor, friend, relative.
801 x 5 = 4,005 rows. The neutral row is the question unchanged, so only 3,204
need the model.

    GPUS="4,5,6,7" MODEL=Qwen/Qwen3-32B nohup bash run_inputs.sh > logs/run_inputs.out 2>&1 &

Output: results/generation/rewrites.csv

## Step 2: turn 2 counterfactuals

Nine per input: 3 certainty x 3 emotion. Same speaker as the input it follows.
4,005 x 9 = 36,045 rows. Needs step 1 to have finished.

    GPUS="4,5,6,7" MODEL=Qwen/Qwen3-32B nohup bash run_counterfactuals.sh > logs/run_cf.out 2>&1 &

Output: results/generation/pushbacks.csv

## How the pushback is built

One incorrect option is picked per question from the answer key and fixed, so all
nine pushbacks for a question argue for the same wrong option and only the framing
changes. Generation does not depend on what any model answered.

## Parallelism and resume

One process per GPU, each taking every Nth row. Each writes its own shard file
and skips ids it already wrote, so rerunning the same command resumes where it
stopped. Logs are one file per step and shard in logs/.

## Env vars

    PYTHON       interpreter, default python
    GPUS         comma separated, nvidia-smi numbering
    MODEL        hugging face model id
    LIMIT        only the first N rows, for a quick check
    ALLOW        set by the run scripts, the GPU a process may use
    SHARD        set by the run scripts, "i/n"

## Files

    generation/01_framing_rewrite.py   builds turn 1 prompts
    generation/02_counterfactual.py    builds turn 2 prompts
    generation/run_generation.py       fills a prompt CSV from the model
    generation/merge_generation.py     folds shards back into one CSV
    common.py                          GPU pinning, model loading, sharding, resume