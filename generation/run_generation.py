"""Send the prompts in a CSV to a locally loaded model and collect the replies.

Reads a CSV with an id column and a prompt column, writes one JSONL row per reply.
Shards split the rows, and ids already written are skipped, so an interrupted run
resumes with the same command.

  INPUT        prompt CSV
  OUTPUT       JSONL to append to
  COLUMN       name of the reply field, e.g. rewritten or pushback
  MODEL        hugging face model id
  TEMPERATURE  0 for faithful rewriting, 0.7 for varied pushbacks
  MAX_NEW      generated token cap
  LIMIT        only the first N rows, for a quick check

    ALLOW=0 SHARD=0/4 INPUT=... OUTPUT=... COLUMN=rewritten python generation/run_generation.py
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import Model, run_jobs, setup_logging

csv.field_size_limit(sys.maxsize)

INPUT = os.getenv("INPUT", "results/prompts/speaker_prompts.csv")
OUTPUT = os.getenv("OUTPUT", "results/generation/rewrites.jsonl")
COLUMN = os.getenv("COLUMN", "rewritten")
ID_COLUMN = os.getenv("ID_COLUMN", "job_id")
PROMPT_COLUMN = os.getenv("PROMPT_COLUMN", "prompt")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_NEW = int(os.getenv("MAX_NEW", "600"))
LIMIT = int(os.getenv("LIMIT", "0"))


def main():
    log = setup_logging(f"gen_{COLUMN}")

    with open(INPUT, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    jobs = [row for row in rows if row.get(PROMPT_COLUMN, "").strip()]
    log.info("%s: %d rows, %d with a prompt", INPUT, len(rows), len(jobs))
    if LIMIT:
        jobs = jobs[:LIMIT]

    model = Model()
    log.info("model %s, temperature %s, max_new %d", model.name, TEMPERATURE, MAX_NEW)

    def worker(job):
        reply = model.chat([{"role": "user", "content": job[PROMPT_COLUMN]}],
                           max_new=MAX_NEW, temperature=TEMPERATURE)
        return {ID_COLUMN: job[ID_COLUMN], "model": model.name,
                "temperature": TEMPERATURE, COLUMN: " ".join(reply.split())}

    run_jobs(jobs, worker, OUTPUT, ID_COLUMN, log, label=COLUMN)


if __name__ == "__main__":
    main()