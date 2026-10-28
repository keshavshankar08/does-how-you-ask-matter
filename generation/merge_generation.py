"""Put the generated replies back into the prompt CSV, so the next stage can read one file.

  INPUT    the prompt CSV
  OUTPUT   the JSONL base name the shards were written under
  CSV      where to write the filled CSV
  COLUMN   the reply field

The prompt column is left out of the CSV. It is the instruction block sent to the
model, the same text on every row apart from the slots, rebuilt by 01 and 02, and
most of the file size.

    INPUT=... OUTPUT=... CSV=... COLUMN=rewritten python generation/merge_generation.py
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_shards

csv.field_size_limit(sys.maxsize)

INPUT = os.getenv("INPUT", "results/prompts/speaker_prompts.csv")
OUTPUT = os.getenv("OUTPUT", "results/generation/rewrites.jsonl")
CSV_OUT = os.getenv("CSV", "results/generation/rewrites.csv")
COLUMN = os.getenv("COLUMN", "rewritten")
ID_COLUMN = os.getenv("ID_COLUMN", "job_id")


def main():
    with open(INPUT, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    replies = {record[ID_COLUMN]: record[COLUMN] for record in load_shards(OUTPUT)}

    wanted = [row for row in rows if row.get("prompt", "").strip()]
    missing = [row[ID_COLUMN] for row in wanted if row[ID_COLUMN] not in replies]
    print(f"{len(rows)} rows, {len(replies)} replies, {len(missing)} still missing")
    if missing:
        print("first missing:", missing[:5])

    columns = [c for c in rows[0] if c != "prompt"]
    if COLUMN not in columns:
        columns.append(COLUMN)
    Path(CSV_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if row[ID_COLUMN] in replies:
                row[COLUMN] = replies[row[ID_COLUMN]]
            writer.writerow(row)
    print(f"wrote {CSV_OUT}")


if __name__ == "__main__":
    main()