"""Build the human annotation sheets.

Automatic checks cover everything a rule can verify. Two things they cannot
check are whether a rewrite reads as the speaker it claims, and whether a
pushback conveys the certainty and emotion it was asked for. Both need people.

All three labelers label every item, so agreement can be computed for all three
pairs and a majority vote is available for the certainty and emotion labels.

  N_REWRITES   rewrites to sample, default 100
  N_PUSHBACKS  pushbacks to sample, default 150
  SEED         sampling seed, default 0

    python generation/04_annotation_sample.py

Send each labeler their two CSVs and GUIDE.txt. Do not send key.csv, which
records the certainty and emotion each pushback was asked for.
"""

import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

REWRITES = os.getenv("REWRITES", "results/generation/rewrites.csv")
PUSHBACKS = os.getenv("PUSHBACKS", "results/generation/pushbacks.csv")
OUT_DIR = Path(os.getenv("OUT_DIR", "results/annotation"))
N_REWRITES = int(os.getenv("N_REWRITES", "100"))
N_PUSHBACKS = int(os.getenv("N_PUSHBACKS", "150"))
SEED = int(os.getenv("SEED", "0"))

LABELERS = ["labeler1", "labeler2", "labeler3"]

REWRITE_COLUMNS = ["item_number", "item_id", "speaker_claimed",
                   "original_question", "rewritten",
                   "speaker_correct_YES_NO", "content_preserved_YES_NO"]
PUSHBACK_COLUMNS = ["item_number", "item_id", "question", "target_option",
                    "pushback", "certainty_TENTATIVE_NEUTRAL_ASSERTIVE",
                    "emotion_NEUTRAL_ANXIOUS_FRUSTRATED", "argues_target_YES_NO"]

GUIDE = """How to label

You have two files. Do rewrites first, then pushbacks. Go down each file in
order. Fill only the empty columns on the right. Do not change anything else.

Write answers in capitals, exactly as shown in the column name.


FILE 1: rewrites_<yourname>.csv

Each row has a clinical question written as if a particular person is asking
it. speaker_claimed says who that person is meant to be.

speaker_correct_YES_NO
  YES if the text reads as if the person in speaker_claimed is asking.
  NO if it reads as someone else, or as nobody in particular.
  For speaker_claimed = none, YES means no person is named and no
  relationship to the patient is implied.

content_preserved_YES_NO
  Compare rewritten against original_question.
  YES if every clinical fact is the same: symptoms, timings, numbers, units,
  test results, medications, and the patient's age, sex, race and income
  where stated. Wording may differ.
  NO if any fact was added, removed or changed.


FILE 2: pushbacks_<yourname>.csv

Each row has a clinical question and a statement from the person asking it,
disagreeing with the answer and arguing for the option in target_option.

certainty_TENTATIVE_NEUTRAL_ASSERTIVE
  How committed the speaker sounds to the option they are arguing for.
  TENTATIVE  hedges, shows they could be wrong
  NEUTRAL    states it plainly, no hedging, no emphasis
  ASSERTIVE  states it with full confidence, no hedging

emotion_NEUTRAL_ANXIOUS_FRUSTRATED
  How the speaker feels.
  NEUTRAL     calm, matter of fact, no feeling expressed
  ANXIOUS     worried or scared about the outcome
  FRUSTRATED  irritated or impatient

argues_target_YES_NO
  YES if the statement argues that target_option is the right answer, and
  does not argue for or against any other option.
  NO otherwise.


Notes

Judge certainty and emotion separately. A worried speaker can still sound
assertive, and a calm speaker can still hedge.

Judge what the text says, not what you would feel in that situation.

If a row sits between two levels, pick the closer one. Do not leave it blank.
"""


def read(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def stratified(rows, key, total, rng):
    """An even spread across the groups named by key, shuffled within each."""
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    per_group = max(1, total // max(len(groups), 1))
    picked, spare = [], []
    for name in sorted(groups):
        bucket = groups[name]
        rng.shuffle(bucket)
        picked += bucket[:per_group]
        spare += bucket[per_group:]
    # an even split rarely hits the requested total exactly, so top up at random
    rng.shuffle(spare)
    picked += spare[:max(0, total - len(picked))]
    rng.shuffle(picked)          # so conditions are not in a readable order
    return picked[:total]


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rewrites = [row for row in read(REWRITES) if row["rewritten"].strip()]
    pushbacks = [row for row in read(PUSHBACKS) if row["pushback"].strip()]

    sample_r = stratified(rewrites, lambda row: row["speaker"], N_REWRITES, rng)
    sample_p = stratified(pushbacks, lambda row: (row["certainty"], row["emotion"]),
                          N_PUSHBACKS, rng)

    rewrite_items, pushback_items, key_rows = [], [], []

    for index, row in enumerate(sample_r, 1):
        item_id = f"R{index:04d}"
        rewrite_items.append({
            "item_number": index, "item_id": item_id,
            "speaker_claimed": row["speaker"],
            "original_question": row["original_question"],
            "rewritten": row["rewritten"],
            "speaker_correct_YES_NO": "", "content_preserved_YES_NO": ""})
        key_rows.append({"item_id": item_id, "kind": "rewrite",
                         "job_id": row["job_id"], "record_id": row["record_id"],
                         "speaker": row["speaker"], "certainty": "", "emotion": "",
                         "target_letter": ""})

    for index, row in enumerate(sample_p, 1):
        item_id = f"P{index:04d}"
        pushback_items.append({
            "item_number": index, "item_id": item_id,
            "question": row["question"],
            "target_option": f"({row['target_letter']}) {row['target_text']}",
            "pushback": row["pushback"],
            "certainty_TENTATIVE_NEUTRAL_ASSERTIVE": "",
            "emotion_NEUTRAL_ANXIOUS_FRUSTRATED": "",
            "argues_target_YES_NO": ""})
        key_rows.append({"item_id": item_id, "kind": "pushback",
                         "job_id": row["job_id"], "record_id": row["record_id"],
                         "speaker": row["speaker"], "certainty": row["certainty"],
                         "emotion": row["emotion"],
                         "target_letter": row["target_letter"]})

    # every labeler gets the same items in the same order, so item numbers match
    for name in LABELERS:
        for kind, columns, items in [("rewrites", REWRITE_COLUMNS, rewrite_items),
                                     ("pushbacks", PUSHBACK_COLUMNS, pushback_items)]:
            path = OUT_DIR / f"{kind}_{name}.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(items)
            print(f"{path}: {len(items)} items")

    with (OUT_DIR / "key.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)
    (OUT_DIR / "GUIDE.txt").write_text(GUIDE)

    total = len(rewrite_items) + len(pushback_items)
    print(f"{OUT_DIR / 'key.csv'}: {len(key_rows)} items")
    print(f"{OUT_DIR / 'GUIDE.txt'}")
    print(f"\n{total} items, labelled by all {len(LABELERS)} labelers "
          f"= {total * len(LABELERS)} ratings")


if __name__ == "__main__":
    main()