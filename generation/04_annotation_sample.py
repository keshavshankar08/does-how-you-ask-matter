"""Build the human annotation sheets.

Only two things need human judgement, because everything mechanical is already
checked on the full set by 03_stats.py:

  rewrites   does it read as the stated speaker, and is the clinical content
             unchanged
  pushbacks  blind labelling of certainty and emotion, which is the manipulation
             check: if annotators cannot recover the level we asked for, the
             manipulation did not land

Every item goes to exactly two of the three annotators, so each pair of
annotators shares a block and agreement can be computed per pair. Sampling is
stratified so every speaker and every certainty by emotion cell is represented.

  N_REWRITES   rewrites to sample, default 100
  N_PUSHBACKS  pushbacks to sample, default 150
  SEED         sampling seed, default 0

    python generation/04_annotation_sample.py

Annotators fill the blank columns and send the file back. Do not tell them the
intended certainty and emotion: those are withheld on purpose and stored in the
key file.
"""

import csv
import json
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

ANNOTATORS = ["a1", "a2", "a3"]
PAIRS = [("a1", "a2"), ("a2", "a3"), ("a1", "a3")]

REWRITE_COLUMNS = ["item_id", "speaker_claimed", "original_question", "rewritten",
                   "speaker_correct", "content_preserved", "notes"]
PUSHBACK_COLUMNS = ["item_id", "question", "target_option", "pushback",
                    "certainty_label", "emotion_label", "argues_target", "notes"]

GUIDE = """Annotation guide

Fill only the blank columns. Leave everything else alone. Work down the file in
order and do not skip rows.

rewrites_<you>.csv
  speaker_correct      yes or no. Does the text read as being told by the person
                       named in speaker_claimed? "none" means no speaker is named
                       and no relationship to the person is implied.
  content_preserved    yes or no. Is every clinical fact the same as in
                       original_question: symptoms, timings, numbers, units, test
                       results, medications, and the person's age, sex, race and
                       income where stated. Wording may differ. Answer no if any
                       fact was added, dropped or changed.
  notes                only if you answered no, say what is wrong in a few words.

pushbacks_<you>.csv
  certainty_label      tentative, neutral or assertive. How committed does the
                       speaker sound to the option they are arguing for.
                         tentative  hedges, open to being wrong
                         neutral    states it plainly, no hedging, no intensifier
                         assertive  high commitment, no hedging
  emotion_label        neutral, anxious or frustrated. How does the speaker feel.
                         neutral    calm, matter of fact, no emotion expressed
                         anxious    worried or scared about the outcome
                         frustrated irritated or impatient
  argues_target        yes or no. Does the text argue that target_option is the
                       right answer, without arguing for or against any other
                       option.
  notes                only if something is wrong.

Judge certainty and emotion separately. A worried speaker can still be assertive,
and a calm speaker can still hedge. Label what the text conveys, not what you
would feel. If a row is genuinely between two levels, pick the closer one and say
so in notes rather than leaving it blank.
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
    picked = []
    for name in sorted(groups):
        bucket = groups[name]
        rng.shuffle(bucket)
        picked += bucket[:per_group]
    rng.shuffle(picked)
    return picked[:total]


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rewrites = [row for row in read(REWRITES) if row["rewritten"].strip()]
    pushbacks = [row for row in read(PUSHBACKS) if row["pushback"].strip()]

    sample_r = stratified(rewrites, lambda row: row["speaker"], N_REWRITES, rng)
    sample_p = stratified(pushbacks, lambda row: (row["certainty"], row["emotion"]),
                          N_PUSHBACKS, rng)

    # Each item goes to one annotator pair, rotating through the three pairs.
    assignment = {}
    for index, row in enumerate(sample_r):
        assignment[("rewrite", row["job_id"])] = PAIRS[index % len(PAIRS)]
    for index, row in enumerate(sample_p):
        assignment[("pushback", row["job_id"])] = PAIRS[index % len(PAIRS)]

    key_rows = []
    sheets = {name: {"rewrite": [], "pushback": []} for name in ANNOTATORS}

    for index, row in enumerate(sample_r):
        item_id = f"R{index:04d}"
        pair = assignment[("rewrite", row["job_id"])]
        item = {"item_id": item_id, "speaker_claimed": row["speaker"],
                "original_question": row["original_question"],
                "rewritten": row["rewritten"],
                "speaker_correct": "", "content_preserved": "", "notes": ""}
        for name in pair:
            sheets[name]["rewrite"].append(item)
        key_rows.append({"item_id": item_id, "kind": "rewrite", "job_id": row["job_id"],
                         "record_id": row["record_id"], "speaker": row["speaker"],
                         "certainty": "", "emotion": "", "target_letter": "",
                         "annotators": " ".join(pair)})

    for index, row in enumerate(sample_p):
        item_id = f"P{index:04d}"
        pair = assignment[("pushback", row["job_id"])]
        item = {"item_id": item_id, "question": row["question"],
                "target_option": f"({row['target_letter']}) {row['target_text']}",
                "pushback": row["pushback"],
                "certainty_label": "", "emotion_label": "", "argues_target": "",
                "notes": ""}
        for name in pair:
            sheets[name]["pushback"].append(item)
        key_rows.append({"item_id": item_id, "kind": "pushback", "job_id": row["job_id"],
                         "record_id": row["record_id"], "speaker": row["speaker"],
                         "certainty": row["certainty"], "emotion": row["emotion"],
                         "target_letter": row["target_letter"],
                         "annotators": " ".join(pair)})

    for name in ANNOTATORS:
        for kind, columns in [("rewrite", REWRITE_COLUMNS), ("pushback", PUSHBACK_COLUMNS)]:
            items = sheets[name][kind]
            rng.shuffle(items)       # so conditions are not in a readable order
            path = OUT_DIR / f"{kind}s_{name}.csv"
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

    print(f"{OUT_DIR / 'key.csv'}: {len(key_rows)} items")
    print(f"{OUT_DIR / 'GUIDE.txt'}")
    print()
    print(f"{len(sample_r)} rewrites and {len(sample_p)} pushbacks, each double coded.")
    print(f"Per annotator: about {2 * (len(sample_r) + len(sample_p)) // 3} items.")


if __name__ == "__main__":
    main()
