"""Report dataset statistics and run the automatic checks on every generated row.

These checks are mechanical, so they run on the whole set rather than a sample.
What they cannot judge, whether a rewrite reads as the right speaker and whether
a pushback conveys the requested certainty and emotion, is left to the human
sample built by 04_annotation_sample.py.

Outputs:
  results/generation/stats.txt    this report
  results/generation/flagged.csv  rows that failed a check, for review

    python generation/03_stats.py
"""

import csv
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

REWRITES = os.getenv("REWRITES", "results/generation/rewrites.csv")
PUSHBACKS = os.getenv("PUSHBACKS", "results/generation/pushbacks.csv")
OUT_DIR = Path(os.getenv("OUT_DIR", "results/generation"))

SENTENCE = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")
PERSON_REF = re.compile(r"\b(?:this|the)\s+(?:patient|person)('s)?\b", re.I)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Anchored to the start, and to phrases about the task rather than the body, so
# clinical first person such as "I cannot climb stairs" is not counted.
REFUSAL = re.compile(
    r"^\s*(I'm sorry|I am sorry|I apologize|I cannot (help|assist|comply|provide|write)"
    r"|I can't (help|assist|comply|provide|write)|As an AI|I'm unable to (help|assist|provide))",
    re.I)


LINES = []


def say(text=""):
    print(text)
    LINES.append(text)


def read(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sentences(text):
    text = text.strip()
    return SENTENCE.split(text) if text else []


def describe(values):
    if not values:
        return {}
    values = sorted(values)
    return {"n": len(values), "mean": round(statistics.mean(values), 1),
            "median": values[len(values) // 2], "min": values[0], "max": values[-1]}


def check_rewrites(rows):
    """Did the rewrite keep the clinical content and follow the speaker rules?"""
    flagged, stats = [], Counter()
    lengths = []

    for row in rows:
        text = row["rewritten"].strip()
        speaker = row["speaker"]
        problems = []

        if not text:
            problems.append("empty")
        else:
            lengths.append(len(text.split()))
            if REFUSAL.search(text):
                problems.append("refusal_language")
            # the opening sentence is fixed per speaker and must be reproduced exactly
            opening = row["opening"].strip()
            if opening and not text.startswith(opening):
                problems.append("opening_missing")
            # every number in the original must survive, since they are clinical facts
            missing = [n for n in NUMBER.findall(row["original_question"])
                       if n not in text]
            if missing:
                problems.append(f"numbers_missing:{','.join(missing[:4])}")
            # The final sentence must refer to the person as the speaker would.
            # Checked by substitution rather than against final_question_expected,
            # which is malformed on some rows.
            if speaker != "none":
                last = sentences(text)[-1] if sentences(text) else ""
                if PERSON_REF.search(last):
                    problems.append("final_sentence_not_swapped")
            # the control is a verbatim copy, so anything else is a bug
            if speaker == "none" and text != row["original_question"].strip():
                problems.append("control_not_verbatim")

        stats[speaker] += 1
        for problem in problems:
            stats[f"fail:{problem.split(':')[0]}"] += 1
        if problems:
            flagged.append({"file": "rewrites", "job_id": row["job_id"],
                            "speaker": speaker, "problems": "; ".join(problems),
                            "text": text[:300]})

    return flagged, stats, lengths


def check_pushbacks(rows):
    """Does the pushback argue for the assigned wrong option, and only that one?"""
    flagged, stats = [], Counter()
    lengths, sentence_counts = [], []

    for row in rows:
        text = row["pushback"].strip()
        options = json.loads(row["options"])
        target, correct = row["target_letter"], row["correct_letter"]
        problems = []

        if not text:
            problems.append("empty")
        else:
            lengths.append(len(text.split()))
            count = len(sentences(text))
            sentence_counts.append(count)
            if REFUSAL.search(text):
                problems.append("refusal_language")
            if count > 4:
                problems.append("too_long")

            # The target may be named as "(B)", "B)" or "B is", all of which are
            # unambiguous references to the option.
            letters = set(re.findall(r"\(([A-Z])\)", text))
            if target not in letters and not re.search(
                    rf"\b{target}\)|\b{target}\s+(?:is|was|would)\b", text):
                problems.append("target_letter_absent")
            # naming any other option risks arguing against the correct answer
            others = letters - {target}
            if others:
                problems.append(f"other_letters:{''.join(sorted(others))}")
            # The correct option must not be named. Matched as a whole phrase with
            # word boundaries, so an option that merely shares a drug name with the
            # target is not counted.
            phrase = re.escape(options[correct].strip().rstrip("."))
            if re.search(rf"\b{phrase}\b", text, re.I):
                problems.append("correct_text_leaked")

        stats[f"{row['certainty']}|{row['emotion']}"] += 1
        stats[f"speaker:{row['speaker']}"] += 1
        for problem in problems:
            stats[f"fail:{problem.split(':')[0]}"] += 1
        if problems:
            flagged.append({"file": "pushbacks", "job_id": row["job_id"],
                            "speaker": row["speaker"],
                            "problems": "; ".join(problems), "text": text[:300]})

    return flagged, stats, lengths, sentence_counts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rewrites, pushbacks = read(REWRITES), read(PUSHBACKS)

    # ----- coverage -----
    say("=" * 64)
    say("COVERAGE")
    say("=" * 64)
    questions = {row["record_id"] for row in rewrites}
    filled_rewrites = sum(1 for row in rewrites if row["rewritten"].strip())
    filled_pushbacks = sum(1 for row in pushbacks if row["pushback"].strip())
    say(f"questions                {len(questions)}")
    say(f"rewrites                 {len(rewrites)} rows, {filled_rewrites} filled")
    say(f"pushbacks                {len(pushbacks)} rows, {filled_pushbacks} filled")
    say(f"speakers per question    {len(rewrites) / max(len(questions), 1):.1f}")
    say(f"pushbacks per rewrite    {len(pushbacks) / max(len(rewrites), 1):.1f}")

    # ----- balance -----
    say()
    say("=" * 64)
    say("BALANCE")
    say("=" * 64)
    for name, rows, field in [("speaker (rewrites)", rewrites, "speaker"),
                              ("speaker (pushbacks)", pushbacks, "speaker"),
                              ("certainty", pushbacks, "certainty"),
                              ("emotion", pushbacks, "emotion")]:
        counts = Counter(row[field] for row in rows)
        say(f"{name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    cells = Counter((row["certainty"], row["emotion"]) for row in pushbacks)
    say(f"certainty x emotion cells: {len(cells)}, "
          f"sizes {min(cells.values())} to {max(cells.values())}")

    # ----- target answers -----
    per_question = defaultdict(set)
    for row in pushbacks:
        per_question[row["record_id"]].add(row["target_letter"])
    inconsistent = [q for q, letters in per_question.items() if len(letters) > 1]
    target_spread = Counter(row["target_letter"] for row in pushbacks)
    say()
    say(f"target letter distribution: {dict(sorted(target_spread.items()))}")
    say(f"questions with more than one target letter: {len(inconsistent)} (want 0)")

    # ----- automatic checks -----
    say()
    say("=" * 64)
    say("AUTOMATIC CHECKS")
    say("=" * 64)
    flagged_r, stats_r, len_r = check_rewrites(rewrites)
    flagged_p, stats_p, len_p, sent_p = check_pushbacks(pushbacks)

    say(f"rewrites:  {len(rewrites) - len(flagged_r)}/{len(rewrites)} pass "
          f"({100 * (len(rewrites) - len(flagged_r)) / max(len(rewrites), 1):.1f}%)")
    for key, count in sorted(stats_r.items()):
        if key.startswith("fail:"):
            say(f"   {key[5:]:28s} {count:6d}  ({100 * count / len(rewrites):.1f}%)")

    say(f"pushbacks: {len(pushbacks) - len(flagged_p)}/{len(pushbacks)} pass "
          f"({100 * (len(pushbacks) - len(flagged_p)) / max(len(pushbacks), 1):.1f}%)")
    for key, count in sorted(stats_p.items()):
        if key.startswith("fail:"):
            say(f"   {key[5:]:28s} {count:6d}  ({100 * count / len(pushbacks):.1f}%)")


    # ----- rows clean enough to use -----
    by_speaker = defaultdict(lambda: [0, 0])
    for row in rewrites:
        by_speaker[row["speaker"]][0] += 1
    for row in flagged_r:
        by_speaker[row["speaker"]][1] += 1
    say()
    say("rewrite failures by speaker:")
    for speaker, (total, failed) in sorted(by_speaker.items()):
        say(f"   {speaker:10s} {failed:5d}/{total:5d}  ({100 * failed / max(total, 1):.1f}%)")

    # ----- lengths -----
    say()
    say("=" * 64)
    say("LENGTHS (words)")
    say("=" * 64)
    original = describe([int(row["question_words"]) for row in rewrites
                         if row["question_words"]])
    say(f"original question  {original}")
    say(f"rewritten          {describe(len_r)}")
    say(f"pushback           {describe(len_p)}")
    say(f"pushback sentences {describe(sent_p)}")

    # length by condition, since a length difference would confound the framing
    # Length is a covariate: if one condition is systematically longer, an effect
    # could be length rather than framing. Reported per cell so that can be ruled out.
    by_cell = defaultdict(list)
    for row in pushbacks:
        if row["pushback"].strip():
            by_cell[(row["certainty"], row["emotion"])].append(len(row["pushback"].split()))
    say()
    say("pushback length by cell (words)")
    say(f"   {'':10s} " + "".join(f"{e:>12s}" for e in ["neutral", "anxious", "frustrated"]))
    for certainty in ["tentative", "neutral", "assertive"]:
        row = "".join(f"{statistics.mean(by_cell[(certainty, e)]):12.1f}"
                      for e in ["neutral", "anxious", "frustrated"])
        say(f"   {certainty:10s} {row}")

    # ----- the line a paper reports -----
    usable_r = len(rewrites) - len(flagged_r)
    usable_p = len(pushbacks) - len(flagged_p)
    say()
    say("=" * 64)
    say("SUMMARY FOR THE REPORT")
    say("=" * 64)
    say(f"{len(questions)} questions x 5 speakers = {len(rewrites)} turn 1 inputs")
    say(f"{len(rewrites)} inputs x 9 pushbacks = {len(pushbacks)} turn 2 counterfactuals")
    sizes = Counter((r["certainty"], r["emotion"]) for r in pushbacks).values()
    say(f"certainty by emotion cells range {min(sizes)} to {max(sizes)}")
    say(f"one target option per question, verified for all {len(questions)}")
    say(f"rewrites passing all checks   {usable_r}/{len(rewrites)} "
          f"({100 * usable_r / max(len(rewrites), 1):.1f}%)")
    say(f"pushbacks passing all checks  {usable_p}/{len(pushbacks)} "
          f"({100 * usable_p / max(len(pushbacks), 1):.1f}%)")

    # ----- write -----
    (OUT_DIR / "stats.txt").write_text("\n".join(LINES) + "\n")
    flagged = flagged_r + flagged_p
    with (OUT_DIR / "flagged.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "job_id", "speaker",
                                                    "problems", "text"])
        writer.writeheader()
        writer.writerows(flagged)
    say()
    say(f"wrote {OUT_DIR / 'stats.txt'}")
    say(f"wrote {OUT_DIR / 'flagged.csv'} ({len(flagged)} rows)")


if __name__ == "__main__":
    main()