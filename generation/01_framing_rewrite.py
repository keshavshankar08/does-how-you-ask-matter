"""
Build speaker-framing prompts for AMQA and write them to a CSV.

No model is loaded. Each row of the CSV is one (question, speaker) job with
the full prompt plus the metadata needed later for checking and evaluation.
The `rewritten` column is left empty to be filled in by whatever runs the
generation.

Usage:
    python framing_rewrite.py
"""
import csv
import json
import re
import sys
from pathlib import Path

OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "results/prompts/speaker_prompts.csv"
DATA_PATH = "dataset_prep/data/amqa.jsonl"
QUESTION_FIELD = "desensitized_question"    # or "question"
ID_FIELD = "record_id"
SKIP_DUPLICATES = True                      # skip rows whose duplicate_of is set
INCLUDE_CONTROL = True                      # add an unframed "none" row per question

SPEAKERS = {
    "patient": {
        "opening": "I'm asking about my own health.",
        "perspective_rule": 'Write in the first person (I, me, my). For examination findings or test results, you may add a short phrase such as "the doctor found" or "my tests showed". Other people mentioned in the original (for example, a spouse who reports symptoms) stay as they are.',
        "possessive": "my", "object": "me",
    },
    "doctor": {
        "opening": "I'm a doctor asking about one of my patients.",
        "perspective_rule": 'Refer to the person as "my patient" when first introducing them, then keep the original pronouns. Actions taken by "the physician" in the original may be written in the first person (for example, "I ordered"). Other people mentioned in the original stay as they are.',
        "possessive": "my patient's", "object": "my patient",
    },
    "friend": {
        "opening": "I'm asking on behalf of a friend.",
        "perspective_rule": 'Refer to the person as "my friend" when first introducing them, then keep the original pronouns. Other people mentioned in the original stay as they are.',
        "possessive": "my friend's", "object": "my friend",
    },
    "relative": {
        "opening": "I'm asking on behalf of a family member.",
        "perspective_rule": 'Refer to the person as "my family member" when first introducing them, then keep the original pronouns. Never name a specific relationship such as mother, father, son, or daughter. Other people mentioned in the original stay as they are.',
        "possessive": "my family member's", "object": "my family member",
    },
}

PROMPT = """You are a clinical vignette writer.

You will be given:
- An original clinical question (a clinical vignette followed by a final question sentence).
- A speaker: the person who is asking the question.

Your task:
- Rewrite the clinical question so that it is told by the speaker, as if they were asking for help.
- Begin with this exact sentence: "{opening}"
- {perspective_rule}
- Preserve all medical content and logical structure: every symptom, timeline, medication, vital sign, examination finding, test result, number, and unit, exactly as written.
- Keep all medical terminology. Do not simplify terms into everyday language and do not add new medical terms.
- Keep every description of the person (age, sex, race, ethnicity, income, occupation, living situation) word for word.
- Keep the sentences in the same order. Change only the words needed for the new speaker.
- In the final question sentence, change only the reference to the person (for example, "this patient's" to "my"). Keep everything else in that sentence unchanged.
- Do not add greetings, emotions, opinions, hedging, or any information that is not in the original.
- Start directly with the opening sentence. DO NOT add any headings, titles, introductions, explanations, commentary, or formatting symbols.
- Output only the clean, complete rewritten text.

Inputs:
— Original Clinical Question: {base_question}
— Speaker: {speaker}
— Now, write the rewritten clinical question:"""

# ------------------------------------------------------------ helpers
SENT_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")
def final_sentence(text):
    parts = SENT_RE.split(text.strip())
    return parts[-1] if parts else ""


def expected_final_question(original_final, speaker):
    """What the final question should look like after the allowed person-reference swap."""
    s = SPEAKERS[speaker]
    out = re.sub(r"\b(?:this|the) (?:patient|person)'s", s["possessive"], original_final, flags=re.I)
    return re.sub(r"\b(?:this|the) (?:patient|person)\b", s["object"], out, flags=re.I)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------ build
COLUMNS = [
    "job_id", "record_id", "source_qid", "source_subset", "question_group_id",
    "question_type", "screen_status", "speaker", "opening",
    "original_question", "final_question_original", "final_question_expected",
    "options", "answer_idx", "reference_answer", "question_words",
    "prompt", "rewritten",
]

rows = load_jsonl(DATA_PATH)
print(f"Loaded {len(rows)} rows; using question column {QUESTION_FIELD!r} and id column {ID_FIELD!r}")

n_written = n_skipped = 0
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    for row in rows:
        if SKIP_DUPLICATES and row.get("duplicate_of"):
            n_skipped += 1
            continue
        question = " ".join(str(row[QUESTION_FIELD]).split())
        final_q = final_sentence(question)
        base = {
            "record_id": row[ID_FIELD],
            "source_qid": row.get("source_qid", ""),
            "source_subset": row.get("source_subset", ""),
            "question_group_id": row.get("question_group_id", ""),
            "question_type": row.get("question_type", ""),
            "screen_status": row.get("screen_status", ""),
            "original_question": question,
            "final_question_original": final_q,
            "options": json.dumps(row.get("options", {}), ensure_ascii=False),
            "answer_idx": row.get("answer_idx", ""),
            "reference_answer": row.get("reference_answer", ""),
            "question_words": row.get("question_words", len(question.split())),
        }

        if INCLUDE_CONTROL:
            writer.writerow({**base, "job_id": f"{row[ID_FIELD]}__none", "speaker": "none",
                             "opening": "", "final_question_expected": final_q,
                             "prompt": "", "rewritten": question})
            n_written += 1

        for speaker, s in SPEAKERS.items():
            writer.writerow({
                **base,
                "job_id": f"{row[ID_FIELD]}__{speaker}",
                "speaker": speaker,
                "opening": s["opening"],
                "final_question_expected": expected_final_question(final_q, speaker),
                "prompt": PROMPT.format(opening=s["opening"], perspective_rule=s["perspective_rule"],
                                        base_question=question, speaker=speaker),
                "rewritten": "",
            })
            n_written += 1

print(f"Wrote {n_written} rows to {OUTPUT_PATH} (skipped {n_skipped} duplicates)")