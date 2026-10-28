"""
Builds the pushback prompt for the first row of AMQA, checks that every piece of
data is filled in correctly, and prints it.

counterfactual.py loads the model when imported, so this pulls PROMPT, the rules,
the field names, and the helper functions out of the file's text instead.

It reads the prepared AMQA file with the json module, so the datasets package
is not needed.

Run with: python -m unittest test_pushback_prompt.py -v
"""

import ast
import json
import random
import unittest
from pathlib import Path

SCRIPT_NAME = "counterfactual.py"
HELPERS = ("get_options", "get_correct_letter")

# Load the constants and helper functions from the script without running it.
source = (Path(__file__).parent / SCRIPT_NAME).read_text(encoding="utf-8")
ns = {"json": json}
for node in ast.parse(source).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        try:
            ns[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            pass  # not a literal (e.g. sys.argv[1], model loading)
    elif isinstance(node, ast.FunctionDef) and node.name in HELPERS:
        exec(compile(ast.Module([node], []), SCRIPT_NAME, "exec"), ns)

# Settings come from the script when it defines them, otherwise these defaults.
AMQA_FILE = ns.get("DATA_PATH", "dataset_prep/data/amqa.jsonl")
QUESTION_FIELD = ns.get("QUESTION_FIELD", "question")
ID_FIELD = ns.get("ID_FIELD", "record_id")

SPEAKER = "patient"
CERTAINTY = "tentative"
EMOTION = "frustrated"


class TestFirstRowPrompt(unittest.TestCase):

    def test_first_row(self):
        with open(AMQA_FILE, encoding="utf-8") as f:
            row = json.loads(next(line for line in f if line.strip()))

        for field in (QUESTION_FIELD, ID_FIELD, "options", "answer_idx"):
            self.assertIn(field, row, f"{field!r} missing. Columns: {list(row)}")

        question = row[QUESTION_FIELD]
        qid = str(row[ID_FIELD])
        self.assertGreater(len(question.split()), 10,
                           f"Question column looks wrong: {question!r}")

        options = ns["get_options"](row)
        correct = ns["get_correct_letter"](row, options)

        # The helpers should agree with the raw AMQA fields.
        self.assertEqual(set(options), set("ABCD"))
        for letter in "ABCD":
            self.assertEqual(options[letter], row["options"][letter])
        self.assertEqual(correct, row["answer_idx"])
        self.assertEqual(options[correct], row["reference_answer"])

        wrong = [letter for letter in "ABCD" if letter != correct]
        target = random.Random(f"{ns['SEED']}-{qid}").choice(wrong)

        prompt = ns["PROMPT"].format(
            speaker_rule=ns["SPEAKER_RULES"][SPEAKER],
            certainty_rule=ns["CERTAINTY_RULES"][CERTAINTY],
            emotion_rule=ns["EMOTION_RULES"][EMOTION],
            base_question=question,
            option_a=options["A"],
            option_b=options["B"],
            option_c=options["C"],
            option_d=options["D"],
            correct_letter=correct,
            correct_text=options[correct],
            target_letter=target,
            target_text=options[target],
            speaker=SPEAKER,
            certainty=CERTAINTY,
            emotion=EMOTION,
        )
        print(f"\nQuestion column: {QUESTION_FIELD!r} | ID column: {ID_FIELD!r}")
        print(f"\n===== question {qid}: {SPEAKER} / {CERTAINTY} / {EMOTION} =====\n{prompt}\n")

        self.assertNotEqual(target, correct)
        self.assertIn(ns["SPEAKER_RULES"][SPEAKER], prompt)
        self.assertIn(f'- Certainty: {ns["CERTAINTY_RULES"][CERTAINTY]}\n', prompt)
        self.assertIn(f'- Emotional tone: {ns["EMOTION_RULES"][EMOTION]}\n', prompt)
        self.assertIn(f"— Clinical Question: {question}\n", prompt)
        for letter in "ABCD":
            self.assertIn(f"({letter}) {options[letter]}\n", prompt)
        self.assertIn(f"— Correct Answer: ({correct}) {options[correct]}\n", prompt)
        self.assertIn(f"— Target Answer: ({target}) {options[target]}\n", prompt)
        self.assertIn(f"— Speaker: {SPEAKER}\n", prompt)
        self.assertIn(f"— Certainty: {CERTAINTY}\n", prompt)
        self.assertIn(f"— Emotional Tone: {EMOTION}\n", prompt)
        self.assertNotRegex(prompt, r"\{[a-z_]+\}", "Unfilled placeholder in prompt")


if __name__ == "__main__":
    unittest.main(verbosity=2)