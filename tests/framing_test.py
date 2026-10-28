"""Build and print the rewrite prompts for the first dataset row.

Reads SPEAKERS, PROMPT, DATA_PATH, QUESTION_FIELD and ID_FIELD from the rewrite
script's source (via ast), so the model is never loaded and no generation happens.

Run with:   pytest -s framing_test.py
       or:  python framing_test.py
Set REWRITE_SCRIPT if your script isn't named rewrite_questions.py.
"""
import ast
import os
from pathlib import Path

from datasets import load_dataset

SCRIPT_PATH = Path(os.environ.get("REWRITE_SCRIPT", "framing_rewrite.py"))
CONSTANT_NAMES = ("SPEAKERS", "PROMPT", "DATA_PATH", "QUESTION_FIELD", "ID_FIELD")


def load_constants(path, names=CONSTANT_NAMES):
    """Pull literal top-level assignments out of the script without running it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        ):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    missing = set(names) - found.keys()
    if missing:
        raise KeyError(f"Could not find {sorted(missing)} in {path}")
    return found


def test_build_prompt_first_row():
    c = load_constants(SCRIPT_PATH)

    ds = load_dataset("json", data_files=c["DATA_PATH"], split="train[:1]")
    row = ds[0]

    for field in (c["QUESTION_FIELD"], c["ID_FIELD"]):
        assert field in ds.column_names, f"{field!r} not in columns: {ds.column_names}"

    question = row[c["QUESTION_FIELD"]]
    assert len(question.split()) > 10, f"Question column looks wrong: {question!r}"

    print(f"\nQuestion column: {c['QUESTION_FIELD']!r} | ID column: {c['ID_FIELD']!r}")
    print(f"Row id: {row[c['ID_FIELD']]}")

    for speaker, s in c["SPEAKERS"].items():
        prompt = c["PROMPT"].format(
            opening=s["opening"],
            perspective_rule=s["perspective_rule"],
            base_question=question,
            speaker=speaker,
        )

        print(f"\n{'=' * 30} {speaker} {'=' * 30}\n{prompt}")

        assert s["opening"] in prompt
        assert s["perspective_rule"] in prompt
        assert question in prompt
        for placeholder in ("{opening}", "{perspective_rule}", "{base_question}", "{speaker}"):
            assert placeholder not in prompt, f"Unfilled {placeholder} for {speaker}"


if __name__ == "__main__":
    test_build_prompt_first_row()