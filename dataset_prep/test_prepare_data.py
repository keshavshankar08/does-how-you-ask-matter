import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import prepare_data as prep


def record(record_id, question="Question?", answer="Answer.", options=None, answer_idx=""):
    return {
        "record_id": record_id, "dataset": "MedQuAD", "question": question,
        "reference_answer": answer, "options": options or {},
        "answer_idx": answer_idx, "question_type": "information",
    }


def amqa_item(qid="0"):
    item = {
        "question_id": qid, "original_question": "Original question?",
        "desensitized_question": "Neutral question?",
        "options": {"A": "Answer.", "B": "Other answer."},
        "answer": "Answer.", "answer_idx": "A", "test_model_answer_original": "B",
    }
    for condition in prep.CONDITIONS:
        item[f"adv_question_{condition}"] = f"Question for {condition}?"
        item[f"adv_description_{condition}"] = f"Description for {condition}."
    return item


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = self.root / "sources"
        self.xml = self.sources / "MedQuAD/subset/a.xml"
        self.xml.parent.mkdir(parents=True)
        self.xml.write_text(
            '<Document url="https://example.org/source"><Focus>Topic</Focus><QAPairs>'
            '<QAPair><Question qid="1" qtype="information">  Q &amp; é?  </Question>'
            '<Answer>Text <b>inside</b> text.</Answer></QAPair>'
            '<QAPair><Question qid="2" qtype="information">Empty answer?</Question>'
            '<Answer /></QAPair></QAPairs></Document>', encoding="utf-8"
        )
        self.amqa = self.sources / "AMQA/AMQA_Dataset/AMQA_Dataset.jsonl"
        self.amqa.parent.mkdir(parents=True)
        self.write_amqa([amqa_item()])

    def tearDown(self):
        self.temp.cleanup()

    def write_amqa(self, rows):
        prep.write_jsonl(self.amqa, rows)

    def test_medquad_text_and_ids(self):
        before = self.xml.read_bytes()
        rows = prep.read_medquad(self.sources / "MedQuAD")
        self.assertEqual(rows[0]["question"], "  Q & é?  ")
        self.assertEqual(rows[0]["reference_answer"], "Text inside text.")
        self.assertEqual(rows[0]["record_id"], "medquad:subset/a.xml#1")
        self.assertEqual(rows[0]["source_url"], "https://example.org/source")
        self.assertEqual(self.xml.read_bytes(), before)
        self.assertEqual(prep.screen(rows)[1]["screen_status"], "missing_answer")

    def test_medquad_qid_is_scoped_by_file(self):
        self.xml.with_name("b.xml").write_bytes(self.xml.read_bytes())
        rows = prep.screen(prep.read_medquad(self.sources / "MedQuAD"))
        self.assertEqual(len({r["record_id"] for r in rows}), 4)
        self.assertEqual(rows[2]["duplicate_of"], rows[0]["record_id"])

    def test_empty_xml_is_counted(self):
        self.xml.with_name("empty.xml").write_text("<Document><QAPairs /></Document>")
        rows = prep.read_medquad(self.sources / "MedQuAD")
        self.assertEqual(len(rows), 2)
        answers = [SimpleNamespace(stdout=prep.SOURCES["MedQuAD"][1]), SimpleNamespace(stdout="")]
        with patch.object(prep.subprocess, "run", side_effect=answers):
            info = prep.source_info(self.sources / "MedQuAD", "MedQuAD", rows)
        self.assertEqual(info["file_count"], 2)
        self.assertEqual(info["files_without_records"], ["subset/empty.xml"])

    def test_missing_question(self):
        self.xml.write_text('<Document><QAPairs><QAPair><Answer>A</Answer></QAPair></QAPairs></Document>')
        rows = prep.screen(prep.read_medquad(self.sources / "MedQuAD"))
        self.assertEqual(rows[0]["screen_status"], "missing_question")

    def test_legacy_medquad_schema(self):
        self.xml.write_text(
            '<doc url="https://example.org/legacy"><doctitle-focus>Legacy topic</doctitle-focus>'
            '<qaPairs><pair><question qid="legacy-1" qtype="information"> Legacy question? </question>'
            '<answer>Legacy answer.</answer></pair></qaPairs></doc>'
        )
        rows = prep.read_medquad(self.sources / "MedQuAD")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question"], " Legacy question? ")
        self.assertEqual(rows[0]["focus"], "Legacy topic")
        self.assertEqual(rows[0]["source_qid"], "legacy-1")
        self.assertEqual(rows[0]["source_schema"], "doc")

    def test_diseasefile_root(self):
        text = self.xml.read_text(encoding="utf-8").replace("Document", "DiseaseFile")
        self.xml.write_text(text, encoding="utf-8")
        rows = prep.read_medquad(self.sources / "MedQuAD")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_schema"], "DiseaseFile")

    def test_missing_or_invalid_xml(self):
        with self.assertRaisesRegex(ValueError, "No MedQuAD"):
            prep.read_medquad(self.root / "absent")
        self.xml.write_text("<bad")
        with self.assertRaisesRegex(ValueError, "Invalid XML"):
            prep.read_medquad(self.sources / "MedQuAD")
        self.xml.write_text("<Document />")
        with self.assertRaisesRegex(ValueError, "Unexpected QA"):
            prep.read_medquad(self.sources / "MedQuAD")

    def test_amqa_reference_answer_and_variants(self):
        before = self.amqa.read_bytes()
        rows, variants = prep.read_amqa(self.sources / "AMQA")
        self.assertEqual(rows[0]["answer_idx"], "A")
        self.assertEqual(rows[0]["reference_answer"], "Answer.")
        self.assertEqual(rows[0]["desensitized_question"], "Neutral question?")
        self.assertFalse(any(k.startswith("test_model") for k in rows[0]))
        self.assertEqual(len(variants), 6)
        prep.check(prep.screen(rows), variants)
        self.assertTrue(all(v["question_group_id"] == rows[0]["question_group_id"] for v in variants))
        self.assertEqual(self.amqa.read_bytes(), before)

    def test_amqa_bad_key_options_and_fields(self):
        for field, value, message in (
            ("answer_idx", "B", "does not match"),
            ("options", {}, "invalid options"),
            ("original_question", None, "invalid original_question"),
            ("adv_question_white", "", "empty white variant"),
            ("question_id", "", "Missing question_id"),
        ):
            with self.subTest(field=field):
                item = amqa_item()
                item[field] = value
                self.write_amqa([item])
                with self.assertRaisesRegex(ValueError, message):
                    prep.read_amqa(self.sources / "AMQA")

    def test_amqa_duplicate_ids(self):
        self.write_amqa([amqa_item("0"), amqa_item(0)])
        with self.assertRaisesRegex(ValueError, "Duplicate AMQA"):
            prep.read_amqa(self.sources / "AMQA")

    def test_json_errors_and_bom(self):
        self.amqa.write_text("bad\n")
        with self.assertRaisesRegex(ValueError, "Invalid JSON.*:1"):
            list(prep.load_jsonl(self.amqa))
        self.amqa.write_text("[]\n")
        with self.assertRaisesRegex(ValueError, "Expected an object"):
            list(prep.load_jsonl(self.amqa))
        self.amqa.write_text(json.dumps(amqa_item()) + "\n\n", encoding="utf-8-sig")
        self.assertEqual(len(prep.read_amqa(self.sources / "AMQA")[0]), 1)

    def test_deduplication_preserves_text(self):
        rows = prep.screen([
            record("a"), record("b", question="  Question?\n"),
            record("c", answer=""), record("d", question=""),
            record("e", options={"A": "Option"}),
        ])
        self.assertEqual([r["screen_status"] for r in rows],
                         ["candidate", "duplicate_qa", "missing_answer", "missing_question", "candidate"])
        self.assertEqual(rows[1]["question"], "  Question?\n")
        self.assertEqual(rows[1]["duplicate_of"], "a")
        prep.check(rows, [])

    def test_multiple_answers_are_retained(self):
        rows = prep.screen([record("a"), record("b", answer="Different.")])
        self.assertTrue(all(r["screen_status"] == "candidate" for r in rows))
        self.assertTrue(all("multiple_reference_answers" in r["review_flags"] for r in rows))
        self.assertEqual(rows[0]["question_group_id"], rows[1]["question_group_id"])

    def test_answer_labels_and_option_text(self):
        options = {"A": "Answer.", "B": "Answer."}
        rows = prep.screen([record("a", options=options, answer_idx="A"),
                            record("b", options=options, answer_idx="B")])
        self.assertTrue(all(r["screen_status"] == "candidate" for r in rows))
        self.assertTrue(all("duplicate_option_text" in r["review_flags"] for r in rows))

    def test_duplicate_record_id(self):
        with self.assertRaisesRegex(ValueError, "Duplicate record_id"):
            prep.screen([record("a"), record("a")])

    def test_variant_checks(self):
        rows, variants = prep.read_amqa(self.sources / "AMQA")
        prep.screen(rows)
        with self.assertRaisesRegex(ValueError, "six variants"):
            prep.check(rows, variants[:-1])
        orphan = copy.deepcopy(variants)
        orphan[0]["record_id"] = "amqa:missing"
        with self.assertRaisesRegex(ValueError, "Missing variant parent"):
            prep.check(rows, orphan)
        changed = copy.deepcopy(variants)
        changed[0]["question"] += " changed"
        with self.assertRaisesRegex(ValueError, "Text changed"):
            prep.check(rows, changed)
        duplicated = copy.deepcopy(variants)
        duplicated[1]["variant_id"] = duplicated[0]["variant_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate variant_id"):
            prep.check(rows, duplicated)

    def test_text_hash_check(self):
        rows = prep.screen([record("a")])
        rows[0]["question"] += "changed"
        with self.assertRaisesRegex(ValueError, "Text changed"):
            prep.check(rows, [])

    def test_source_versions(self):
        rows = prep.read_medquad(self.sources / "MedQuAD")
        with patch.object(prep.subprocess, "run", return_value=SimpleNamespace(stdout="wrong")):
            with self.assertRaisesRegex(ValueError, "expected commit"):
                prep.source_info(self.sources / "MedQuAD", "MedQuAD", rows)
        answers = [SimpleNamespace(stdout=prep.SOURCES["MedQuAD"][1]), SimpleNamespace(stdout=" M file")]
        with patch.object(prep.subprocess, "run", side_effect=answers):
            with self.assertRaisesRegex(ValueError, "Modified source"):
                prep.source_info(self.sources / "MedQuAD", "MedQuAD", rows)

    def test_repeated_runs_and_output_preservation(self):
        with patch.object(prep, "source_info", return_value={"commit": "fixture"}):
            first, second = self.root / "first", self.root / "second"
            summary = prep.prepare(self.sources, first)
            prep.prepare(self.sources, second)
            self.assertEqual(summary["datasets"]["MedQuAD"]["candidate"], 1)
            self.assertEqual(summary["datasets"]["MedQuAD"]["missing_answer"], 1)
            self.assertEqual(summary["amqa_variants"], 6)
            self.assertEqual({path.name for path in first.iterdir()}, {
                "medquad.jsonl", "amqa.jsonl", "amqa_variants.jsonl", "excluded.jsonl", "summary.json",
            })
            self.assertFalse({"review_sample", "variant_review_sample", "sample_seed"} & summary.keys())
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            saved = first / "amqa.jsonl"
            saved.write_text('{"record_id": "manually-edited"}\n')
            before = saved.read_bytes()
            with self.assertRaisesRegex(ValueError, "Output already exists"):
                prep.prepare(self.sources, first)
            self.assertEqual(saved.read_bytes(), before)

    def test_bad_input_creates_no_output(self):
        self.amqa.write_text("invalid")
        output = self.root / "bad"
        with self.assertRaises(ValueError):
            prep.prepare(self.sources, output)
        self.assertFalse(output.exists())

    def test_output_cannot_be_inside_sources(self):
        with self.assertRaisesRegex(ValueError, "separate directories"):
            prep.prepare(self.sources, self.sources / "output")


if __name__ == "__main__":
    unittest.main()
