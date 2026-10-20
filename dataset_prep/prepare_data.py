import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


SOURCES = {
    "MedQuAD": ("https://github.com/abachaa/MedQuAD", "577bd37b96c02d1833b2c9eed2de9f96964e96cb"),
    "AMQA": ("https://github.com/XY-Showing/AMQA", "335b1752ba4912d00050fa6531ba8d2c2aef6fb9"),
}
CONDITIONS = ("white", "black", "high_income", "low_income", "male", "female")


def normalize(text):
    return " ".join(text.split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON: {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object: {path}:{line_number}")
            yield row


def read_medquad(root):
    root = Path(root)
    paths = sorted(root.glob("*/*.xml"))
    if not paths:
        raise ValueError(f"No MedQuAD XML files: {root}")
    rows = []
    for path in paths:
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        try:
            document = ET.fromstring(raw)
        except ET.ParseError as error:
            raise ValueError(f"Invalid XML: {path}") from error
        if document.tag in ("Document", "DiseaseFile"):
            container, pair_tag, question_tag, answer_tag, focus_tag = "QAPairs", "QAPair", "Question", "Answer", "Focus"
        elif document.tag == "doc":
            container, pair_tag, question_tag, answer_tag, focus_tag = "qaPairs", "pair", "question", "answer", "doctitle-focus"
        else:
            raise ValueError(f"Unexpected XML root: {path}")
        pairs = document.findall(f"./{container}/{pair_tag}")
        if document.find(container) is None or len(document.findall(f".//{pair_tag}")) != len(pairs) or len(document.findall(f".//{question_tag}")) != sum(len(pair.findall(question_tag)) for pair in pairs):
            raise ValueError(f"Unexpected QA structure: {path}")
        relative = path.relative_to(root).as_posix()
        for index, pair in enumerate(pairs, 1):
            if len(pair.findall(question_tag)) > 1 or len(pair.findall(answer_tag)) > 1:
                raise ValueError(f"Repeated QA element: {path}#{index}")
            question, answer = pair.find(question_tag), pair.find(answer_tag)
            rows.append({
                "record_id": f"medquad:{relative}#{index}",
                "dataset": "MedQuAD", "source_file": relative,
                "source_file_sha256": file_hash,
                "source_qid": question.get("qid", "") if question is not None else "",
                "source_subset": path.parent.name,
                "source_schema": document.tag,
                "source_url": document.get("url", ""),
                "focus": document.findtext(focus_tag, ""),
                "question_type": question.get("qtype", "") if question is not None else "",
                "question": "".join(question.itertext()) if question is not None else "",
                "reference_answer": "".join(answer.itertext()) if answer is not None else "",
                "options": {}, "answer_idx": "", "desensitized_question": "",
            })
    if not rows:
        raise ValueError(f"No MedQuAD QA records: {root}")
    return rows


def read_amqa(root):
    path = Path(root) / "AMQA_Dataset/AMQA_Dataset.jsonl"
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, variants, ids = [], [], set()
    for line_number, item in enumerate(load_jsonl(path), 1):
        qid = item.get("question_id")
        if type(qid) not in (str, int) or not str(qid).strip():
            raise ValueError(f"Missing question_id: {path}:{line_number}")
        qid = str(qid)
        if qid in ids:
            raise ValueError(f"Duplicate AMQA question_id: {qid}")
        ids.add(qid)
        fields = ["original_question", "desensitized_question", "answer", "answer_idx"]
        fields += [f"{prefix}_{condition}" for condition in CONDITIONS
                   for prefix in ("adv_question", "adv_description")]
        for field in fields:
            if not isinstance(item.get(field), str):
                raise ValueError(f"AMQA {qid}: invalid {field}")
        options = item.get("options")
        if not isinstance(options, dict) or len(options) < 2 or not all(
            isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
            for key, value in options.items()
        ):
            raise ValueError(f"AMQA {qid}: invalid options")
        if options.get(item["answer_idx"]) != item["answer"]:
            raise ValueError(f"AMQA {qid}: answer does not match answer_idx")
        record_id = f"amqa:{qid}"
        rows.append({
            "record_id": record_id, "dataset": "AMQA",
            "source_file": "AMQA_Dataset/AMQA_Dataset.jsonl",
            "source_file_sha256": file_hash, "source_qid": qid,
            "source_subset": "AMQA_original", "source_schema": "AMQA",
            "source_url": SOURCES["AMQA"][0],
            "focus": "", "question_type": "clinical_mcq",
            "question": item["original_question"], "reference_answer": item["answer"],
            "options": options, "answer_idx": item["answer_idx"],
            "desensitized_question": item["desensitized_question"],
        })
        for condition in CONDITIONS:
            text = item[f"adv_question_{condition}"]
            if not text.strip():
                raise ValueError(f"AMQA {qid}: empty {condition} variant")
            variants.append({
                "variant_id": f"{record_id}:{condition}", "record_id": record_id,
                "condition": condition, "question": text,
                "added_description": item[f"adv_description_{condition}"],
                "question_sha256": digest(text), "review_status": "pending",
            })
    if not rows:
        raise ValueError(f"Empty AMQA dataset: {path}")
    return rows, variants


def screen(rows):
    seen, ids, answers = {}, set(), defaultdict(set)
    for row in rows:
        if row["record_id"] in ids:
            raise ValueError(f"Duplicate record_id: {row['record_id']}")
        ids.add(row["record_id"])
        row["question_sha256"] = digest(row["question"])
        row["answer_sha256"] = digest(row["reference_answer"])
        row["question_words"] = len(row["question"].split())
        row["answer_words"] = len(row["reference_answer"].split())
        group = [row["dataset"], normalize(row["question"]), row["options"]]
        row["question_group_id"] = digest(json.dumps(group, sort_keys=True, ensure_ascii=False))
        row["duplicate_of"] = ""
        answer = (normalize(row["reference_answer"]), row["answer_idx"])
        if not row["question"].strip():
            row["screen_status"] = "missing_question"
        elif not row["reference_answer"].strip():
            row["screen_status"] = "missing_answer"
        else:
            answers[row["question_group_id"]].add(answer)
            key = (row["question_group_id"], *answer)
            if key in seen:
                row["screen_status"] = "duplicate_qa"
                row["duplicate_of"] = seen[key]
            else:
                seen[key] = row["record_id"]
                row["screen_status"] = "candidate"
        flags = []
        text = normalize(row["reference_answer"]).lower()
        if row["question_type"] in ("treatment", "prevention", "outlook", "susceptibility", "research", "frequency"):
            flags.append("clinical_or_time_sensitive")
        if row["question_type"] == "support groups":
            flags.append("support_resource")
        if re.match(r"^(see |for (more |additional )?information|please (contact|see))", text):
            flags.append("reference_only")
        if row["answer_words"] > 500:
            flags.append("long_answer")
        if re.search(r"not known|unknown|depends? on|not fully understood", text):
            flags.append("uncertainty_or_context")
        if len(set(normalize(value) for value in row["options"].values())) != len(row["options"]):
            flags.append("duplicate_option_text")
        row["review_flags"] = flags
    for row in rows:
        if len(answers[row["question_group_id"]]) > 1:
            row["review_flags"].append("multiple_reference_answers")
    return rows


def check(rows, variants):
    lookup = {row["record_id"]: row for row in rows}
    if len(variants) != 6 * sum(row["dataset"] == "AMQA" for row in rows):
        raise ValueError("Expected six variants per AMQA question")
    for row in rows:
        if digest(row["question"]) != row["question_sha256"] or digest(row["reference_answer"]) != row["answer_sha256"]:
            raise ValueError(f"Text changed: {row['record_id']}")
        if row["duplicate_of"]:
            other = lookup[row["duplicate_of"]]
            if row["question_group_id"] != other["question_group_id"] or normalize(row["reference_answer"]) != normalize(other["reference_answer"]) or row["answer_idx"] != other["answer_idx"]:
                raise ValueError(f"Invalid duplicate: {row['record_id']}")
    variant_ids, conditions = set(), defaultdict(set)
    for variant in variants:
        if variant["variant_id"] in variant_ids:
            raise ValueError(f"Duplicate variant_id: {variant['variant_id']}")
        variant_ids.add(variant["variant_id"])
        base = lookup.get(variant["record_id"])
        if base is None or base["dataset"] != "AMQA":
            raise ValueError(f"Missing variant parent: {variant['variant_id']}")
        variant["question_group_id"] = base["question_group_id"]
        conditions[base["record_id"]].add(variant["condition"])
        if digest(variant["question"]) != variant["question_sha256"]:
            raise ValueError(f"Text changed: {variant['variant_id']}")
    for row in rows:
        if row["dataset"] == "AMQA" and conditions[row["record_id"]] != set(CONDITIONS):
            raise ValueError(f"Missing AMQA variants: {row['record_id']}")


def source_info(root, name, rows):
    root = Path(root)
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    commit = result.stdout.strip()
    if commit != SOURCES[name][1]:
        raise ValueError(f"{name}: expected commit {SOURCES[name][1]}, found {commit}")
    result = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise ValueError(f"Modified source checkout: {root}")
    paths = sorted(root.glob("*/*.xml")) if name == "MedQuAD" else [root / "AMQA_Dataset/AMQA_Dataset.jsonl"]
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    used_files = {row["source_file"] for row in rows}
    for row in rows:
        if files[row["source_file"]] != row["source_file_sha256"]:
            raise ValueError(f"Source changed during loading: {row['source_file']}")
    checksum = digest("".join(f"{path}\0{sha}\n" for path, sha in sorted(files.items())))
    return {"url": SOURCES[name][0], "commit": commit, "file_count": len(files),
            "content_sha256": checksum, "files_without_records": sorted(set(files) - used_files)}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(sources, output):
    sources, output = Path(sources).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if output == sources or sources in output.parents or output in sources.parents:
        raise ValueError("Sources and output must be separate directories")
    medquad = read_medquad(sources / "MedQuAD")
    amqa, variants = read_amqa(sources / "AMQA")
    source_versions = {name: source_info(sources / name, name, rows)
                       for name, rows in (("MedQuAD", medquad), ("AMQA", amqa))}
    rows = screen(medquad + amqa)
    check(rows, variants)
    kept = [row for row in rows if row["screen_status"] == "candidate"]
    output.mkdir(parents=True)
    for name in ("MedQuAD", "AMQA"):
        write_jsonl(output / f"{name.lower()}.jsonl", [row for row in kept if row["dataset"] == name])
    write_jsonl(output / "excluded.jsonl", [row for row in rows if row["screen_status"] != "candidate"])
    write_jsonl(output / "amqa_variants.jsonl", variants)
    summary = {"schema_version": 1, "sources": source_versions, "datasets": {},
               "amqa_variants": len(variants)}
    for name in ("MedQuAD", "AMQA"):
        subset = [row for row in rows if row["dataset"] == name]
        counts = Counter(row["screen_status"] for row in subset)
        summary["datasets"][name] = {
            "read": len(subset),
            "source_schemas": dict(sorted(Counter(row["source_schema"] for row in subset).items())),
            **{key: counts[key] for key in ("candidate", "missing_question", "missing_answer", "duplicate_qa")},
            "repeated_source_qids": len(subset) - len({row["source_qid"] for row in subset}),
            "review_flags": dict(sorted(Counter(flag for row in subset if row["screen_status"] == "candidate" for flag in row["review_flags"]).items())),
        }
    summary["checks"] = {name: "passed" for name in (
        "record_ids", "answer_keys", "duplicate_links", "text_hashes", "variant_links", "source_versions"
    )}
    summary["files_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(output.iterdir())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare MedQuAD and AMQA.")
    parser.add_argument("--sources", type=Path, default=Path("sources"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        summary = prepare(args.sources, args.output)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
    for name, counts in summary["datasets"].items():
        print(f"{name}: {counts['read']} read, {counts['candidate']} kept, "
              f"{counts['missing_answer']} missing answers, {counts['duplicate_qa']} duplicates")
    print(f"AMQA variants: {summary['amqa_variants']}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
