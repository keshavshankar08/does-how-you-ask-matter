"""Score the returned annotation sheets.

Two numbers matter:

  agreement       how often the two annotators who saw an item gave the same
                  label, reported as raw agreement and Cohen's kappa. Kappa
                  corrects for agreement that would happen by chance, so it is
                  the one to report. Above 0.6 is usually called substantial and
                  above 0.8 near perfect.
  recovery        for certainty and emotion, how often annotators recovered the
                  level the generator was asked for. This is the manipulation
                  check: if annotators cannot tell an assertive pushback from a
                  tentative one, the manipulation did not land.

    python generation/05_agreement.py
"""

import csv
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

csv.field_size_limit(sys.maxsize)

DIR = Path(os.getenv("ANNOTATION_DIR", "results/annotation"))
ANNOTATORS = ["a1", "a2", "a3"]
REWRITE_FIELDS = ["speaker_correct", "content_preserved"]
PUSHBACK_FIELDS = ["certainty_label", "emotion_label", "argues_target"]


def cohens_kappa(pairs):
    """pairs is a list of (label_a, label_b). Returns raw agreement and kappa."""
    if not pairs:
        return None, None
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    labels = {label for pair in pairs for label in pair}
    expected = sum((sum(1 for a, _ in pairs if a == label) / total) *
                   (sum(1 for _, b in pairs if b == label) / total)
                   for label in labels)
    if expected >= 1:
        return observed, 1.0 if observed == 1 else 0.0
    return observed, (observed - expected) / (1 - expected)


def load_sheets(kind):
    sheets = {}
    for name in ANNOTATORS:
        path = DIR / f"{kind}s_{name}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            sheets[name] = {row["item_id"]: row for row in csv.DictReader(stream)}
    return sheets


def norm(value):
    return value.strip().lower()


def report(kind, fields, key):
    sheets = load_sheets(kind)
    if not sheets:
        print(f"no {kind} sheets in {DIR}")
        return

    print("=" * 64)
    print(kind.upper())
    print("=" * 64)
    for name, rows in sorted(sheets.items()):
        done = sum(1 for row in rows.values() if all(norm(row[f]) for f in fields))
        print(f"  {name}: {done}/{len(rows)} items complete")

    for field in fields:
        print()
        print(f"{field}")
        all_pairs = []
        for left, right in combinations(sorted(sheets), 2):
            shared = set(sheets[left]) & set(sheets[right])
            pairs = [(norm(sheets[left][i][field]), norm(sheets[right][i][field]))
                     for i in shared
                     if norm(sheets[left][i][field]) and norm(sheets[right][i][field])]
            if not pairs:
                continue
            observed, kappa = cohens_kappa(pairs)
            print(f"  {left} vs {right}: n={len(pairs):4d}  agreement {observed:.2f}  kappa {kappa:.2f}")
            all_pairs += pairs
        if all_pairs:
            observed, kappa = cohens_kappa(all_pairs)
            print(f"  pooled:     n={len(all_pairs):4d}  agreement {observed:.2f}  kappa {kappa:.2f}")

        counts = Counter(norm(row[field]) for rows in sheets.values()
                         for row in rows.values() if norm(row[field]))
        print(f"  labels used: {dict(sorted(counts.items()))}")

    # manipulation check: did annotators recover the requested level
    intended = {field: field.replace("_label", "") for field in fields
                if field.endswith("_label")}
    for field, column in intended.items():
        matched = total = 0
        confusion = defaultdict(Counter)
        for rows in sheets.values():
            for item_id, row in rows.items():
                want = key.get(item_id, {}).get(column, "").strip().lower()
                got = norm(row[field])
                if not want or not got:
                    continue
                total += 1
                matched += want == got
                confusion[want][got] += 1
        if total:
            print()
            print(f"{column} recovery: {matched}/{total} = {100 * matched / total:.1f}%")
            levels = sorted(confusion)
            width = max(len(x) for x in levels) + 2
            print("  intended".ljust(width + 2) + "".join(x.ljust(width) for x in levels))
            for want in levels:
                row_total = sum(confusion[want].values())
                cells = "".join(f"{confusion[want][got]}".ljust(width) for got in levels)
                print(f"  {want.ljust(width)}  {cells} (n={row_total})")


def main():
    key_path = DIR / "key.csv"
    key = {}
    if key_path.exists():
        with key_path.open(encoding="utf-8", newline="") as stream:
            key = {row["item_id"]: row for row in csv.DictReader(stream)}
    else:
        print(f"no key at {key_path}, recovery cannot be computed")

    report("rewrite", REWRITE_FIELDS, key)
    print()
    report("pushback", PUSHBACK_FIELDS, key)


if __name__ == "__main__":
    main()
