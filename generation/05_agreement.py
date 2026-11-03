"""Score the returned annotation sheets.

Two numbers.

  agreement  how often two labelers gave the same label, as raw agreement and
             Cohen's kappa. Kappa corrects for agreement expected by chance,
             which matters when one answer dominates: two labelers answering
             YES 90% of the time agree 81% of the time by chance alone. Above
             0.6 is usually called substantial, above 0.8 near perfect. Kappa
             is unstable when one label dominates, so read it next to raw
             agreement rather than on its own.

  recovery   for certainty and emotion, how often a labeler chose the level
             the generator was asked for. This is the manipulation check. If
             labelers cannot tell an assertive pushback from a tentative one,
             certainty is not functioning as an experimental variable.

Since all three labelers label every item, agreement is reported for each pair
and pooled, and recovery uses the majority label where two or more agree.

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
LABELERS = ["labeler1", "labeler2", "labeler3"]

REWRITE_FIELDS = ["speaker_correct_YES_NO", "content_preserved_YES_NO"]
PUSHBACK_FIELDS = ["certainty_TENTATIVE_NEUTRAL_ASSERTIVE",
                   "emotion_NEUTRAL_ANXIOUS_FRUSTRATED",
                   "argues_target_YES_NO"]

# which key column holds the level each label field should recover
RECOVERS = {"certainty_TENTATIVE_NEUTRAL_ASSERTIVE": "certainty",
            "emotion_NEUTRAL_ANXIOUS_FRUSTRATED": "emotion"}


def kappa(pairs):
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


def load(kind):
    sheets = {}
    for name in LABELERS:
        path = DIR / f"{kind}_{name}.csv"
        if path.exists():
            with path.open(encoding="utf-8", newline="") as stream:
                sheets[name] = {row["item_id"]: row for row in csv.DictReader(stream)}
    return sheets


def norm(value):
    return value.strip().upper()


def majority(values):
    """The label at least two labelers chose, or None."""
    counts = Counter(v for v in values if v)
    if counts and counts.most_common(1)[0][1] >= 2:
        return counts.most_common(1)[0][0]
    return None


def report(kind, fields, key):
    sheets = load(kind)
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
        print(field)
        pooled = []
        for left, right in combinations(sorted(sheets), 2):
            shared = set(sheets[left]) & set(sheets[right])
            pairs = [(norm(sheets[left][i][field]), norm(sheets[right][i][field]))
                     for i in shared
                     if norm(sheets[left][i][field]) and norm(sheets[right][i][field])]
            if not pairs:
                continue
            observed, k = kappa(pairs)
            print(f"  {left} vs {right}: n={len(pairs):4d}  agreement {observed:.2f}  kappa {k:.2f}")
            pooled += pairs
        if pooled:
            observed, k = kappa(pooled)
            print(f"  pooled:              n={len(pooled):4d}  agreement {observed:.2f}  kappa {k:.2f}")

        counts = Counter(norm(row[field]) for rows in sheets.values()
                         for row in rows.values() if norm(row[field]))
        print(f"  labels used: {dict(sorted(counts.items()))}")

        column = RECOVERS.get(field)
        if not column or not key:
            continue
        matched = total = 0
        confusion = defaultdict(Counter)
        for item_id in sorted(set().union(*(set(s) for s in sheets.values()))):
            want = key.get(item_id, {}).get(column, "").strip().upper()
            got = majority(norm(s[item_id][field]) for s in sheets.values()
                           if item_id in s)
            if not want or not got:
                continue
            total += 1
            matched += want == got
            confusion[want][got] += 1
        if total:
            print()
            print(f"  {column} recovery (majority label): "
                  f"{matched}/{total} = {100 * matched / total:.1f}%")
            levels = sorted(set(confusion) | {g for c in confusion.values() for g in c})
            width = max(len(x) for x in levels) + 2
            print("    asked for".ljust(width + 4) + "".join(x.ljust(width) for x in levels))
            for want in levels:
                if want not in confusion:
                    continue
                cells = "".join(str(confusion[want][got]).ljust(width) for got in levels)
                print(f"    {want.ljust(width)}  {cells}")


def main():
    key_path = DIR / "key.csv"
    key = {}
    if key_path.exists():
        with key_path.open(encoding="utf-8", newline="") as stream:
            key = {row["item_id"]: row for row in csv.DictReader(stream)}
    else:
        print(f"no key at {key_path}, recovery cannot be computed")

    report("rewrites", REWRITE_FIELDS, key)
    print()
    report("pushbacks", PUSHBACK_FIELDS, key)


if __name__ == "__main__":
    main()