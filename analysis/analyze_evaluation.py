"""Summarize turn-2 target adoption and framing effects from evaluation logs."""

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


FACTORS = {
    "speaker": ["none", "doctor", "patient", "friend", "relative"],
    "certainty": ["neutral", "assertive", "tentative"],
    "emotion": ["neutral", "anxious", "frustrated"],
}
COMPARISONS = {
    "Gemma 3 by size": ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    "Gemma vs. MedGemma at 27B": ["gemma-3-27b-it", "medgemma-27b-text-it"],
}
MODEL_NAMES = {
    "gemma-3-4b-it": "Gemma 3 4B",
    "gemma-3-12b-it": "Gemma 3 12B",
    "gemma-3-27b-it": "Gemma 3 27B",
    "medgemma-27b-text-it": "MedGemma 27B",
    "llama-3-2-3b-instruct": "Llama 3.2 3B",
    "llama-3-1-8b-instruct": "Llama 3.1 8B",
    "llama-3-3-70b-instruct": "Llama 3.3 70B",
    "llama3-med42-8b": "Med42 8B",
    "llama3-med42-70b": "Med42 70B",
    "mistral-7b-instruct-v0-3": "Mistral 7B",
    "biomistral-7b": "BioMistral 7B",
}


def load_results(root):
    cols = ["job_id", "record_id", "model_tag", "turn1_correct", "correct_letter",
            "target_letter", "final_letter", "label", *FACTORS]
    trials = pd.read_csv(root / "analysis_table.csv", usecols=cols, dtype=str,
                         keep_default_na=False)
    if not trials.turn1_correct.isin(["True", "False"]).all():
        raise ValueError("Invalid turn 1 correctness value")
    trials = trials.loc[trials.turn1_correct.eq("True")].copy()
    trials["parsed"] = trials.final_letter.ne("")
    trials["to_target"] = trials.parsed & trials.final_letter.eq(trials.target_letter)
    kept = trials.parsed & trials.final_letter.eq(trials.correct_letter)
    trials["other_wrong"] = trials.parsed & ~kept & ~trials.to_target
    trials["unparsed"] = ~trials.parsed
    trials["hold"] = trials.label.eq("hold")
    trials["drift"] = trials.label.eq("drift")
    if (trials.empty or trials.duplicated(["model_tag", "job_id"]).any()
            or not trials.label.eq("unparsed").equals(trials.unparsed)
            or not (trials.hold | trials.drift).equals(kept)
            or not trials.label.eq("reverse").equals(trials.parsed & ~kept)):
        raise ValueError("Expected unique, fully judged trials with consistent labels")
    if (trials.correct_letter.eq(trials.target_letter).any()
            or trials[["correct_letter", "target_letter"]].eq("").any().any()
            or trials.groupby("record_id")[["correct_letter", "target_letter"]]
                     .nunique().gt(1).any().any()):
        raise ValueError("Expected one fixed incorrect target per question")
    for factor, levels in FACTORS.items():
        if set(trials[factor]) != set(levels):
            raise ValueError(f"Unexpected {factor} levels")

    rows = []
    for path in sorted(root.glob("*/turn1.jsonl")):
        with path.open(encoding="utf-8") as source:
            for row in map(json.loads, source):
                rows.append({"model_tag": path.parent.name,
                             **{k: row[k] for k in ("record_id", "speaker", "correct_letter",
                                                    "is_correct", "parse_failed")}})
    turn1 = pd.DataFrame(rows)
    if turn1.empty or set(trials.model_tag) != set(turn1.model_tag):
        raise ValueError("Expected turn 1 logs for every model in the analysis table")
    keys = ["model_tag", "record_id", "speaker"]
    source = trials.merge(turn1, on=keys, how="left", validate="many_to_one",
                          suffixes=("", "_turn1"))
    if (not source.is_correct.eq(True).all()
            or not source.correct_letter.eq(source.correct_letter_turn1).all()):
        raise ValueError("Turn 2 row does not match a correct turn 1 answer")
    return turn1, trials


def model_summary(turn1, trials):
    first = turn1.groupby("model_tag").agg(turn1_n=("is_correct", "size"),
                                           turn1_correct_n=("is_correct", "sum"),
                                           turn1_unparsed_n=("parse_failed", "sum"))
    grouped = trials.groupby("model_tag")
    counts = grouped[["parsed", "unparsed", "to_target", "other_wrong", "hold", "drift"]].sum()
    counts = counts.add_suffix("_n")
    results = first.join(grouped.size().rename("turn2_n")).join(counts)
    results["turn1_accuracy_pct"] = 100 * results.turn1_correct_n / results.turn1_n
    results["turn2_unparsed_pct"] = 100 * results.unparsed_n / results.turn2_n
    results["target_adoption_pct"] = 100 * results.to_target_n / results.parsed_n
    results["other_wrong_pct"] = 100 * results.other_wrong_n / results.parsed_n
    results["drift_pct_given_kept"] = 100 * results.drift_n / (results.hold_n + results.drift_n)
    results["target_adoption_lower_pct"] = 100 * results.to_target_n / results.turn2_n
    results["target_adoption_upper_pct"] = (
        100 * (results.to_target_n + results.unparsed_n) / results.turn2_n)
    return results


def matched_models(parsed):
    rows = []
    for comparison, models in COMPARISONS.items():
        group = parsed.loc[parsed.model_tag.isin(models)]
        paired = (group.pivot(index=["job_id", "record_id"], columns="model_tag", values="to_target")
                       .reindex(columns=models).dropna())
        if paired.empty:
            raise ValueError(f"No shared trials for {comparison}")
        for model in models:
            adopted = int(paired[model].sum())
            rows.append(dict(comparison=comparison, model_tag=model, trials=len(paired),
                             questions=paired.index.get_level_values("record_id").nunique(),
                             to_target_n=adopted, target_adoption_pct=100 * adopted / len(paired)))
    return pd.DataFrame(rows)


def framing_summary(parsed):
    rows = []
    for factor, levels in FACTORS.items():
        keys = ["model_tag", "record_id", *(f for f in FACTORS if f != factor)]
        blocks = (parsed.pivot(index=keys, columns=factor, values="to_target")
                        .reindex(columns=levels).dropna())
        reference = levels[0]
        for model, group in blocks.groupby(level="model_tag"):
            rates = 100 * group.mean()
            for level in levels[1:]:
                rows.append(dict(model_tag=model, factor=factor, level=level, reference=reference,
                                 blocks=len(group),
                                 questions=group.index.get_level_values("record_id").nunique(),
                                 target_adoption_pct=rates[level], reference_pct=rates[reference],
                                 difference_pp=rates[level] - rates[reference]))
    return pd.DataFrame(rows)


def plot_models(matched, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.45), sharey=True, layout="constrained")
    for ax, (comparison, group) in zip(axes, matched.groupby("comparison", sort=False)):
        names = group.model_tag.map(MODEL_NAMES).str.rsplit(" ", n=1).str.join("\n")
        colors = ["#147C72" if tag.startswith("medgemma") else "#395B97" for tag in group.model_tag]
        bars = ax.bar(names, group.target_adoption_pct, color=colors, width=0.58)
        ax.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=10)
        ax.set_title(f"{comparison}\n{group.trials.iloc[0]:,} matched trials", fontsize=11)
        ax.set_ylim(0, 104)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color="#DBE1E9", linewidth=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0, pad=6)
    fig.suptitle("Adoption of the user's wrong answer on shared trials", fontsize=14,
                 fontweight="semibold")
    fig.savefig(out, dpi=250)
    plt.close(fig)


def plot_framing(framing, out):
    columns = [(factor, level) for factor, levels in FACTORS.items() for level in levels[1:]]
    matrix = framing.pivot(index="model_tag", columns=["factor", "level"], values="difference_pp")
    order = [m for m in MODEL_NAMES if m in matrix.index]
    order += sorted(set(matrix.index).difference(order))
    matrix = matrix.loc[order, columns]
    limit = max(1, matrix.abs().max().max())
    fig, ax = plt.subplots(figsize=(10.8, 5.45), layout="constrained")
    grid = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(columns)),
                  [f"{level.title()}\nvs. {FACTORS[factor][0].title()}" for factor, level in columns])
    ax.set_yticks(range(len(order)), [MODEL_NAMES.get(m, m) for m in order])
    ax.tick_params(length=0, pad=7)
    for boundary in (3.5, 5.5):
        ax.axvline(boundary, color="white", linewidth=2.5)
    for i, values in enumerate(matrix.to_numpy()):
        for j, value in enumerate(values):
            color = "white" if abs(value) > limit * 0.60 else "#182634"
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", color=color, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Change in target adoption on matched framing trials",
                 fontsize=14, fontweight="semibold", pad=12)
    fig.colorbar(grid, ax=ax, label="Difference (percentage points)", shrink=0.82, pad=0.02)
    fig.savefig(out, dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    turn1, trials = load_results(args.eval_dir)
    parsed = trials.loc[trials.parsed]
    summary = model_summary(turn1, trials)
    matched = matched_models(parsed)
    framing = framing_summary(parsed)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "model_summary.csv", float_format="%.6f")
    matched.to_csv(args.out / "matched_models.csv", index=False, float_format="%.6f")
    framing.to_csv(args.out / "framing_by_model.csv", index=False, float_format="%.6f")
    plot_models(matched, args.out / "matched_models.png")
    plot_framing(framing, args.out / "framing_by_model.png")


if __name__ == "__main__":
    main()
