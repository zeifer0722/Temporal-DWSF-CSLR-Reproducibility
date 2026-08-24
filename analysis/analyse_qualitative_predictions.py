from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "qualitative_data"
    / "experiment_records"
    / "qualitative_analysis"
)

BASELINE_FILE = (
    DATA_DIR
    / "baseline_e25_test_predictions_all_heads.csv"
)

TEMPORAL_FILE = (
    DATA_DIR
    / "temporal_v3_seed0_test_predictions_all_heads.csv"
)

TABLE_DIR = PROJECT_ROOT / "thesis_tables"
PNG_DIR = PROJECT_ROOT / "thesis_figures" / "png"
PDF_DIR = PROJECT_ROOT / "thesis_figures" / "pdf"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

RANKING_OUTPUT = TABLE_DIR / "qualitative_sample_ranking.csv"
SUMMARY_OUTPUT = TABLE_DIR / "qualitative_error_summary.csv"
OUTCOME_OUTPUT = TABLE_DIR / "qualitative_sample_outcomes.csv"
SELECTED_OUTPUT = TABLE_DIR / "qualitative_selected_samples.csv"

FIGURE_PNG = PNG_DIR / "figure_error_type_comparison.png"
FIGURE_PDF = PDF_DIR / "figure_error_type_comparison.pdf"


# ---------------------------------------------------------
# 2. Token utilities
# ---------------------------------------------------------
def split_tokens(value):
    """Convert a gloss string into a token list."""
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return text.split()


def align_tokens(reference_tokens, hypothesis_tokens):
    """
    Compute deterministic Levenshtein alignment.

    Operations:
    M = match
    S = substitution
    D = deletion
    I = insertion

    Tie-breaking prefers substitution, then deletion,
    then insertion. This keeps the error decomposition
    reproducible.
    """
    n = len(reference_tokens)
    m = len(hypothesis_tokens)

    distance = [[0] * (m + 1) for _ in range(n + 1)]
    backtrace = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        distance[i][0] = i
        backtrace[i][0] = "D"

    for j in range(1, m + 1):
        distance[0][j] = j
        backtrace[0][j] = "I"

    priority = {
        "M": 0,
        "S": 0,
        "D": 1,
        "I": 2,
    }

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference_tokens[i - 1] == hypothesis_tokens[j - 1]:
                diagonal_operation = "M"
                diagonal_cost = distance[i - 1][j - 1]
            else:
                diagonal_operation = "S"
                diagonal_cost = distance[i - 1][j - 1] + 1

            candidates = [
                (diagonal_cost, diagonal_operation),
                (distance[i - 1][j] + 1, "D"),
                (distance[i][j - 1] + 1, "I"),
            ]

            best_cost, best_operation = min(
                candidates,
                key=lambda item: (item[0], priority[item[1]]),
            )

            distance[i][j] = best_cost
            backtrace[i][j] = best_operation

    aligned = []
    i = n
    j = m

    while i > 0 or j > 0:
        operation = backtrace[i][j]

        if operation == "M":
            aligned.append((reference_tokens[i - 1], hypothesis_tokens[j - 1], "M"))
            i -= 1
            j -= 1
        elif operation == "S":
            aligned.append((reference_tokens[i - 1], hypothesis_tokens[j - 1], "S"))
            i -= 1
            j -= 1
        elif operation == "D":
            aligned.append((reference_tokens[i - 1], None, "D"))
            i -= 1
        elif operation == "I":
            aligned.append((None, hypothesis_tokens[j - 1], "I"))
            j -= 1
        else:
            raise RuntimeError(f"Invalid alignment state at i={i}, j={j}")

    aligned.reverse()
    return aligned


def count_operations(alignment):
    counts = {
        "matches": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }

    for _, _, operation in alignment:
        if operation == "M":
            counts["matches"] += 1
        elif operation == "S":
            counts["substitutions"] += 1
        elif operation == "D":
            counts["deletions"] += 1
        elif operation == "I":
            counts["insertions"] += 1

    counts["total_errors"] = (
        counts["substitutions"]
        + counts["deletions"]
        + counts["insertions"]
    )

    return counts


# ---------------------------------------------------------
# 3. Load and validate prediction files
# ---------------------------------------------------------
if not BASELINE_FILE.exists():
    raise FileNotFoundError(f"Cannot find baseline predictions: {BASELINE_FILE}")

if not TEMPORAL_FILE.exists():
    raise FileNotFoundError(f"Cannot find temporal predictions: {TEMPORAL_FILE}")

baseline = pd.read_csv(BASELINE_FILE)
temporal = pd.read_csv(TEMPORAL_FILE)

required_columns = {
    "name",
    "reference",
    "reference_length",
    "fuse",
}

for label, dataframe in [("baseline", baseline), ("temporal", temporal)]:
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{label} CSV is missing columns: {sorted(missing)}"
        )

if baseline["name"].tolist() != temporal["name"].tolist():
    raise ValueError(
        "The two prediction files contain different sample names or sample orders."
    )

baseline_references = baseline["reference"].fillna("").astype(str)
temporal_references = temporal["reference"].fillna("").astype(str)

if not baseline_references.equals(temporal_references):
    raise ValueError("Ground-truth references do not match.")

print(f"Loaded {len(baseline)} matched test samples.")


# ---------------------------------------------------------
# 4. Per-sample edit-error analysis
# ---------------------------------------------------------
records = []

baseline_totals = {
    "substitutions": 0,
    "deletions": 0,
    "insertions": 0,
    "matches": 0,
    "total_errors": 0,
}

temporal_totals = {
    "substitutions": 0,
    "deletions": 0,
    "insertions": 0,
    "matches": 0,
    "total_errors": 0,
}

total_reference_tokens = 0

for index in range(len(baseline)):
    name = baseline.loc[index, "name"]
    reference = baseline.loc[index, "reference"]

    baseline_prediction = baseline.loc[index, "fuse"]
    temporal_prediction = temporal.loc[index, "fuse"]

    reference_tokens = split_tokens(reference)
    baseline_tokens = split_tokens(baseline_prediction)
    temporal_tokens = split_tokens(temporal_prediction)

    total_reference_tokens += len(reference_tokens)

    baseline_alignment = align_tokens(reference_tokens, baseline_tokens)
    temporal_alignment = align_tokens(reference_tokens, temporal_tokens)

    baseline_counts = count_operations(baseline_alignment)
    temporal_counts = count_operations(temporal_alignment)

    for key in baseline_totals:
        baseline_totals[key] += baseline_counts[key]
        temporal_totals[key] += temporal_counts[key]

    error_reduction = baseline_counts["total_errors"] - temporal_counts["total_errors"]

    if error_reduction > 0:
        outcome = "improved"
    elif error_reduction < 0:
        outcome = "worse"
    else:
        outcome = "equal"

    records.append(
        {
            "name": name,
            "reference_length": len(reference_tokens),
            "reference": "" if pd.isna(reference) else str(reference),
            "baseline_prediction": "" if pd.isna(baseline_prediction) else str(baseline_prediction),
            "temporal_prediction": "" if pd.isna(temporal_prediction) else str(temporal_prediction),
            "baseline_substitutions": baseline_counts["substitutions"],
            "baseline_deletions": baseline_counts["deletions"],
            "baseline_insertions": baseline_counts["insertions"],
            "baseline_total_errors": baseline_counts["total_errors"],
            "temporal_substitutions": temporal_counts["substitutions"],
            "temporal_deletions": temporal_counts["deletions"],
            "temporal_insertions": temporal_counts["insertions"],
            "temporal_total_errors": temporal_counts["total_errors"],
            "error_reduction": error_reduction,
            "deletion_reduction": baseline_counts["deletions"] - temporal_counts["deletions"],
            "substitution_reduction": baseline_counts["substitutions"] - temporal_counts["substitutions"],
            "insertion_reduction": baseline_counts["insertions"] - temporal_counts["insertions"],
            "outcome": outcome,
            "prediction_changed": baseline_tokens != temporal_tokens,
        }
    )


# ---------------------------------------------------------
# 5. Save ranked sample table
# ---------------------------------------------------------
ranking = pd.DataFrame(records)
ranking = ranking.sort_values(
    by=["error_reduction", "reference_length", "name"],
    ascending=[False, False, True],
)
ranking.to_csv(RANKING_OUTPUT, index=False)


# ---------------------------------------------------------
# 6. Save global error summary
# ---------------------------------------------------------
baseline_wer = baseline_totals["total_errors"] / total_reference_tokens * 100
temporal_wer = temporal_totals["total_errors"] / total_reference_tokens * 100

summary = pd.DataFrame(
    [
        {
            "model": "Baseline e25",
            "substitutions": baseline_totals["substitutions"],
            "deletions": baseline_totals["deletions"],
            "insertions": baseline_totals["insertions"],
            "total_errors": baseline_totals["total_errors"],
            "reference_tokens": total_reference_tokens,
            "wer_percent": baseline_wer,
        },
        {
            "model": "Temporal DWSF v3 seed 0",
            "substitutions": temporal_totals["substitutions"],
            "deletions": temporal_totals["deletions"],
            "insertions": temporal_totals["insertions"],
            "total_errors": temporal_totals["total_errors"],
            "reference_tokens": total_reference_tokens,
            "wer_percent": temporal_wer,
        },
    ]
)

summary.to_csv(SUMMARY_OUTPUT, index=False)


# ---------------------------------------------------------
# 7. Save sample-outcome counts
# ---------------------------------------------------------
outcome_counts = (
    ranking["outcome"]
    .value_counts()
    .reindex(["improved", "equal", "worse"], fill_value=0)
    .rename_axis("outcome")
    .reset_index(name="number_of_samples")
)

outcome_counts.to_csv(OUTCOME_OUTPUT, index=False)


# ---------------------------------------------------------
# 8. Select qualitative examples by outcome
# ---------------------------------------------------------
def select_candidate(candidates, sort_columns, ascending, selection_label, selection_reason):
    if candidates.empty:
        raise ValueError(f"No eligible sample found for {selection_label}")

    selected_row = (
        candidates.sort_values(by=sort_columns, ascending=ascending).iloc[0].copy()
    )
    selected_row["selection_label"] = selection_label
    selected_row["selection_reason"] = selection_reason
    return selected_row


improved_row = select_candidate(
    ranking[ranking["outcome"] == "improved"],
    sort_columns=["error_reduction", "reference_length", "name"],
    ascending=[False, False, True],
    selection_label="main_improved",
    selection_reason="Largest edit-distance reduction; longest reference among tied candidates",
)

equal_changed_row = select_candidate(
    ranking[(ranking["outcome"] == "equal") & (ranking["prediction_changed"])],
    sort_columns=["reference_length", "name"],
    ascending=[False, True],
    selection_label="appendix_equal_changed",
    selection_reason="Unchanged edit distance with a changed prediction; longest reference among eligible candidates",
)

worse_row = select_candidate(
    ranking[ranking["outcome"] == "worse"],
    sort_columns=["error_reduction", "reference_length", "name"],
    ascending=[True, False, True],
    selection_label="appendix_worse",
    selection_reason="Largest edit-distance increase; longest reference among tied candidates",
)

selected_examples = pd.DataFrame([improved_row, equal_changed_row, worse_row])
selected_examples.to_csv(SELECTED_OUTPUT, index=False)


# ---------------------------------------------------------
# 9. Create error-type comparison figure
# ---------------------------------------------------------
categories = ["Deletions", "Substitutions", "Insertions"]

baseline_values = [
    baseline_totals["deletions"],
    baseline_totals["substitutions"],
    baseline_totals["insertions"],
]

temporal_values = [
    temporal_totals["deletions"],
    temporal_totals["substitutions"],
    temporal_totals["insertions"],
]

x_positions = np.arange(len(categories))
bar_width = 0.36

fig, ax = plt.subplots(figsize=(9.2, 5.4))

baseline_bars = ax.bar(
    x_positions - bar_width / 2,
    baseline_values,
    bar_width,
    label="Baseline e25",
)

temporal_bars = ax.bar(
    x_positions + bar_width / 2,
    temporal_values,
    bar_width,
    label="Temporal DWSF",
)

ax.set_xticks(x_positions)
ax.set_xticklabels(categories)
ax.set_ylabel("Number of edit errors")
ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
ax.legend(frameon=False, loc="upper right")

maximum_value = max(baseline_values + temporal_values)
ax.set_ylim(0, maximum_value * 1.14)

for bar in baseline_bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + maximum_value * 0.012,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

for bar in temporal_bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + maximum_value * 0.030,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

fig.tight_layout()
fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
fig.savefig(FIGURE_PDF, bbox_inches="tight")
plt.show()
plt.close(fig)


# ---------------------------------------------------------
# 10. Print results
# ---------------------------------------------------------
print("\nGlobal error summary")
print("-" * 80)
print(summary.to_string(index=False))

print("\nSample-level outcomes")
print("-" * 80)
print(outcome_counts.to_string(index=False))

print("\nTop improved candidate samples")
print("-" * 80)

display_columns = [
    "name",
    "reference_length",
    "baseline_total_errors",
    "temporal_total_errors",
    "error_reduction",
    "deletion_reduction",
    "substitution_reduction",
    "insertion_reduction",
]

print(ranking[display_columns].head(15).to_string(index=False))

print("\nSelected qualitative examples")
print("-" * 80)

selected_display_columns = [
    "selection_label",
    "name",
    "reference_length",
    "baseline_total_errors",
    "temporal_total_errors",
    "error_reduction",
    "outcome",
    "prediction_changed",
]

print(selected_examples[selected_display_columns].to_string(index=False))
print(f"Saved selected examples: {SELECTED_OUTPUT}")
print(f"\nSaved ranking: {RANKING_OUTPUT}")
print(f"Saved summary: {SUMMARY_OUTPUT}")
print(f"Saved outcomes: {OUTCOME_OUTPUT}")
print(f"Saved PNG: {FIGURE_PNG}")
print(f"Saved PDF: {FIGURE_PDF}")
