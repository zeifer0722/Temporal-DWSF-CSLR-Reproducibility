from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIAGNOSTIC_DIR = PROJECT_ROOT / "data" / "raw_diagnostics"
COMMITTED_DERIVED_DIR = PROJECT_ROOT / "data" / "derived_tables"
GENERATED_DIR = PROJECT_ROOT / "generated"

BASELINE_FILE = RAW_DIAGNOSTIC_DIR / "baseline_e25_test_predictions_all_heads.csv"
TEMPORAL_FILE = RAW_DIAGNOSTIC_DIR / "temporal_v3_seed0_test_predictions_all_heads.csv"
GENERATED_SELECTED_FILE = GENERATED_DIR / "tables" / "qualitative_selected_samples.csv"
COMMITTED_SELECTED_FILE = COMMITTED_DERIVED_DIR / "qualitative_selected_samples.csv"

FIGURE_DIR = GENERATED_DIR / "figures"
TABLE_DIR = GENERATED_DIR / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

ALIGNMENT_TABLE = TABLE_DIR / "qualitative_selected_alignments.csv"
SAMPLE_SUMMARY_TABLE = TABLE_DIR / "qualitative_selected_samples_summary.csv"

SELECTION_ORDER = [
    "main_improved",
    "appendix_equal_changed",
    "appendix_worse",
]

MAX_COLUMNS_PER_BLOCK = 9
ROW_LABELS = ["Reference", "Baseline e25", "Temporal DWSF"]
ROW_Y = {"Reference": 2, "Baseline e25": 1, "Temporal DWSF": 0}

CELL_COLOURS = {
    "reference": "#E6E6E6",
    "M": "#D9EAD3",
    "S": "#FCE5CD",
    "D": "#F4CCCC",
    "I": "#D9D2E9",
    "blank": "#FFFFFF",
}
EDGE_COLOUR = "#666666"


def split_tokens(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    return text.split() if text else []


def align_tokens(reference_tokens, hypothesis_tokens):
    """Return a deterministic unit-cost Levenshtein alignment."""
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

    priority = {"M": 0, "S": 0, "D": 1, "I": 2}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = (
                distance[i - 1][j - 1],
                "M",
            ) if reference_tokens[i - 1] == hypothesis_tokens[j - 1] else (
                distance[i - 1][j - 1] + 1,
                "S",
            )
            candidates = [
                diagonal,
                (distance[i - 1][j] + 1, "D"),
                (distance[i][j - 1] + 1, "I"),
            ]
            best_cost, best_operation = min(
                candidates,
                key=lambda item: (item[0], priority[item[1]]),
            )
            distance[i][j] = best_cost
            backtrace[i][j] = best_operation

    alignment = []
    i = n
    j = m
    while i > 0 or j > 0:
        operation = backtrace[i][j]
        if operation == "M":
            alignment.append((reference_tokens[i - 1], hypothesis_tokens[j - 1], "M"))
            i -= 1
            j -= 1
        elif operation == "S":
            alignment.append((reference_tokens[i - 1], hypothesis_tokens[j - 1], "S"))
            i -= 1
            j -= 1
        elif operation == "D":
            alignment.append((reference_tokens[i - 1], None, "D"))
            i -= 1
        elif operation == "I":
            alignment.append((None, hypothesis_tokens[j - 1], "I"))
            j -= 1
        else:
            raise RuntimeError(f"Invalid alignment state: i={i}, j={j}")

    alignment.reverse()
    return alignment


def count_operations(alignment):
    counts = {
        "matches": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }
    operation_to_key = {
        "M": "matches",
        "S": "substitutions",
        "D": "deletions",
        "I": "insertions",
    }
    for _, _, operation in alignment:
        counts[operation_to_key[operation]] += 1
    counts["total_errors"] = (
        counts["substitutions"] + counts["deletions"] + counts["insertions"]
    )
    return counts


def alignment_to_reference_map(alignment, reference_tokens):
    number_of_reference_tokens = len(reference_tokens)
    insertions = {position: [] for position in range(number_of_reference_tokens + 1)}
    reference_cells = [None] * number_of_reference_tokens
    reference_position = 0

    for reference_token, hypothesis_token, operation in alignment:
        if operation == "I":
            insertions[reference_position].append(hypothesis_token)
            continue

        if reference_position >= number_of_reference_tokens:
            raise RuntimeError("Reference alignment position exceeded the reference length.")

        expected_reference = reference_tokens[reference_position]
        if reference_token != expected_reference:
            raise RuntimeError("Reference token mismatch during alignment merging.")

        reference_cells[reference_position] = {
            "token": "EMPTY" if hypothesis_token is None else hypothesis_token,
            "operation": operation,
        }
        reference_position += 1

    if reference_position != number_of_reference_tokens:
        raise RuntimeError("Not all reference tokens were aligned.")

    return insertions, reference_cells


def build_common_columns(reference_tokens, baseline_alignment, temporal_alignment):
    baseline_insertions, baseline_cells = alignment_to_reference_map(
        baseline_alignment,
        reference_tokens,
    )
    temporal_insertions, temporal_cells = alignment_to_reference_map(
        temporal_alignment,
        reference_tokens,
    )

    columns = []
    number_of_reference_tokens = len(reference_tokens)

    for gap_position in range(number_of_reference_tokens + 1):
        baseline_gap = baseline_insertions[gap_position]
        temporal_gap = temporal_insertions[gap_position]
        insertion_columns = max(len(baseline_gap), len(temporal_gap))

        for insertion_index in range(insertion_columns):
            baseline_token = (
                baseline_gap[insertion_index]
                if insertion_index < len(baseline_gap)
                else ""
            )
            temporal_token = (
                temporal_gap[insertion_index]
                if insertion_index < len(temporal_gap)
                else ""
            )
            columns.append(
                {
                    "column_type": "insertion",
                    "reference_token": "-",
                    "baseline_token": baseline_token,
                    "baseline_operation": "I" if baseline_token else "blank",
                    "temporal_token": temporal_token,
                    "temporal_operation": "I" if temporal_token else "blank",
                }
            )

        if gap_position < number_of_reference_tokens:
            baseline_cell = baseline_cells[gap_position]
            temporal_cell = temporal_cells[gap_position]
            columns.append(
                {
                    "column_type": "reference",
                    "reference_token": reference_tokens[gap_position],
                    "baseline_token": baseline_cell["token"],
                    "baseline_operation": baseline_cell["operation"],
                    "temporal_token": temporal_cell["token"],
                    "temporal_operation": temporal_cell["operation"],
                }
            )

    return columns


def draw_cell(axis, column_index, row_y, token, operation, is_reference=False):
    face_colour = CELL_COLOURS["reference"] if is_reference else CELL_COLOURS[operation]
    rectangle = Rectangle(
        (column_index + 0.04, row_y - 0.34),
        0.92,
        0.68,
        facecolor=face_colour,
        edgecolor=EDGE_COLOUR,
        linewidth=0.7,
    )
    axis.add_patch(rectangle)
    axis.text(
        column_index + 0.50,
        row_y,
        token,
        ha="center",
        va="center",
        fontsize=9.0,
    )


def draw_recovered_deletion_highlight(axis, column_index):
    bottom = ROW_Y["Temporal DWSF"] - 0.40
    top = ROW_Y["Baseline e25"] + 0.40
    axis.add_patch(
        Rectangle(
            (column_index + 0.01, bottom),
            0.98,
            top - bottom,
            fill=False,
            edgecolor="black",
            linewidth=1.8,
        )
    )


def require_inputs():
    missing = [path for path in [BASELINE_FILE, TEMPORAL_FILE] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "The retained per-sample prediction exports are required. "
            f"Missing: {missing_text}. Place them in data/raw_diagnostics/."
        )


require_inputs()
baseline = pd.read_csv(BASELINE_FILE).set_index("name")
temporal = pd.read_csv(TEMPORAL_FILE).set_index("name")

SELECTED_SAMPLE_FILE = (
    GENERATED_SELECTED_FILE if GENERATED_SELECTED_FILE.exists() else COMMITTED_SELECTED_FILE
)
if not SELECTED_SAMPLE_FILE.exists():
    raise FileNotFoundError(
        "Cannot find the selected qualitative-sample table. Run "
        "analyse_qualitative_predictions.py or restore data/derived_tables/qualitative_selected_samples.csv."
    )

selected_df = pd.read_csv(SELECTED_SAMPLE_FILE)
required_selection_columns = {"selection_label", "name", "selection_reason"}
missing_selection_columns = required_selection_columns - set(selected_df.columns)
if missing_selection_columns:
    raise ValueError(
        f"Selected-sample table is missing columns: {sorted(missing_selection_columns)}"
    )

selected_specs = []
for selection_label in SELECTION_ORDER:
    matching_rows = selected_df[selected_df["selection_label"] == selection_label]
    if len(matching_rows) != 1:
        raise ValueError(
            f"Expected exactly one row for {selection_label}, but found {len(matching_rows)}."
        )

    selected_row = matching_rows.iloc[0]
    sample_name = selected_row["name"]
    sample_id = str(sample_name).rsplit("-", 1)[-1]

    if sample_name not in baseline.index or sample_name not in temporal.index:
        raise KeyError(f"Selected sample is missing from prediction exports: {sample_name}")

    selected_specs.append(
        {
            "selection_label": selection_label,
            "name": sample_name,
            "short_name": f"Sample {sample_id}",
            "output_name": f"qualitative_alignment_{sample_id}",
            "selection_reason": selected_row["selection_reason"],
        }
    )

print("\nAutomatically selected qualitative samples")
print("-" * 80)
for specification in selected_specs:
    print(f"{specification['selection_label']}: {specification['name']}")


def plot_sample(sample_specification):
    sample_name = sample_specification["name"]
    reference_tokens = split_tokens(baseline.loc[sample_name, "reference"])
    baseline_tokens = split_tokens(baseline.loc[sample_name, "fuse"])
    temporal_tokens = split_tokens(temporal.loc[sample_name, "fuse"])

    baseline_alignment = align_tokens(reference_tokens, baseline_tokens)
    temporal_alignment = align_tokens(reference_tokens, temporal_tokens)
    baseline_counts = count_operations(baseline_alignment)
    temporal_counts = count_operations(temporal_alignment)
    columns = build_common_columns(reference_tokens, baseline_alignment, temporal_alignment)

    number_of_blocks = math.ceil(len(columns) / MAX_COLUMNS_PER_BLOCK)
    figure_height = max(5.2, 2.9 * number_of_blocks + 1.7)
    fig, axes = plt.subplots(
        number_of_blocks,
        1,
        figsize=(15.5, figure_height),
        squeeze=False,
    )
    axes = axes.flatten()
    recovered_deletion_count = 0

    for block_index, axis in enumerate(axes):
        start = block_index * MAX_COLUMNS_PER_BLOCK
        end = min(start + MAX_COLUMNS_PER_BLOCK, len(columns))
        block_columns = columns[start:end]

        for local_column_index, column in enumerate(block_columns):
            draw_cell(
                axis,
                local_column_index,
                ROW_Y["Reference"],
                column["reference_token"],
                "reference",
                is_reference=True,
            )
            draw_cell(
                axis,
                local_column_index,
                ROW_Y["Baseline e25"],
                column["baseline_token"],
                column["baseline_operation"],
            )
            draw_cell(
                axis,
                local_column_index,
                ROW_Y["Temporal DWSF"],
                column["temporal_token"],
                column["temporal_operation"],
            )

            if column["baseline_operation"] == "D" and column["temporal_operation"] == "M":
                draw_recovered_deletion_highlight(axis, local_column_index)
                recovered_deletion_count += 1

        axis.set_xlim(-2.35, len(block_columns) + 0.10)
        axis.set_ylim(-0.55, 2.55)
        axis.axis("off")

        for row_label in ROW_LABELS:
            axis.text(
                -0.12,
                ROW_Y[row_label],
                row_label,
                ha="right",
                va="center",
                fontsize=10.5,
                fontweight="bold",
            )

        axis.text(
            -2.25,
            2.43,
            f"Columns {start + 1}-{end}",
            ha="left",
            va="center",
            fontsize=8.5,
        )

    legend_handles = [
        Patch(facecolor=CELL_COLOURS["reference"], edgecolor=EDGE_COLOUR, label="Reference token"),
        Patch(facecolor=CELL_COLOURS["M"], edgecolor=EDGE_COLOUR, label="Match"),
        Patch(facecolor=CELL_COLOURS["S"], edgecolor=EDGE_COLOUR, label="Substitution"),
        Patch(facecolor=CELL_COLOURS["D"], edgecolor=EDGE_COLOUR, label="Deletion"),
        Patch(facecolor=CELL_COLOURS["I"], edgecolor=EDGE_COLOUR, label="Insertion"),
        Patch(facecolor="none", edgecolor="black", linewidth=1.8, label="Recovered deletion"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=6,
        frameon=False,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0.02, 0.03, 1.00, 0.88])

    output_png = FIGURE_DIR / f"{sample_specification['output_name']}.png"
    output_pdf = FIGURE_DIR / f"{sample_specification['output_name']}.pdf"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    alignment_rows = [
        {"sample_name": sample_name, "column_index": index, **column}
        for index, column in enumerate(columns, start=1)
    ]
    summary_row = {
        "sample_name": sample_name,
        "selection_label": sample_specification["selection_label"],
        "short_name": sample_specification["short_name"],
        "selection_reason": sample_specification["selection_reason"],
        "reference_length": len(reference_tokens),
        "baseline_substitutions": baseline_counts["substitutions"],
        "baseline_deletions": baseline_counts["deletions"],
        "baseline_insertions": baseline_counts["insertions"],
        "baseline_total_errors": baseline_counts["total_errors"],
        "temporal_substitutions": temporal_counts["substitutions"],
        "temporal_deletions": temporal_counts["deletions"],
        "temporal_insertions": temporal_counts["insertions"],
        "temporal_total_errors": temporal_counts["total_errors"],
        "error_reduction": baseline_counts["total_errors"] - temporal_counts["total_errors"],
        "recovered_deletions": recovered_deletion_count,
        "reference": " ".join(reference_tokens),
        "baseline_prediction": " ".join(baseline_tokens),
        "temporal_prediction": " ".join(temporal_tokens),
        "png_file": str(output_png.relative_to(PROJECT_ROOT)),
        "pdf_file": str(output_pdf.relative_to(PROJECT_ROOT)),
    }

    print("\n" + "=" * 80)
    print(sample_specification["short_name"])
    print(f"Selection role: {sample_specification['selection_label']}")
    print(f"Sample: {sample_name}")
    print(f"Baseline errors: {baseline_counts['total_errors']}")
    print(f"Temporal errors: {temporal_counts['total_errors']}")
    print(f"Recovered baseline deletions: {recovered_deletion_count}")
    print(f"Saved PNG: {output_png}")
    print(f"Saved PDF: {output_pdf}")

    return alignment_rows, summary_row


all_alignment_rows = []
all_summary_rows = []
for sample_specification in selected_specs:
    alignment_rows, summary_row = plot_sample(sample_specification)
    all_alignment_rows.extend(alignment_rows)
    all_summary_rows.append(summary_row)

pd.DataFrame(all_alignment_rows).to_csv(ALIGNMENT_TABLE, index=False)
pd.DataFrame(all_summary_rows).to_csv(SAMPLE_SUMMARY_TABLE, index=False)

print("\n" + "=" * 80)
print("Qualitative alignment export completed")
print(f"Saved alignment table: {ALIGNMENT_TABLE}")
print(f"Saved selected-sample summary: {SAMPLE_SUMMARY_TABLE}")
print("=" * 80)
