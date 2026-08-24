from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_ROOT = PROJECT_ROOT / "evidence_bundle"

OUTPUT_PNG_DIR = PROJECT_ROOT / "thesis_figures" / "png"
OUTPUT_PDF_DIR = PROJECT_ROOT / "thesis_figures" / "pdf"
OUTPUT_TABLE = (
    PROJECT_ROOT
    / "thesis_tables"
    / "selected_temporal_examples.csv"
)

COMBINED_PNG = (
    PROJECT_ROOT
    / "thesis_figures"
    / "png"
    / "figure_temporal_scales_three_cases.png"
)

COMBINED_PDF = (
    PROJECT_ROOT
    / "thesis_figures"
    / "pdf"
    / "figure_temporal_scales_three_cases.pdf"
)

OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Locate the frame-level temporal-weight CSV
# ---------------------------------------------------------
candidates = list(
    EVIDENCE_ROOT.rglob("temporal_weights_frame_level.csv")
)

if not candidates:
    raise FileNotFoundError(
        "Could not find temporal_weights_frame_level.csv "
        f"inside {EVIDENCE_ROOT}"
    )

preferred = [
    path for path in candidates
    if "dwsf_v3_from_e25_gateonly_bs2" in str(path)
]

INPUT_FILE = preferred[0] if preferred else sorted(candidates)[0]

print(f"Using temporal-weight file:\n{INPUT_FILE}\n")


# ---------------------------------------------------------
# 3. Load and validate data
# ---------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

required_columns = {
    "name",
    "t",
    "s_left",
    "s_face",
    "s_right",
    "s_body",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"The frame-level CSV is missing columns: "
        f"{sorted(missing_columns)}"
    )

scale_columns = [
    "s_left",
    "s_face",
    "s_right",
    "s_body",
]


# ---------------------------------------------------------
# 4. Calculate sample-level temporal variation
# ---------------------------------------------------------
def population_std(series):
    return series.std(ddof=0)


sample_summary = (
    df.groupby("name")
    .agg(
        num_steps=("t", "count"),
        left_std=("s_left", population_std),
        face_std=("s_face", population_std),
        right_std=("s_right", population_std),
        body_std=("s_body", population_std),
    )
    .reset_index()
)

sample_summary["variation_score"] = sample_summary[
    ["left_std", "face_std", "right_std", "body_std"]
].mean(axis=1)

sample_summary = sample_summary[
    (sample_summary["num_steps"] >= 10)
    & sample_summary["variation_score"].notna()
].copy()

if len(sample_summary) < 3:
    raise ValueError(
        "Fewer than three valid samples are available."
    )


# ---------------------------------------------------------
# 5. Select representative samples using quantiles
# ---------------------------------------------------------
target_quantiles = {
    "main_median": 0.50,
    "appendix_q25": 0.25,
    "appendix_q75": 0.75,
}

selected_rows = []
used_names = set()

for label, quantile in target_quantiles.items():
    target_score = sample_summary["variation_score"].quantile(
        quantile
    )

    ranked = sample_summary.assign(
        distance=(
            sample_summary["variation_score"] - target_score
        ).abs()
    ).sort_values(["distance", "name"])

    chosen = ranked[
        ~ranked["name"].isin(used_names)
    ].iloc[0].copy()

    chosen["selection_label"] = label
    chosen["target_quantile"] = quantile
    chosen["target_score"] = target_score

    selected_rows.append(chosen)
    used_names.add(chosen["name"])

selected = pd.DataFrame(selected_rows)

selected_columns = [
    "selection_label",
    "target_quantile",
    "name",
    "num_steps",
    "variation_score",
    "target_score",
    "left_std",
    "face_std",
    "right_std",
    "body_std",
]

selected[selected_columns].to_csv(
    OUTPUT_TABLE,
    index=False,
)

print("Selected temporal examples:")
print(
    selected[selected_columns].to_string(index=False)
)

selected_names = selected["name"].tolist()

selected_frame_data = df[
    df["name"].isin(selected_names)
]

all_selected_scales = selected_frame_data[
    scale_columns
].to_numpy()

global_lower = min(all_selected_scales.min(), 1.0)
global_upper = max(all_selected_scales.max(), 1.0)

global_margin = max(
    (global_upper - global_lower) * 0.08,
    0.01,
)

COMMON_YMIN = global_lower - global_margin
COMMON_YMAX = global_upper + global_margin

print(
    f"\nCommon y-axis range: "
    f"{COMMON_YMIN:.4f} to {COMMON_YMAX:.4f}"
)

# ---------------------------------------------------------
# 6. Plotting function
# ---------------------------------------------------------
output_names = {
    "main_median": "figure_temporal_scales_representative",
    "appendix_q25": "appendix_temporal_scales_q25",
    "appendix_q75": "appendix_temporal_scales_q75",
}

figure_titles = {
    "main_median": "Temporal stream scaling in a representative test sample",
    "appendix_q25": "Temporal stream scaling in a lower-variation test sample",
    "appendix_q75": "Temporal stream scaling in a higher-variation test sample",
}

line_columns = {
    "Left-hand": "s_left",
    "Face": "s_face",
    "Right-hand": "s_right",
    "Composite": "s_body",
}

combined_panel_order = [
    "appendix_q25",
    "main_median",
    "appendix_q75",
]

combined_panel_labels = {
    "appendix_q25": "(a) Lower variation (q25)",
    "main_median": "(b) Median variation",
    "appendix_q75": "(c) Higher variation (q75)",
}


def plot_combined_figure():
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.6),
        sharey=True,
    )

    for axis, selection_label in zip(
        axes,
        combined_panel_order,
    ):
        matching_rows = selected[
            selected["selection_label"] == selection_label
        ]

        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected exactly one selected sample "
                f"for {selection_label}, but found "
                f"{len(matching_rows)}."
            )

        selection_row = matching_rows.iloc[0]
        sample_name = selection_row["name"]

        sample_df = (
            df[df["name"] == sample_name]
            .sort_values("t")
            .copy()
        )

        for display_name, column_name in line_columns.items():
            axis.plot(
                sample_df["t"],
                sample_df[column_name],
                linewidth=1.7,
                label=display_name,
            )

        axis.axhline(
            y=1.0,
            linestyle="--",
            linewidth=1.0,
        )

        axis.set_ylim(COMMON_YMIN, COMMON_YMAX)

        axis.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )

        axis.set_xlabel(
            "Encoded temporal position",
            fontsize=9,
        )

        axis.grid(
            axis="both",
            linestyle=":",
            linewidth=0.7,
            alpha=0.6,
        )

        axis.text(
            0.03,
            0.96,
            combined_panel_labels[selection_label],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    axes[0].set_ylabel(
        "Temporal scale",
        fontsize=10,
    )

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0.01, 0.02, 1.0, 0.90])

    fig.savefig(COMBINED_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(COMBINED_PDF, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"\nSaved combined PNG: {COMBINED_PNG}")
    print(f"Saved combined PDF: {COMBINED_PDF}")


def plot_sample(selection_row):
    selection_label = selection_row["selection_label"]
    sample_name = selection_row["name"]

    sample_df = (
        df[df["name"] == sample_name]
        .sort_values("t")
        .copy()
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for display_name, column_name in line_columns.items():
        ax.plot(
            sample_df["t"],
            sample_df[column_name],
            linewidth=1.8,
            label=display_name,
        )

    ax.axhline(y=1.0, linestyle="--", linewidth=1.1)
    ax.set_xlabel("Encoded temporal position")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Temporal scale")
    ax.set_title(figure_titles[selection_label])

    ax.grid(axis="both", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
    )

    ax.set_ylim(COMMON_YMIN, COMMON_YMAX)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_stem = output_names[selection_label]
    png_path = OUTPUT_PNG_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_PDF_DIR / f"{output_stem}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"\nSample: {sample_name}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


for _, row in selected.iterrows():
    plot_sample(row)

plot_combined_figure()

print(f"\nSaved selection table: {OUTPUT_TABLE}")
