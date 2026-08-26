from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "source_csv"
GENERATED_DIR = PROJECT_ROOT / "generated"

INPUT_FILE = SOURCE_DIR / "zero_mask_summary.csv"
OUTPUT_PNG = GENERATED_DIR / "figures" / "figure_zero_mask_comparison.png"
OUTPUT_PDF = GENERATED_DIR / "figures" / "figure_zero_mask_comparison.pdf"
OUTPUT_TABLE = GENERATED_DIR / "tables" / "zero_mask_results_with_delta.csv"


if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

results = pd.read_csv(INPUT_FILE)

required_columns = {
    "method",
    "none",
    "left",
    "face",
    "right",
    "body",
}

missing_columns = required_columns - set(results.columns)
if missing_columns:
    raise ValueError(
        f"The input CSV is missing columns: {sorted(missing_columns)}"
    )

selected_methods = [
    "Baseline e25",
    "Sequence gate from e25",
    "Temporal DWSF v3 from e25",
    "Baseline e30",
]

ordered = (
    results[results["method"].isin(selected_methods)]
    .set_index("method")
    .reindex(selected_methods)
)

if ordered["none"].isna().any():
    missing_methods = ordered[ordered["none"].isna()].index.tolist()
    raise ValueError(f"Could not find results for: {missing_methods}")

mask_columns = ["none", "left", "face", "right", "body"]
delta_results = ordered.copy()

for column in ["left", "face", "right", "body"]:
    delta_results[f"delta_{column}"] = (
        delta_results[column] - delta_results["none"]
    )

OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
delta_results.reset_index().to_csv(OUTPUT_TABLE, index=False)

print("Zero-masking WER results:")
print(ordered[mask_columns].to_string())

print("\nWER degradation relative to the no-mask condition:")
print(
    delta_results[
        ["delta_left", "delta_face", "delta_right", "delta_body"]
    ].to_string()
)

mask_labels = [
    "No mask",
    "Left-hand\nstream masked",
    "Face\nstream masked",
    "Right-hand\nstream masked",
    "Composite\nstream masked",
]
x_positions = list(range(len(mask_labels)))

line_styles = {
    "Baseline e25": ("o", "-"),
    "Sequence gate from e25": ("s", "--"),
    "Temporal DWSF v3 from e25": ("^", "-."),
    "Baseline e30": ("D", ":"),
}

display_labels = {
    "Baseline e25": "Fixed baseline",
    "Sequence gate from e25": "Sequence-level gate",
    "Temporal DWSF v3 from e25": "Temporal DWSF",
    "Baseline e30": "Full-model continuation",
}

fig, ax = plt.subplots(figsize=(9.4, 5.4))

for method in selected_methods:
    marker, linestyle = line_styles[method]
    wer_values = ordered.loc[method, mask_columns].astype(float).tolist()

    ax.plot(
        x_positions,
        wer_values,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.8,
        markersize=6,
        label=display_labels[method],
    )

ax.set_xticks(x_positions)
ax.set_xticklabels(mask_labels)
ax.set_xlabel("Evaluation condition")
ax.set_ylabel("Test word error rate (WER, %)")
ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)

all_values = ordered[mask_columns].to_numpy(dtype=float).flatten()
ax.set_ylim(all_values.min() - 0.5, all_values.max() + 0.7)

ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.subplots_adjust(right=0.75)
fig.tight_layout()

OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"\nSaved PNG: {OUTPUT_PNG}")
print(f"Saved PDF: {OUTPUT_PDF}")
print(f"Saved table: {OUTPUT_TABLE}")
