from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_csv"
GENERATED_DIR = ROOT / "generated"

MAIN_RESULTS_FILE = SOURCE_DIR / "main_results.csv"
MULTISEED_FILE = SOURCE_DIR / "multiseed_raw_results.csv"
OUTPUT_PNG = GENERATED_DIR / "figures" / "figure_controlled_wer_multiseed.png"
OUTPUT_PDF = GENERATED_DIR / "figures" / "figure_controlled_wer_multiseed.pdf"
OUTPUT_TABLE = GENERATED_DIR / "tables" / "figure_controlled_wer_multiseed_data.csv"


for file_path in [MAIN_RESULTS_FILE, MULTISEED_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find required input file: {file_path}")

main_results = pd.read_csv(MAIN_RESULTS_FILE)
raw_results = pd.read_csv(MULTISEED_FILE)

baseline_rows = main_results[main_results["model"] == "Baseline e25"]
if len(baseline_rows) != 1:
    raise ValueError("Expected exactly one Baseline e25 row.")

baseline_wer = float(baseline_rows.iloc[0]["test_wer"])

required_methods = ["Sequence gate", "Temporal DWSF v3"]
required_seeds = [0, 123, 3407]

selected = raw_results[raw_results["method"].isin(required_methods)].copy()
selected = selected[selected["seed"].isin(required_seeds)]

if len(selected) != 6:
    raise ValueError("Expected six matched gate results across three seeds.")

for method in required_methods:
    method_seeds = sorted(selected.loc[selected["method"] == method, "seed"].astype(int).tolist())
    if method_seeds != required_seeds:
        raise ValueError(f"Unexpected seed set for {method}: {method_seeds}")

method_display = {
    "Sequence gate": "Sequence-level gate",
    "Temporal DWSF v3": "Temporal DWSF",
}
selected["display_method"] = selected["method"].map(method_display)

summary_rows = [
    {
        "method": "Fixed baseline",
        "n_seeds": 1,
        "mean_test_wer": baseline_wer,
        "sample_sd_test_wer": 0.0,
        "minimum_test_wer": baseline_wer,
        "maximum_test_wer": baseline_wer,
        "improvement_over_baseline": 0.0,
    }
]

for method, group in selected.groupby("display_method"):
    values = group["test_wer"].to_numpy(dtype=float)
    mean_wer = float(np.mean(values))
    sd_wer = float(np.std(values, ddof=1))

    summary_rows.append(
        {
            "method": method,
            "n_seeds": len(values),
            "mean_test_wer": mean_wer,
            "sample_sd_test_wer": sd_wer,
            "minimum_test_wer": float(np.min(values)),
            "maximum_test_wer": float(np.max(values)),
            "improvement_over_baseline": float(baseline_wer - mean_wer),
        }
    )

summary_df = pd.DataFrame(summary_rows)
OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(OUTPUT_TABLE, index=False)

print("\nRaw matched-seed values:")
print(selected[["display_method", "seed", "test_wer"]].to_string(index=False))
print("\nSummary statistics:")
print(summary_df.to_string(index=False))

y_positions = {
    "Fixed baseline": 2,
    "Sequence-level gate": 1,
    "Temporal DWSF": 0,
}

seq_row = summary_df.loc[summary_df["method"] == "Sequence-level gate"].iloc[0]
temp_row = summary_df.loc[summary_df["method"] == "Temporal DWSF"].iloc[0]

seq_mean = float(seq_row["mean_test_wer"])
seq_sd = float(seq_row["sample_sd_test_wer"])
temp_mean = float(temp_row["mean_test_wer"])
temp_sd = float(temp_row["sample_sd_test_wer"])

fig, ax = plt.subplots(figsize=(10.2, 4.8))

ax.plot(
    baseline_wer,
    y_positions["Fixed baseline"],
    marker="o",
    markersize=7,
    linestyle="None",
    zorder=3,
)

ax.errorbar(
    seq_mean,
    y_positions["Sequence-level gate"],
    xerr=seq_sd,
    fmt="o",
    capsize=4,
    elinewidth=1.5,
    markersize=7,
    zorder=3,
)

ax.errorbar(
    temp_mean,
    y_positions["Temporal DWSF"],
    xerr=temp_sd,
    fmt="o",
    capsize=4,
    elinewidth=1.5,
    markersize=7,
    zorder=3,
)


def add_label(x, y, text, dx=8, dy=0):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92),
        clip_on=False,
        zorder=5,
    )


add_label(baseline_wer, y_positions["Fixed baseline"], f"{baseline_wer:.2f}", dx=8, dy=4)
add_label(seq_mean, y_positions["Sequence-level gate"], f"{seq_mean:.2f} +/- {seq_sd:.2f}", dx=8, dy=2)
add_label(temp_mean, y_positions["Temporal DWSF"], f"{temp_mean:.2f} +/- {temp_sd:.2f}", dx=8, dy=-8)

ax.set_yticks([2, 1, 0])
ax.set_yticklabels(["Fixed baseline", "Sequence-level gate", "Temporal DWSF"], fontsize=10)
ax.set_xlabel("Test word error rate (WER, %)", fontsize=11)

all_x = [baseline_wer, seq_mean - seq_sd, seq_mean + seq_sd, temp_mean - temp_sd, temp_mean + temp_sd]
ax.set_xlim(min(all_x) - 0.12, max(all_x) + 0.22)
ax.set_ylim(-0.2, 2.35)
ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout(pad=1.2)

OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"\nSaved PNG: {OUTPUT_PNG}")
print(f"Saved PDF: {OUTPUT_PDF}")
print(f"Saved plot data: {OUTPUT_TABLE}")
