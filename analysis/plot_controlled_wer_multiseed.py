from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PNG = ROOT / "thesis_figures" / "png" / "figure_controlled_wer_multiseed.png"
OUTPUT_PDF = ROOT / "thesis_figures" / "pdf" / "figure_controlled_wer_multiseed.pdf"
OUTPUT_TABLE = ROOT / "thesis_tables" / "figure_controlled_wer_multiseed_data.csv"


# ============================================================
# 1. Controlled WER data
#    Baseline e25 is used as the fixed reference checkpoint
# ============================================================
baseline_wer = 31.7445

seed_results = {
    "Sequence-level gate": {
        0: 31.3689,
        123: 31.6037,
        3407: 31.5567,
    },
    "Temporal DWSF v3": {
        0: 31.0167,
        123: 31.0636,
        3407: 31.0871,
    },
}


# ============================================================
# 2. Build raw seed-level table
# ============================================================
raw_rows = []
for method, results in seed_results.items():
    for seed, wer in results.items():
        raw_rows.append(
            {
                "method": method,
                "seed": seed,
                "test_wer": wer,
            }
        )

raw_df = pd.DataFrame(raw_rows).sort_values(["method", "seed"]).reset_index(drop=True)

print("\nRaw seed-level values:")
print(raw_df.to_string(index=False))


# ============================================================
# 3. Summary statistics
# ============================================================
summary_rows = [
    {
        "method": "Baseline e25",
        "n_seeds": 1,
        "mean_test_wer": baseline_wer,
        "sample_sd_test_wer": 0.0,
        "minimum_test_wer": baseline_wer,
        "maximum_test_wer": baseline_wer,
        "improvement_over_baseline": 0.0,
    }
]

for method, group in raw_df.groupby("method"):
    values = group["test_wer"].to_numpy(dtype=float)
    mean_wer = float(np.mean(values))
    sd_wer = float(np.std(values, ddof=1))
    min_wer = float(np.min(values))
    max_wer = float(np.max(values))
    improvement = float(baseline_wer - mean_wer)

    summary_rows.append(
        {
            "method": method,
            "n_seeds": len(values),
            "mean_test_wer": mean_wer,
            "sample_sd_test_wer": sd_wer,
            "minimum_test_wer": min_wer,
            "maximum_test_wer": max_wer,
            "improvement_over_baseline": improvement,
        }
    )

summary_df = pd.DataFrame(summary_rows)

print("\nSummary statistics:")
print(summary_df.to_string(index=False))


# ============================================================
# 4. Save summary table
# ============================================================
OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(OUTPUT_TABLE, index=False)


# ============================================================
# 5. Plot
# ============================================================
y_positions = {
    "Baseline e25": 2,
    "Sequence-level gate": 1,
    "Temporal DWSF v3": 0,
}

seq_row = summary_df.loc[summary_df["method"] == "Sequence-level gate"].iloc[0]
temp_row = summary_df.loc[summary_df["method"] == "Temporal DWSF v3"].iloc[0]

seq_mean = float(seq_row["mean_test_wer"])
seq_sd = float(seq_row["sample_sd_test_wer"])

temp_mean = float(temp_row["mean_test_wer"])
temp_sd = float(temp_row["sample_sd_test_wer"])

fig, ax = plt.subplots(figsize=(10.2, 4.8))

ax.plot(
    baseline_wer,
    y_positions["Baseline e25"],
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
    y_positions["Temporal DWSF v3"],
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
        bbox=dict(
            boxstyle="round,pad=0.18",
            fc="white",
            ec="none",
            alpha=0.92,
        ),
        clip_on=False,
        zorder=5,
    )


add_label(baseline_wer, y_positions["Baseline e25"], f"{baseline_wer:.2f}", dx=8, dy=4)
add_label(seq_mean, y_positions["Sequence-level gate"], f"{seq_mean:.2f} ± {seq_sd:.2f}", dx=8, dy=2)
add_label(temp_mean, y_positions["Temporal DWSF v3"], f"{temp_mean:.2f} ± {temp_sd:.2f}", dx=8, dy=-8)

ax.set_yticks([2, 1, 0])
ax.set_yticklabels(
    [
        "Fixed baseline",
        "Sequence-level gate",
        "Temporal DWSF",
    ],
    fontsize=10,
)

ax.set_xlabel("Test word error rate (WER, %)", fontsize=11)

all_x = [
    baseline_wer,
    seq_mean - seq_sd,
    seq_mean + seq_sd,
    temp_mean - temp_sd,
    temp_mean + temp_sd,
]
xmin = min(all_x) - 0.12
xmax = max(all_x) + 0.22
ax.set_xlim(xmin, xmax)
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
