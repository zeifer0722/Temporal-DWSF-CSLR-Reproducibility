from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "source_csv"
GENERATED_TABLE_DIR = PROJECT_ROOT / "generated" / "tables"

SUMMARY_FILE = SOURCE_DIR / "multiseed_summary.csv"
RAW_FILE = SOURCE_DIR / "multiseed_raw_results.csv"
OUTPUT_CSV = GENERATED_TABLE_DIR / "parameter_efficiency_comparison.csv"
OUTPUT_MARKDOWN = GENERATED_TABLE_DIR / "parameter_efficiency_comparison.md"

BASELINE_E25_WER = 31.7445
BASELINE_E30_WER = 30.6175
BASELINE_TOTAL_PARAMETERS = 44_159_272
GATE_PARAMETERS = 67_904
GATED_MODEL_TOTAL_PARAMETERS = BASELINE_TOTAL_PARAMETERS + GATE_PARAMETERS
BASELINE_E30_TRAINING_TIME = "02:24:47"


def time_to_seconds(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    parts = [int(part) for part in text.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Training time must use H:MM:SS format: {value}")

    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_time(seconds):
    rounded_seconds = int(round(seconds))
    hours = rounded_seconds // 3600
    remainder = rounded_seconds % 3600
    minutes = remainder // 60
    seconds = remainder % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_integer(value):
    return f"{int(value):,}"


if not SUMMARY_FILE.exists():
    raise FileNotFoundError(f"Cannot find multi-seed summary: {SUMMARY_FILE}")
if not RAW_FILE.exists():
    raise FileNotFoundError(f"Cannot find multi-seed raw results: {RAW_FILE}")

summary = pd.read_csv(SUMMARY_FILE)
raw = pd.read_csv(RAW_FILE)

required_summary_columns = {
    "method",
    "n_seeds",
    "mean_test_wer",
    "sample_sd_test_wer",
}
required_raw_columns = {"method", "seed", "test_wer", "training_time"}

missing_summary = required_summary_columns - set(summary.columns)
missing_raw = required_raw_columns - set(raw.columns)

if missing_summary:
    raise ValueError(f"Summary file is missing columns: {sorted(missing_summary)}")
if missing_raw:
    raise ValueError(f"Raw file is missing columns: {sorted(missing_raw)}")

method_order = ["Sequence gate", "Temporal DWSF v3"]
summary = (
    summary[summary["method"].isin(method_order)]
    .set_index("method")
    .reindex(method_order)
    .reset_index()
)

if summary["mean_test_wer"].isna().any():
    missing_methods = summary.loc[
        summary["mean_test_wer"].isna(),
        "method",
    ].tolist()
    raise ValueError(f"Missing multi-seed results for: {missing_methods}")


def method_statistics(method_name):
    summary_row = summary[summary["method"] == method_name].iloc[0]
    method_raw = raw[raw["method"] == method_name].copy()

    if len(method_raw) != 3:
        raise ValueError(
            f"{method_name} should have three seed runs, but {len(method_raw)} were found."
        )

    training_seconds = method_raw["training_time"].apply(time_to_seconds)
    if training_seconds.isna().any():
        raise ValueError(f"Missing training time for {method_name}")

    return {
        "n_seeds": int(summary_row["n_seeds"]),
        "mean_wer": float(summary_row["mean_test_wer"]),
        "sd_wer": float(summary_row["sample_sd_test_wer"]),
        "mean_training_time": seconds_to_time(training_seconds.mean()),
        "minimum_training_time": seconds_to_time(training_seconds.min()),
        "maximum_training_time": seconds_to_time(training_seconds.max()),
    }


sequence = method_statistics("Sequence gate")
temporal = method_statistics("Temporal DWSF v3")

gate_updated_share = GATE_PARAMETERS / GATED_MODEL_TOTAL_PARAMETERS * 100
sequence_improvement = BASELINE_E25_WER - sequence["mean_wer"]
temporal_improvement = BASELINE_E25_WER - temporal["mean_wer"]
baseline_e30_improvement = BASELINE_E25_WER - BASELINE_E30_WER

rows = [
    {
        "method": "Baseline e25 (fixed reference)",
        "evaluation_runs": 1,
        "total_model_parameters": BASELINE_TOTAL_PARAMETERS,
        "additional_gate_parameters": 0,
        "parameters_updated_in_stage": 0,
        "updated_parameter_share_percent": 0.0,
        "additional_epochs": 0,
        "mean_additional_training_time": "-",
        "test_wer": f"{BASELINE_E25_WER:.4f}",
        "wer_improvement_over_e25": "0.0000",
        "comparison_role": "Fixed checkpoint reference; no additional adaptation",
    },
    {
        "method": "Sequence-level gate",
        "evaluation_runs": sequence["n_seeds"],
        "total_model_parameters": GATED_MODEL_TOTAL_PARAMETERS,
        "additional_gate_parameters": GATE_PARAMETERS,
        "parameters_updated_in_stage": GATE_PARAMETERS,
        "updated_parameter_share_percent": gate_updated_share,
        "additional_epochs": 5,
        "mean_additional_training_time": sequence["mean_training_time"],
        "test_wer": f"{sequence['mean_wer']:.4f} +/- {sequence['sd_wer']:.4f}",
        "wer_improvement_over_e25": f"{sequence_improvement:.4f}",
        "comparison_role": "Gate-only adaptation; three matched random seeds",
    },
    {
        "method": "Temporal DWSF",
        "evaluation_runs": temporal["n_seeds"],
        "total_model_parameters": GATED_MODEL_TOTAL_PARAMETERS,
        "additional_gate_parameters": GATE_PARAMETERS,
        "parameters_updated_in_stage": GATE_PARAMETERS,
        "updated_parameter_share_percent": gate_updated_share,
        "additional_epochs": 5,
        "mean_additional_training_time": temporal["mean_training_time"],
        "test_wer": f"{temporal['mean_wer']:.4f} +/- {temporal['sd_wer']:.4f}",
        "wer_improvement_over_e25": f"{temporal_improvement:.4f}",
        "comparison_role": "Temporal gate-only adaptation; three matched random seeds",
    },
    {
        "method": "Baseline e30 (full-model continuation)",
        "evaluation_runs": 1,
        "total_model_parameters": BASELINE_TOTAL_PARAMETERS,
        "additional_gate_parameters": 0,
        "parameters_updated_in_stage": BASELINE_TOTAL_PARAMETERS,
        "updated_parameter_share_percent": 100.0,
        "additional_epochs": 5,
        "mean_additional_training_time": BASELINE_E30_TRAINING_TIME,
        "test_wer": f"{BASELINE_E30_WER:.4f}",
        "wer_improvement_over_e25": f"{baseline_e30_improvement:.4f}",
        "comparison_role": "Full-model continuation; single reference run",
    },
]

table = pd.DataFrame(rows)
GENERATED_TABLE_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(OUTPUT_CSV, index=False)

display_table = table.copy()
for column in [
    "total_model_parameters",
    "additional_gate_parameters",
    "parameters_updated_in_stage",
]:
    display_table[column] = display_table[column].apply(format_integer)

display_table["updated_parameter_share_percent"] = display_table[
    "updated_parameter_share_percent"
].map(lambda value: f"{value:.3f}%")

display_table = display_table.rename(
    columns={
        "method": "Method",
        "evaluation_runs": "Runs",
        "total_model_parameters": "Total parameters",
        "additional_gate_parameters": "Added gate parameters",
        "parameters_updated_in_stage": "Gradient-updated parameters",
        "updated_parameter_share_percent": "Gradient-updated share",
        "additional_epochs": "Additional epochs",
        "mean_additional_training_time": "Mean additional training time",
        "test_wer": "Test WER (%)",
        "wer_improvement_over_e25": "WER reduction vs e25 (pp)",
        "comparison_role": "Experimental role",
    }
)

OUTPUT_MARKDOWN.write_text(display_table.to_markdown(index=False), encoding="utf-8")

print("\nParameter-efficiency comparison")
print("=" * 100)
print(display_table.to_string(index=False))

print("\nTraining-time verification")
print("-" * 100)
print(
    "Sequence-level gate: "
    f"mean={sequence['mean_training_time']}, "
    f"range={sequence['minimum_training_time']}-{sequence['maximum_training_time']}"
)
print(
    "Temporal DWSF: "
    f"mean={temporal['mean_training_time']}, "
    f"range={temporal['minimum_training_time']}-{temporal['maximum_training_time']}"
)
print(
    "\nGate gradient-updated share: "
    f"{gate_updated_share:.6f}% (reported as {gate_updated_share:.3f}%)"
)
print(
    "Temporal mean advantage over sequence gate: "
    f"{sequence['mean_wer'] - temporal['mean_wer']:.4f} WER percentage points"
)
print(
    "Temporal mean improvement over e25: "
    f"{temporal_improvement:.4f} WER percentage points"
)
print(
    "Baseline e30 advantage over Temporal mean: "
    f"{temporal['mean_wer'] - BASELINE_E30_WER:.4f} WER percentage points"
)
print(f"\nSaved CSV: {OUTPUT_CSV}")
print(f"Saved Markdown: {OUTPUT_MARKDOWN}")
