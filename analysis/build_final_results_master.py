from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "source_csv"
GENERATED_TABLE_DIR = PROJECT_ROOT / "generated" / "tables"

MAIN_RESULTS_FILE = SOURCE_DIR / "main_results.csv"
MULTISEED_SUMMARY_FILE = SOURCE_DIR / "multiseed_summary.csv"
MULTISEED_RAW_FILE = SOURCE_DIR / "multiseed_raw_results.csv"

OUTPUT_MASTER = GENERATED_TABLE_DIR / "final_results_master.csv"
OUTPUT_SEED_LEVEL = GENERATED_TABLE_DIR / "final_results_seed_level.csv"

BASELINE_TOTAL_PARAMETERS = 44_159_272
GATE_PARAMETERS = 67_904
GATED_TOTAL_PARAMETERS = BASELINE_TOTAL_PARAMETERS + GATE_PARAMETERS

SEQUENCE_MEAN_TIME = "00:39:51"
TEMPORAL_MEAN_TIME = "00:39:37"
BASELINE_E30_EXTRA_TIME = "02:24:47"


for file_path in [MAIN_RESULTS_FILE, MULTISEED_SUMMARY_FILE, MULTISEED_RAW_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find required source file: {file_path}")

main_results = pd.read_csv(MAIN_RESULTS_FILE)
multiseed_summary = pd.read_csv(MULTISEED_SUMMARY_FILE)
multiseed_raw = pd.read_csv(MULTISEED_RAW_FILE)


def get_main_result(model_name):
    row = main_results[main_results["model"] == model_name]
    if len(row) != 1:
        raise ValueError(
            f"Expected one row for {model_name}, but found {len(row)}."
        )
    return row.iloc[0]


def get_multiseed_result(method_name):
    row = multiseed_summary[multiseed_summary["method"] == method_name]
    if len(row) != 1:
        raise ValueError(
            f"Expected one summary row for {method_name}, but found {len(row)}."
        )
    return row.iloc[0]


baseline_e20 = get_main_result("Baseline e20")
dwsf_v1 = get_main_result("DWSF v1 e20")
dwsf_v2 = get_main_result("DWSF v2 e20")
baseline_e25 = get_main_result("Baseline e25")
baseline_e30 = get_main_result("Baseline e30")
sequence_summary = get_multiseed_result("Sequence gate")
temporal_summary = get_multiseed_result("Temporal DWSF v3")

rows = [
    {
        "experiment_id": "baseline_e20",
        "method": "Fixed-fusion baseline",
        "training_stage": "Epoch 20",
        "initialisation": "From scratch",
        "adaptation_mode": "Full-model training",
        "fusion_level": "Fixed",
        "seed_count": 1,
        "seeds": "0",
        "test_wer_mean": float(baseline_e20["test_wer"]),
        "test_wer_sample_sd": "",
        "test_loss": float(baseline_e20["test_loss"]),
        "total_parameters": BASELINE_TOTAL_PARAMETERS,
        "gradient_updated_parameters": BASELINE_TOTAL_PARAMETERS,
        "gradient_updated_share_percent": 100.0,
        "additional_epochs_from_e25": "",
        "mean_additional_training_time": "",
        "main_result_role": "Development-stage fixed-fusion reference",
        "include_in_main_results": "Yes",
        "source_file": "data/source_csv/main_results.csv",
        "notes": "Single developmental run trained before the final matched gate-only comparison.",
    },
    {
        "experiment_id": "dwsf_v1_e20",
        "method": "DWSF v1",
        "training_stage": "Epoch 20",
        "initialisation": "From scratch",
        "adaptation_mode": "Full-model training",
        "fusion_level": "Sequence-level [B,4]",
        "seed_count": 1,
        "seeds": "0",
        "test_wer_mean": float(dwsf_v1["test_wer"]),
        "test_wer_sample_sd": "",
        "test_loss": float(dwsf_v1["test_loss"]),
        "total_parameters": GATED_TOTAL_PARAMETERS,
        "gradient_updated_parameters": GATED_TOTAL_PARAMETERS,
        "gradient_updated_share_percent": 100.0,
        "additional_epochs_from_e25": "",
        "mean_additional_training_time": "",
        "main_result_role": "Developmental variant: sequence-level direct softmax weighting",
        "include_in_main_results": "Yes",
        "source_file": "data/source_csv/main_results.csv",
        "notes": "Single developmental run preceding the final matched gate-only comparison.",
    },
    {
        "experiment_id": "dwsf_v2_e20",
        "method": "DWSF v2",
        "training_stage": "Epoch 20",
        "initialisation": "From scratch",
        "adaptation_mode": "Full-model training",
        "fusion_level": "Sequence-level [B,4]",
        "seed_count": 1,
        "seeds": "0",
        "test_wer_mean": float(dwsf_v2["test_wer"]),
        "test_wer_sample_sd": "",
        "test_loss": float(dwsf_v2["test_loss"]),
        "total_parameters": GATED_TOTAL_PARAMETERS,
        "gradient_updated_parameters": GATED_TOTAL_PARAMETERS,
        "gradient_updated_share_percent": 100.0,
        "additional_epochs_from_e25": "",
        "mean_additional_training_time": "",
        "main_result_role": "Developmental variant: identity-preserving sequence-level scaling",
        "include_in_main_results": "Yes",
        "source_file": "data/source_csv/main_results.csv",
        "notes": "Improved over v1 in the developmental run but remained above the e20 fixed-fusion reference.",
    },
    {
        "experiment_id": "baseline_e25",
        "method": "Fixed-fusion baseline",
        "training_stage": "Epoch 25",
        "initialisation": "Continued from earlier baseline",
        "adaptation_mode": "Fixed checkpoint reference; no additional adaptation",
        "fusion_level": "Fixed",
        "seed_count": 1,
        "seeds": "0",
        "test_wer_mean": float(baseline_e25["test_wer"]),
        "test_wer_sample_sd": "",
        "test_loss": float(baseline_e25["test_loss"]),
        "total_parameters": BASELINE_TOTAL_PARAMETERS,
        "gradient_updated_parameters": 0,
        "gradient_updated_share_percent": 0.0,
        "additional_epochs_from_e25": 0,
        "mean_additional_training_time": "-",
        "main_result_role": "Fixed checkpoint reference for controlled gate adaptation",
        "include_in_main_results": "Yes",
        "source_file": "data/source_csv/main_results.csv",
        "notes": "No repeated-run error bar because e25 is one fixed checkpoint reference.",
    },
    {
        "experiment_id": "sequence_gate_multiseed",
        "method": "Sequence-level gate",
        "training_stage": "Gate-only adaptation from e25",
        "initialisation": "Same fixed e25 checkpoint",
        "adaptation_mode": "Gate-only; gradient-based optimisation restricted to the gate",
        "fusion_level": "[B,4]",
        "seed_count": int(sequence_summary["n_seeds"]),
        "seeds": "0;123;3407",
        "test_wer_mean": float(sequence_summary["mean_test_wer"]),
        "test_wer_sample_sd": float(sequence_summary["sample_sd_test_wer"]),
        "test_loss": "",
        "total_parameters": GATED_TOTAL_PARAMETERS,
        "gradient_updated_parameters": GATE_PARAMETERS,
        "gradient_updated_share_percent": GATE_PARAMETERS / GATED_TOTAL_PARAMETERS * 100,
        "additional_epochs_from_e25": 5,
        "mean_additional_training_time": SEQUENCE_MEAN_TIME,
        "main_result_role": "Matched sequence-level comparator",
        "include_in_main_results": "Yes",
        "source_file": "data/source_csv/multiseed_summary.csv",
        "notes": "Three matched random seeds. Non-gate learnable parameters were excluded from gradient-based optimisation.",
    },
    {
        "experiment_id": "temporal_dwsf_multiseed",
        "method": "Temporal DWSF",
        "training_stage": "Gate-only adaptation from e25",
        "initialisation": "Same fixed e25 checkpoint",
        "adaptation_mode": "Gate-only; gradient-based optimisation restricted to the gate",
        "fusion_level": "[B,T_enc,4]",
        "seed_count": int(temporal_summary["n_seeds"]),
        "seeds": "0;123;3407",
        "test_wer_mean": float(temporal_summary["mean_test_wer"]),
        "test_wer_sample_sd": float(temporal_summary["sample_sd_test_wer"]),
        "test_loss": "",
        "total_parameters": GATED_TOTAL_PARAMETERS,
        "gradient_updated_parameters": GATE_PARAMETERS,
        "gradient_updated_share_percent": GATE_PARAMETERS / GATED_TOTAL_PARAMETERS * 100,
        "additional_epochs_from_e25": 5,
        "mean_additional_training_time": TEMPORAL_MEAN_TIME,
        "main_result_role": "Proposed encoded-temporal adaptation method",
        "include_in_main_results": "Yes",
        "source_file": "data/source_csv/multiseed_summary.csv",
        "notes": "Three matched random seeds. Non-gate learnable parameters were excluded from gradient-based optimisation.",
    },
    {
        "experiment_id": "baseline_e30",
        "method": "Fixed-fusion baseline",
        "training_stage": "Epoch 30",
        "initialisation": "Full-model continuation from e25",
        "adaptation_mode": "Full-model continuation",
        "fusion_level": "Fixed",
        "seed_count": 1,
        "seeds": "0",
        "test_wer_mean": float(baseline_e30["test_wer"]),
        "test_wer_sample_sd": "",
        "test_loss": float(baseline_e30["test_loss"]),
        "total_parameters": BASELINE_TOTAL_PARAMETERS,
        "gradient_updated_parameters": BASELINE_TOTAL_PARAMETERS,
        "gradient_updated_share_percent": 100.0,
        "additional_epochs_from_e25": 5,
        "mean_additional_training_time": BASELINE_E30_EXTRA_TIME,
        "main_result_role": "Full-model continuation reference; best absolute WER",
        "include_in_main_results": "Yes",
        "source_file": "data/source_csv/main_results.csv",
        "notes": "Single continuation run. Not parameter-matched to the gate-only adaptation conditions.",
    },
]

master = pd.DataFrame(rows)
baseline_e25_wer = float(baseline_e25["test_wer"])
master["wer_improvement_over_e25"] = baseline_e25_wer - master["test_wer_mean"]
master.loc[
    master["experiment_id"] == "baseline_e25",
    "wer_improvement_over_e25",
] = 0.0

GENERATED_TABLE_DIR.mkdir(parents=True, exist_ok=True)
master.to_csv(OUTPUT_MASTER, index=False)

seed_level = multiseed_raw[
    ["method", "seed", "test_wer", "test_loss", "training_time", "log_file"]
].copy()
seed_level["initial_checkpoint"] = "Fixed e25 baseline checkpoint"
seed_level["additional_epochs"] = 5
seed_level.to_csv(OUTPUT_SEED_LEVEL, index=False)

print("\nFinal results master")
print("=" * 100)

display_columns = [
    "experiment_id",
    "method",
    "seed_count",
    "test_wer_mean",
    "test_wer_sample_sd",
    "gradient_updated_parameters",
    "gradient_updated_share_percent",
    "wer_improvement_over_e25",
    "main_result_role",
]
print(master[display_columns].to_string(index=False))

sequence_wer = float(sequence_summary["mean_test_wer"])
temporal_wer = float(temporal_summary["mean_test_wer"])
baseline_e30_wer = float(baseline_e30["test_wer"])
gate_share = GATE_PARAMETERS / GATED_TOTAL_PARAMETERS * 100

print("\nKey verified conclusions")
print("-" * 100)
print(f"Temporal mean advantage over sequence gate: {sequence_wer - temporal_wer:.4f}")
print(f"Temporal mean improvement over e25: {baseline_e25_wer - temporal_wer:.4f}")
print(f"Baseline e30 advantage over Temporal mean: {temporal_wer - baseline_e30_wer:.4f}")
print(f"Gate gradient-updated share: {gate_share:.6f}% (reported as {gate_share:.3f}%)")
print(f"\nSaved master: {OUTPUT_MASTER}")
print(f"Saved seed-level results: {OUTPUT_SEED_LEVEL}")
