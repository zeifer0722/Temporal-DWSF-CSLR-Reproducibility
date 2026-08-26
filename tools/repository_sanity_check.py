from __future__ import annotations

from pathlib import Path
import csv
import math
import re
import sys


ROOT = Path(__file__).resolve().parents[1]

TEXT_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".patch",
    ".md",
    ".txt",
    ".csv",
    ".gitignore",
}

CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
AWS_ACCESS_KEY_PATTERN = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
AWS_SECRET_PATTERN = re.compile(
    r"(?i)(aws_access_key_id|aws_secret_access_key)\s*[=:]\s*\S+"
)

REQUIRED_FILES = [
    "README.md",
    "REPRODUCIBILITY.md",
    "NOTICE.md",
    "configs/final/sequence_gate.yaml",
    "configs/final/temporal_dwsf.yaml",
    "configs/final/full_model_continuation.yaml",
    "patches/current_git_diff.patch",
    "data/source_csv/main_results.csv",
    "data/source_csv/multiseed_raw_results.csv",
    "data/source_csv/multiseed_summary.csv",
    "data/source_csv/zero_mask_summary.csv",
    "data/source_csv/temporal_v3_weight_summary.csv",
    "data/derived_tables/qualitative_error_summary.csv",
    "data/derived_tables/qualitative_sample_outcomes.csv",
    "data/derived_tables/qualitative_selected_samples.csv",
]

EXPECTED_MAIN_WER = {
    "Baseline e20": 33.8812,
    "DWSF v1 e20": 35.4309,
    "DWSF v2 e20": 34.5386,
    "Baseline e25": 31.7445,
    "Baseline e30": 30.6175,
}

EXPECTED_SEED_WER = {
    ("Sequence gate", 0): 31.3689,
    ("Sequence gate", 123): 31.6037,
    ("Sequence gate", 3407): 31.5567,
    ("Temporal DWSF v3", 0): 31.0167,
    ("Temporal DWSF v3", 123): 31.0636,
    ("Temporal DWSF v3", 3407): 31.0871,
}


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if path.name == ".gitignore" or path.suffix.lower() in TEXT_EXTENSIONS:
            yield path


def check_required_files() -> None:
    missing = [relative for relative in REQUIRED_FILES if not (ROOT / relative).exists()]
    if missing:
        fail("Missing required repository files: " + ", ".join(missing))


def check_text_safety() -> None:
    cjk_hits = []
    secret_hits = []

    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            if CJK_PATTERN.search(line):
                cjk_hits.append(f"{path.relative_to(ROOT)}:{line_number}")

            if (
                AWS_ACCESS_KEY_PATTERN.search(line)
                or PRIVATE_KEY_PATTERN.search(line)
                or AWS_SECRET_PATTERN.search(line)
            ):
                secret_hits.append(f"{path.relative_to(ROOT)}:{line_number}")

    if cjk_hits:
        fail("CJK characters found in repository text files: " + ", ".join(cjk_hits))

    if secret_hits:
        fail("Potential credentials or private keys found: " + ", ".join(secret_hits))


def read_csv(relative_path: str):
    with (ROOT / relative_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        fail(f"{label}: expected {expected}, found {actual}")


def check_main_results() -> None:
    rows = read_csv("data/source_csv/main_results.csv")
    by_model = {row["model"]: row for row in rows}

    for model, expected_wer in EXPECTED_MAIN_WER.items():
        if model not in by_model:
            fail(f"Missing model in main_results.csv: {model}")
        assert_close(float(by_model[model]["test_wer"]), expected_wer, f"WER for {model}")


def check_multiseed_results() -> None:
    rows = read_csv("data/source_csv/multiseed_raw_results.csv")
    observed = {(row["method"], int(row["seed"])): float(row["test_wer"]) for row in rows}

    if set(observed) != set(EXPECTED_SEED_WER):
        fail("The matched multi-seed result keys do not match the submitted experiment set.")

    for key, expected_wer in EXPECTED_SEED_WER.items():
        assert_close(observed[key], expected_wer, f"WER for {key}")

    sequence = [value for (method, _), value in observed.items() if method == "Sequence gate"]
    temporal = [value for (method, _), value in observed.items() if method == "Temporal DWSF v3"]

    sequence_mean = sum(sequence) / len(sequence)
    temporal_mean = sum(temporal) / len(temporal)

    assert_close(sequence_mean, 31.509766666666668, "Sequence mean WER", tolerance=1e-12)
    assert_close(temporal_mean, 31.0558, "Temporal mean WER", tolerance=1e-12)
    assert_close(sequence_mean - temporal_mean, 0.45396666666666796, "Mean temporal advantage", tolerance=1e-12)


def check_zero_mask_results() -> None:
    rows = read_csv("data/source_csv/zero_mask_summary.csv")
    by_method = {row["method"]: row for row in rows}

    sequence = by_method["Sequence gate from e25"]
    temporal = by_method["Temporal DWSF v3 from e25"]

    for column in ["left", "face", "right", "body"]:
        assert_close(
            float(sequence[column]),
            float(temporal[column]),
            f"Masked WER equality for {column}",
        )


def check_temporal_scale_summary() -> None:
    rows = read_csv("data/source_csv/temporal_v3_weight_summary.csv")
    by_stream = {row["stream"]: row for row in rows}

    expected_scales = {
        "left": 0.969309,
        "face": 1.012529,
        "right": 0.969703,
        "body": 1.048457,
    }

    for stream, expected in expected_scales.items():
        assert_close(float(by_stream[stream]["mean_scale"]), expected, f"Mean scale for {stream}", tolerance=1e-6)


def check_qualitative_tables() -> None:
    outcome_rows = read_csv("data/derived_tables/qualitative_sample_outcomes.csv")
    counts = {row["outcome"]: int(row["number_of_samples"]) for row in outcome_rows}

    expected_counts = {"improved": 70, "equal": 529, "worse": 43}
    if counts != expected_counts:
        fail(f"Qualitative outcome counts do not match: {counts}")

    if sum(counts.values()) != 642:
        fail("Qualitative outcome counts do not sum to 642 test samples.")

    error_rows = read_csv("data/derived_tables/qualitative_error_summary.csv")
    by_model = {row["model"]: row for row in error_rows}

    baseline = by_model["Baseline e25"]
    temporal = by_model["Temporal DWSF v3 seed 0"]

    expected_baseline = {"substitutions": 626, "deletions": 649, "insertions": 77, "total_errors": 1352}
    expected_temporal = {"substitutions": 646, "deletions": 586, "insertions": 89, "total_errors": 1321}

    for key, expected in expected_baseline.items():
        if int(baseline[key]) != expected:
            fail(f"Baseline qualitative count mismatch for {key}")

    for key, expected in expected_temporal.items():
        if int(temporal[key]) != expected:
            fail(f"Temporal qualitative count mismatch for {key}")

    selected_rows = read_csv("data/derived_tables/qualitative_selected_samples.csv")
    selected_ids = {row["name"].rsplit("-", 1)[-1] for row in selected_rows}
    if selected_ids != {"3258", "6005", "1412"}:
        fail(f"Unexpected qualitative sample selection: {sorted(selected_ids)}")


def check_configs_and_patch() -> None:
    sequence_text = (ROOT / "configs/final/sequence_gate.yaml").read_text(encoding="utf-8")
    temporal_text = (ROOT / "configs/final/temporal_dwsf.yaml").read_text(encoding="utf-8")
    patch_text = (ROOT / "patches/current_git_diff.patch").read_text(encoding="utf-8")

    required_sequence_fragments = [
        "dynamic_fusion: true",
        "fusion_level: sequence",
        "fusion_mid_dim: 64",
        "freeze_non_gate: true",
    ]
    required_temporal_fragments = [
        "dynamic_fusion: true",
        "fusion_level: temporal",
        "fusion_mid_dim: 64",
        "freeze_non_gate: true",
    ]

    for fragment in required_sequence_fragments:
        if fragment not in sequence_text:
            fail(f"Missing sequence configuration fragment: {fragment}")

    for fragment in required_temporal_fragments:
        if fragment not in temporal_text:
            fail(f"Missing temporal configuration fragment: {fragment}")

    for fragment in [
        "stream_scales = 4.0 * stream_weights",
        "param.requires_grad = ('stream_gate' in name)",
        "nn.init.zeros_(self.stream_gate[-1].weight)",
    ]:
        if fragment not in patch_text:
            fail(f"Missing implementation fragment in patch: {fragment}")


def main() -> None:
    check_required_files()
    check_text_safety()
    check_main_results()
    check_multiseed_results()
    check_zero_mask_results()
    check_temporal_scale_summary()
    check_qualitative_tables()
    check_configs_and_patch()
    print("Repository sanity check passed.")


if __name__ == "__main__":
    main()
