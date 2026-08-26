from __future__ import annotations

from pathlib import Path
import csv
import math
import re


ROOT = Path(__file__).resolve().parents[1]

TEXT_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".patch",
    ".md",
    ".txt",
    ".csv",
}

# Covers CJK ideographs, radicals, punctuation, compatibility forms,
# and common CJK presentation forms. This deliberately goes beyond
# Chinese ideographs alone so accidental CJK text cannot enter the
# submission snapshot unnoticed.
CJK_PATTERN = re.compile(
    r"[\u2E80-\u2EFF\u2F00-\u2FDF\u3000-\u303F\u31C0-\u31EF"
    r"\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uFE30-\uFE4F"
    r"\uFF00-\uFFEF]"
)
AWS_ACCESS_KEY_PATTERN = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
AWS_SECRET_PATTERN = re.compile(
    r"(?i)(aws_access_key_id|aws_secret_access_key)\s*[=:]\s*\S+"
)
GITHUB_TOKEN_PATTERN = re.compile(
    r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"
)

RESTRICTED_SUFFIXES = {
    ".pth",
    ".pt",
    ".ckpt",
    ".pkl",
    ".pem",
    ".key",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
}

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
    "data/derived_tables/parameter_efficiency_comparison.csv",
    "data/derived_tables/qualitative_error_summary.csv",
    "data/derived_tables/qualitative_sample_outcomes.csv",
    "data/derived_tables/qualitative_selected_samples.csv",
    "evidence/checkpoint_sha256.txt",
    "evidence/checkpoint_manifest.csv",
    "evidence/README.md",
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

EXPECTED_E25_SHA256 = (
    "7c8d77241f83c401f6993c341671603083bee5d0ae880326a65e593485ea1d62"
)
EXPECTED_E25_RETAINED_PATH = (
    "outputs/Phoenix-2014T_baseline_e10_bs2/best_checkpoint.pth"
)


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def iter_repository_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        yield path


def iter_text_files():
    for path in iter_repository_files():
        if path.name == ".gitignore" or path.suffix.lower() in TEXT_EXTENSIONS:
            yield path


def check_required_files() -> None:
    missing = [relative for relative in REQUIRED_FILES if not (ROOT / relative).exists()]
    if missing:
        fail("Missing required repository files: " + ", ".join(missing))


def check_restricted_files() -> None:
    restricted = []
    for path in iter_repository_files():
        if path.suffix.lower() in RESTRICTED_SUFFIXES:
            restricted.append(str(path.relative_to(ROOT)))

    if restricted:
        fail("Restricted binary/archive files are committed: " + ", ".join(restricted))


def check_text_safety() -> None:
    cjk_hits = []
    secret_hits = []

    for path in iter_text_files():
        relative_path = path.relative_to(ROOT)
        relative_text = str(relative_path)

        if CJK_PATTERN.search(relative_text):
            cjk_hits.append(relative_text)

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            if CJK_PATTERN.search(line):
                cjk_hits.append(f"{relative_path}:{line_number}")

            if (
                AWS_ACCESS_KEY_PATTERN.search(line)
                or PRIVATE_KEY_PATTERN.search(line)
                or AWS_SECRET_PATTERN.search(line)
                or GITHUB_TOKEN_PATTERN.search(line)
            ):
                secret_hits.append(f"{relative_path}:{line_number}")

    if cjk_hits:
        fail("CJK characters found in repository paths or text files: " + ", ".join(cjk_hits))

    if secret_hits:
        fail("Potential credentials or private keys found: " + ", ".join(secret_hits))


def check_python_syntax() -> None:
    syntax_errors = []

    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts:
            continue

        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except (UnicodeDecodeError, SyntaxError) as exc:
            syntax_errors.append(f"{path.relative_to(ROOT)}: {exc}")

    if syntax_errors:
        fail("Python syntax check failed: " + " | ".join(syntax_errors))


def check_portable_code_paths() -> None:
    forbidden_fragments = [
        "/home/ubuntu/",
        "/Users/",
        "C:\\\\Users\\\\",
    ]
    hits = []

    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                hits.append(f"{path.relative_to(ROOT)} -> {fragment}")

    if hits:
        fail("Personal absolute paths found in Python sources: " + ", ".join(hits))


def read_csv(relative_path: str):
    with (ROOT / relative_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(
    actual: float,
    expected: float,
    label: str,
    tolerance: float = 1e-9,
) -> None:
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
    observed = {
        (row["method"], int(row["seed"])): float(row["test_wer"])
        for row in rows
    }

    if set(observed) != set(EXPECTED_SEED_WER):
        fail("The matched multi-seed result keys do not match the submitted experiment set.")

    for key, expected_wer in EXPECTED_SEED_WER.items():
        assert_close(observed[key], expected_wer, f"WER for {key}")

    sequence = [
        value for (method, _), value in observed.items() if method == "Sequence gate"
    ]
    temporal = [
        value
        for (method, _), value in observed.items()
        if method == "Temporal DWSF v3"
    ]

    sequence_mean = sum(sequence) / len(sequence)
    temporal_mean = sum(temporal) / len(temporal)

    assert_close(
        sequence_mean,
        31.509766666666668,
        "Sequence mean WER",
        tolerance=1e-12,
    )
    assert_close(
        temporal_mean,
        31.0558,
        "Temporal mean WER",
        tolerance=1e-12,
    )
    assert_close(
        sequence_mean - temporal_mean,
        0.45396666666666796,
        "Mean temporal advantage",
        tolerance=1e-12,
    )


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
        assert_close(
            float(by_stream[stream]["mean_scale"]),
            expected,
            f"Mean scale for {stream}",
            tolerance=1e-6,
        )


def check_qualitative_tables() -> None:
    outcome_rows = read_csv("data/derived_tables/qualitative_sample_outcomes.csv")
    counts = {
        row["outcome"]: int(row["number_of_samples"])
        for row in outcome_rows
    }

    expected_counts = {"improved": 70, "equal": 529, "worse": 43}
    if counts != expected_counts:
        fail(f"Qualitative outcome counts do not match: {counts}")

    if sum(counts.values()) != 642:
        fail("Qualitative outcome counts do not sum to 642 test samples.")

    error_rows = read_csv("data/derived_tables/qualitative_error_summary.csv")
    by_model = {row["model"]: row for row in error_rows}

    baseline = by_model["Baseline e25"]
    temporal = by_model["Temporal DWSF v3 seed 0"]

    expected_baseline = {
        "substitutions": 626,
        "deletions": 649,
        "insertions": 77,
        "total_errors": 1352,
    }
    expected_temporal = {
        "substitutions": 646,
        "deletions": 586,
        "insertions": 89,
        "total_errors": 1321,
    }

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


def normalise_matched_config(text: str) -> str:
    normalised_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        indentation = line[: len(line) - len(line.lstrip())]

        if stripped.startswith("model_dir:"):
            normalised_lines.append(indentation + "model_dir: <condition-specific>")
        elif stripped.startswith("fusion_level:"):
            normalised_lines.append(indentation + "fusion_level: <condition-specific>")
        else:
            normalised_lines.append(line)

    return "\n".join(normalised_lines)


def check_configs_and_patch() -> None:
    sequence_text = (ROOT / "configs/final/sequence_gate.yaml").read_text(
        encoding="utf-8"
    )
    temporal_text = (ROOT / "configs/final/temporal_dwsf.yaml").read_text(
        encoding="utf-8"
    )
    patch_text = (ROOT / "patches/current_git_diff.patch").read_text(
        encoding="utf-8"
    )

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

    if normalise_matched_config(sequence_text) != normalise_matched_config(temporal_text):
        fail(
            "Sequence-level and Temporal DWSF final configs differ outside "
            "model_dir and fusion_level."
        )

    for fragment in [
        "stream_scales = 4.0 * stream_weights",
        "param.requires_grad = ('stream_gate' in name)",
        "nn.init.zeros_(self.stream_gate[-1].weight)",
    ]:
        if fragment not in patch_text:
            fail(f"Missing implementation fragment in patch: {fragment}")


def check_checkpoint_manifest() -> None:
    rows = read_csv("evidence/checkpoint_manifest.csv")

    if len(rows) != 8:
        fail(f"Expected 8 checkpoint-manifest rows, found {len(rows)}")

    e25_rows = [row for row in rows if row["role"] == "Fixed e25 starting reference"]
    if len(e25_rows) != 1:
        fail("Checkpoint manifest must contain exactly one fixed e25 starting reference.")

    e25 = e25_rows[0]
    if e25["sha256"] != EXPECTED_E25_SHA256:
        fail("Fixed e25 checkpoint SHA256 does not match the retained evidence.")
    if e25["retained_path"] != EXPECTED_E25_RETAINED_PATH:
        fail("Fixed e25 retained path does not match the documented legacy path.")

    sequence_seeds = {
        int(row["seed"])
        for row in rows
        if row["role"] == "Sequence-level gate" and row["seed"]
    }
    temporal_seeds = {
        int(row["seed"])
        for row in rows
        if row["role"] == "Temporal DWSF" and row["seed"]
    }

    if sequence_seeds != {0, 123, 3407}:
        fail(f"Unexpected sequence-gate checkpoint seeds: {sorted(sequence_seeds)}")
    if temporal_seeds != {0, 123, 3407}:
        fail(f"Unexpected Temporal DWSF checkpoint seeds: {sorted(temporal_seeds)}")


def check_public_terminology() -> None:
    for relative_path in ["README.md", "REPRODUCIBILITY.md"]:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        if "Temporal DWSF v3" in text:
            fail(f"Development label appears in dissertation-facing document: {relative_path}")
        if "whole-body" in text.lower():
            fail(f"Deprecated stream terminology appears in: {relative_path}")


def main() -> None:
    check_required_files()
    check_restricted_files()
    check_text_safety()
    check_python_syntax()
    check_portable_code_paths()
    check_main_results()
    check_multiseed_results()
    check_zero_mask_results()
    check_temporal_scale_summary()
    check_qualitative_tables()
    check_configs_and_patch()
    check_checkpoint_manifest()
    check_public_terminology()
    print("Repository sanity check passed.")


if __name__ == "__main__":
    main()
