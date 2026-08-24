from pathlib import Path

import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "multiseed_data"
    / "experiment_records"
    / "multiseed_gateonly"
    / "multiseed_raw_results.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "thesis_tables"
    / "appendix_matched_seed_results.csv"
)

OUTPUT_MARKDOWN = (
    PROJECT_ROOT
    / "thesis_tables"
    / "appendix_matched_seed_results.md"
)


# ============================================================
# 2. Load and validate data
# ============================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find multi-seed raw results: "
        f"{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "method",
    "seed",
    "test_wer",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# 3. Select matched gate results
# ============================================================
SEEDS = [0, 123, 3407]

sequence = (
    df[
        df["method"] == "Sequence gate"
    ][
        ["seed", "test_wer"]
    ]
    .rename(
        columns={
            "test_wer": (
                "sequence_level_wer"
            )
        }
    )
)

temporal = (
    df[
        df["method"]
        == "Temporal DWSF v3"
    ][
        ["seed", "test_wer"]
    ]
    .rename(
        columns={
            "test_wer": (
                "temporal_dwsf_v3_wer"
            )
        }
    )
)


# ============================================================
# 4. Merge matched seeds
# ============================================================
matched = sequence.merge(
    temporal,
    on="seed",
    how="inner",
)

matched = (
    matched[
        matched["seed"].isin(SEEDS)
    ]
    .set_index("seed")
    .reindex(SEEDS)
    .reset_index()
)

if matched.isna().any().any():
    raise ValueError(
        "One or more matched seed results "
        "are missing."
    )

if len(matched) != 3:
    raise ValueError(
        "Expected exactly three matched seeds."
    )


# ============================================================
# 5. Calculate paired temporal advantage
# ============================================================
matched[
    "temporal_advantage_pp"
] = (
    matched["sequence_level_wer"]
    - matched["temporal_dwsf_v3_wer"]
)


# ============================================================
# 6. Verification statistics
# ============================================================
sequence_mean = (
    matched[
        "sequence_level_wer"
    ].mean()
)

temporal_mean = (
    matched[
        "temporal_dwsf_v3_wer"
    ].mean()
)

advantage_mean = (
    matched[
        "temporal_advantage_pp"
    ].mean()
)

sequence_sd = (
    matched[
        "sequence_level_wer"
    ].std(ddof=1)
)

temporal_sd = (
    matched[
        "temporal_dwsf_v3_wer"
    ].std(ddof=1)
)


# ============================================================
# 7. Save machine-readable table
# ============================================================
OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

matched.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# 8. Create thesis-readable table
# ============================================================
display_table = matched.copy()

display_table[
    "sequence_level_wer"
] = display_table[
    "sequence_level_wer"
].map(
    lambda value: f"{value:.4f}"
)

display_table[
    "temporal_dwsf_v3_wer"
] = display_table[
    "temporal_dwsf_v3_wer"
].map(
    lambda value: f"{value:.4f}"
)

display_table[
    "temporal_advantage_pp"
] = display_table[
    "temporal_advantage_pp"
].map(
    lambda value: f"{value:.4f}"
)

display_table = display_table.rename(
    columns={
        "seed": "Seed",
        "sequence_level_wer": (
            "Sequence-level WER (%)"
        ),
        "temporal_dwsf_v3_wer": (
            "Temporal DWSF v3 WER (%)"
        ),
        "temporal_advantage_pp": (
            "Temporal advantage (pp)"
        ),
    }
)

markdown_text = (
    display_table.to_markdown(
        index=False
    )
)

OUTPUT_MARKDOWN.write_text(
    markdown_text,
    encoding="utf-8",
)


# ============================================================
# 9. Print verification
# ============================================================
print(
    "\nAppendix matched-seed results"
)

print("=" * 90)

print(
    display_table.to_string(
        index=False
    )
)

print(
    "\nVerification"
)

print("-" * 90)

print(
    "Sequence mean WER: "
    f"{sequence_mean:.4f}"
)

print(
    "Sequence sample SD: "
    f"{sequence_sd:.4f}"
)

print(
    "Temporal mean WER: "
    f"{temporal_mean:.4f}"
)

print(
    "Temporal sample SD: "
    f"{temporal_sd:.4f}"
)

print(
    "Mean temporal advantage: "
    f"{advantage_mean:.4f} "
    "WER percentage points"
)

print(
    f"\nSaved CSV: "
    f"{OUTPUT_CSV}"
)

print(
    f"Saved Markdown: "
    f"{OUTPUT_MARKDOWN}"
)
