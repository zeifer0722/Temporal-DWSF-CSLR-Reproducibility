from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "source_csv"
GENERATED_TABLE_DIR = PROJECT_ROOT / "generated" / "tables"

INPUT_FILE = SOURCE_DIR / "multiseed_raw_results.csv"
OUTPUT_CSV = GENERATED_TABLE_DIR / "appendix_matched_seed_results.csv"
OUTPUT_MARKDOWN = GENERATED_TABLE_DIR / "appendix_matched_seed_results.md"


if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Cannot find multi-seed raw results: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = {"method", "seed", "test_wer"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

SEEDS = [0, 123, 3407]

sequence = (
    df[df["method"] == "Sequence gate"][["seed", "test_wer"]]
    .rename(columns={"test_wer": "sequence_level_wer"})
)

temporal = (
    df[df["method"] == "Temporal DWSF v3"][["seed", "test_wer"]]
    .rename(columns={"test_wer": "temporal_dwsf_wer"})
)

matched = sequence.merge(temporal, on="seed", how="inner")
matched = (
    matched[matched["seed"].isin(SEEDS)]
    .set_index("seed")
    .reindex(SEEDS)
    .reset_index()
)

if matched.isna().any().any() or len(matched) != 3:
    raise ValueError("Expected exactly three complete matched-seed results.")

matched["temporal_advantage_pp"] = (
    matched["sequence_level_wer"] - matched["temporal_dwsf_wer"]
)

sequence_mean = matched["sequence_level_wer"].mean()
temporal_mean = matched["temporal_dwsf_wer"].mean()
advantage_mean = matched["temporal_advantage_pp"].mean()
sequence_sd = matched["sequence_level_wer"].std(ddof=1)
temporal_sd = matched["temporal_dwsf_wer"].std(ddof=1)

GENERATED_TABLE_DIR.mkdir(parents=True, exist_ok=True)
matched.to_csv(OUTPUT_CSV, index=False)

display_table = matched.copy()
for column in [
    "sequence_level_wer",
    "temporal_dwsf_wer",
    "temporal_advantage_pp",
]:
    display_table[column] = display_table[column].map(lambda value: f"{value:.4f}")

display_table = display_table.rename(
    columns={
        "seed": "Seed",
        "sequence_level_wer": "Sequence-level WER (%)",
        "temporal_dwsf_wer": "Temporal DWSF WER (%)",
        "temporal_advantage_pp": "Temporal advantage (pp)",
    }
)

OUTPUT_MARKDOWN.write_text(
    display_table.to_markdown(index=False),
    encoding="utf-8",
)

print("\nAppendix matched-seed results")
print("=" * 90)
print(display_table.to_string(index=False))

print("\nVerification")
print("-" * 90)
print(f"Sequence mean WER: {sequence_mean:.4f}")
print(f"Sequence sample SD: {sequence_sd:.4f}")
print(f"Temporal mean WER: {temporal_mean:.4f}")
print(f"Temporal sample SD: {temporal_sd:.4f}")
print(f"Mean temporal advantage: {advantage_mean:.4f} WER percentage points")
print(f"\nSaved CSV: {OUTPUT_CSV}")
print(f"Saved Markdown: {OUTPUT_MARKDOWN}")
