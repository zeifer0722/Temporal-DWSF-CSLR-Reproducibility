# Data included in this repository

This directory contains compact, non-dataset experiment tables retained to support the dissertation results.

## `source_csv/`

Compact source tables copied from the retained evidence bundle include:

- developmental and controlled WER results;
- matched multi-seed raw and summary results;
- paired seed-level comparisons;
- zero-mask WER results;
- Temporal DWSF mean scale statistics;
- retained per-head WER verification for the e25 baseline and seed-0 Temporal DWSF run.

## `derived_tables/`

Committed derived tables include:

- parameter-efficiency comparison;
- selected q25/median/q75 temporal examples;
- qualitative edit-error summary;
- per-sample outcome counts;
- final qualitative example selection used by the dissertation: Samples 3258, 6005, and 1412.

## Optional raw diagnostics

Three larger retained outputs are needed to regenerate the corresponding diagnostic analyses from their earliest retained source:

```text
data/raw_diagnostics/temporal_weights_frame_level.csv
data/raw_diagnostics/baseline_e25_test_predictions_all_heads.csv
data/raw_diagnostics/temporal_v3_seed0_test_predictions_all_heads.csv
```

`data/raw_diagnostics/` is ignored in the submitted public repository. Local copies can therefore be placed there without accidental publication.

## Files not redistributed

The public repository does not redistribute:

- RWTH-PHOENIX-Weather 2014T dataset files;
- `gloss2ids.pkl`;
- trained checkpoint binaries;
- AWS credentials, SSH keys, or other secrets.

The compact source and derived tables remain sufficient to inspect the numerical values reported in the dissertation. Complete model rerunning additionally requires legitimate access to the dataset, the required starting checkpoints, and the upstream MSKA implementation.
