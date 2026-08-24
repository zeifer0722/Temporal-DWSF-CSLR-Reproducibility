# Data included in this repository

This directory contains compact, non-dataset experiment tables retained to support the dissertation results.

## `source_csv/`

Compact source tables copied from the retained evidence bundle, including:

- developmental and controlled WER results;
- matched multi-seed raw and summary results;
- paired seed-level comparisons;
- zero-mask WER results;
- Temporal DWSF mean scale statistics;
- retained per-head WER verification for the e25 baseline and seed-0 Temporal DWSF run.

## `derived_tables/`

Final compact tables generated from the retained analysis pipeline, including:

- parameter-efficiency comparison;
- selected q25/median/q75 temporal examples;
- qualitative edit-error summary;
- per-sample outcome counts;
- final qualitative example selection used by the dissertation (Samples 3258, 6005 and 1412).

## Files not redistributed

The public repository does not redistribute:

- RWTH-PHOENIX-Weather 2014T dataset files;
- `gloss2ids.pkl`;
- trained checkpoint binaries;
- the full per-sample prediction exports used by the qualitative analysis;
- the approximately 4 MB frame-level temporal-scale export used to select and plot Figure 4.3.

These larger retained outputs remain in the local evidence bundle. Their aggregate results and deterministic selection outputs are included here so the reported dissertation values remain inspectable.
