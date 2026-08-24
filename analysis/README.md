# Analysis scripts

These scripts reproduce or support the principal tables, figures, and diagnostic analyses reported in the dissertation.

## Main result pipeline

| Script | Dissertation role |
|---|---|
| `build_final_results_master.py` | Consolidates verified baseline, matched-seed gate, parameter, WER, and timing results. |
| `build_parameter_efficiency_table.py` | Builds the parameter-efficiency comparison underlying Table 4.2. |
| `plot_controlled_wer_multiseed.py` | Generates the controlled repeated-seed WER comparison used for Figure 4.1. |
| `plot_zero_mask.py` | Generates the stream-removal sensitivity figure used for Figure 4.2. |
| `plot_temporal_examples.py` | Selects deterministic q25/median/q75 temporal-scale examples and generates the three-panel Figure 4.3. |
| `analyse_qualitative_predictions.py` | Performs deterministic Levenshtein error analysis, sample-outcome analysis, qualitative sample selection, and the edit-error comparison underlying Figure 4.4. |
| `export_sample_3258_keyposes.py` | Extracts the six evenly spaced skeleton poses used as visual context in Figure 4.5. |
| `plot_qualitative_alignments.py` | Generates the selected qualitative alignment visualisations used for the supplementary examples. |
| `build_appendix_seed_table.py` | Builds the matched-seed table reported in Appendix A.1. |

## Naming note

Some retained experiment files and source-table identifiers contain the internal development label `v3`. This is kept where needed to locate the exact experiment records. Dissertation-facing labels use the final method name **Temporal DWSF**.

The code variable / data key `body` is likewise retained from the experiment implementation. In the dissertation this 79-keypoint fourth stream is called the **composite stream**.

## Scripts intentionally omitted

The following local scripts were not uploaded because they were superseded by the final dissertation pipeline:

- `old_plot_controlled_wer.py` — earlier single-run controlled WER figure, superseded by the matched multi-seed version.
- `plot_error_type_stacked.py` — earlier stacked error plot, superseded by the final grouped edit-error comparison generated in `analyse_qualitative_predictions.py`.
- `plot_mean_scales.py` — earlier mean-scale bar chart, superseded by the final three-case temporal-scale trajectory analysis.

## Data

The public repository includes the compact source CSVs used by most final analyses where redistribution is appropriate. The original PHOENIX dataset, model checkpoints and the large frame-level temporal-scale export are not redistributed. Scripts retain their original local paths so the exact experiment snapshot remains traceable.
