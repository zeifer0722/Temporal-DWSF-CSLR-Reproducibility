# Temporal DWSF for Skeleton-Based Continuous Sign Language Recognition

Public reproducibility materials for the MSc dissertation **“Temporal Dynamic Weighted Stream Fusion for Skeleton-Based Continuous Sign Language Recognition”**, University of Bristol, 2026.

## Repository purpose

This repository provides the dissertation-specific experimental configurations, retained implementation patch, analysis scripts, compact source/derived result tables and environment records used to support the reported results for **Temporal Dynamic Weighted Stream Fusion (Temporal DWSF)**.

The work builds on the public **MSKA: Multi-Stream Keypoint Attention Network for Sign Language Recognition and Translation** implementation:

https://github.com/sutwangyan/MSKA

The complete MSKA source code is **not redistributed here**. Instead, this repository records the dissertation-specific modifications as a retained Git patch together with the final configurations and analysis pipeline.

## Main controlled comparison

The final comparison starts from the same fixed 25-epoch MSKA checkpoint (e25) and adapts only the added stream gate for five epochs under matched seeds `0`, `123` and `3407`.

| Condition | Reported test WER |
|---|---:|
| Fixed e25 reference | 31.74% |
| Sequence-level gate | 31.51% ± 0.12 |
| Temporal DWSF | **31.06% ± 0.04** |
| Full-model continuation to e30 | 30.62% |

Temporal DWSF achieved a mean 0.45 WER-percentage-point advantage over the parameter-matched sequence-level gate. The gate contains 67,904 gradient-updated parameters, approximately 0.154% of the gated model. Full-model continuation retained the best absolute WER and is treated as a contextual reference rather than a parameter-matched comparator.

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── NOTICE.md
├── .gitignore
├── configs/
│   └── final/
│       ├── sequence_gate.yaml
│       ├── temporal_dwsf.yaml
│       └── full_model_continuation.yaml
├── patches/
│   └── current_git_diff.patch
├── analysis/
│   ├── README.md
│   ├── build_final_results_master.py
│   ├── build_parameter_efficiency_table.py
│   ├── build_appendix_seed_table.py
│   ├── plot_controlled_wer_multiseed.py
│   ├── plot_zero_mask.py
│   ├── plot_temporal_examples.py
│   ├── analyse_qualitative_predictions.py
│   ├── export_sample_3258_keyposes.py
│   └── plot_qualitative_alignments.py
├── data/
│   ├── README.md
│   ├── source_csv/
│   └── derived_tables/
├── final_tables/
├── multiseed_data/
│   └── experiment_records/multiseed_gateonly/
├── environment/
│   ├── python_version.txt
│   ├── pip_freeze.txt
│   ├── aws_instance_note.txt
│   └── nvidia_smi.txt
└── evidence/
    └── checkpoint_sha256.txt
```

## Core implementation change

The retained patch adds a lightweight four-stream gate after the MSKA stream encoders. The gate uses:

```text
LayerNorm(4C)
→ Linear(4C, 64)
→ ReLU
→ Linear(64, 4, bias=False)
→ Softmax
→ ×4
```

The final layer is zero-initialised. Equal logits therefore yield Softmax proportions `[0.25, 0.25, 0.25, 0.25]`, which become applied scales `[1, 1, 1, 1]` after multiplication by four. This preserves the original fixed-fusion representation at gate insertion.

The sequence-level comparator averages the encoded temporal dimension before the gate, while Temporal DWSF retains that dimension and applies the same shared gate at every encoded temporal position.

## Naming note

The retained MSKA configuration key for the 79-keypoint fourth stream is `body`. In the dissertation this is described as the **composite stream**, because its selected indices are exactly the union of the left-hand, face and right-hand selected keypoint sets. The original configuration key is retained here for fidelity to the experiment records.

## Included experimental evidence

The repository includes compact source and derived tables for the reported WER, matched-seed comparison, parameter-efficiency analysis, stream-removal analysis, mean temporal-scale statistics, per-head verification for the retained e25/seed-0 checkpoints, final temporal-example selection and final qualitative sample selection. See [`data/README.md`](data/README.md).

## Data and checkpoints not redistributed

The following are not redistributed in this repository:

- RWTH-PHOENIX-Weather 2014T dataset files;
- pre-extracted dataset files and `gloss2ids.pkl`;
- trained `.pth` / `.pt` / `.ckpt` checkpoints;
- AWS credentials, SSH keys or other secrets;
- the full per-sample prediction exports used by the qualitative analysis;
- the approximately 4 MB frame-level temporal-scale export used for Figure 4.3;
- large local experiment-output directories.

Complete rerunning of training therefore requires legitimate access to the dataset and required starting checkpoints. The compact retained tables allow the reported dissertation values and deterministic selections to be inspected without redistributing those larger artefacts.

## Reproducibility notes

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the retained environment, controlled configuration details, training/evaluation behaviour and known implementation limitations.

## Public reproducibility scope

This repository is intended to make the dissertation-specific contribution and analysis pipeline inspectable without republishing the complete upstream MSKA codebase. The retained implementation changes are available in [`patches/current_git_diff.patch`](patches/current_git_diff.patch).
