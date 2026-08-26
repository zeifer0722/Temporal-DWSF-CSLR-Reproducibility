# Checkpoint evidence

This directory records SHA256 hashes for the retained checkpoints used by the final dissertation experiments.

## Important naming note for the e25 reference

The fixed e25 starting checkpoint is physically retained at:

`outputs/Phoenix-2014T_baseline_e10_bs2/best_checkpoint.pth`

The directory name is a legacy experiment-output label and does not describe the final training stage of the file. This mapping is supported by the retained qualitative-export record, where the experiment is explicitly labelled `baseline_e25`, loads the path above, and produces fused-head WER `31.7445409721`, matching the dissertation's reported e25 value of `31.7445`.

The same retained path is used as the `--finetune` starting checkpoint in the final matched sequence-level and Temporal DWSF gate runs.

## Files

- `checkpoint_sha256.txt` provides machine-readable SHA256/path pairs.
- `checkpoint_manifest.csv` maps each retained file to its experimental role and seed.

Development labels such as `v3` and legacy output-directory names are preserved in retained paths so the evidence continues to identify the original experiment artefacts exactly. Dissertation-facing terminology uses the final method name **Temporal DWSF**.
