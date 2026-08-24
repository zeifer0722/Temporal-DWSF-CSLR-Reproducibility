# Reproducibility Notes

This document summarises the retained experimental setup for the MSc dissertation **Temporal Dynamic Weighted Stream Fusion for Skeleton-Based Continuous Sign Language Recognition**.

## 1. Environment

The final recorded experiments were run on:

- AWS EC2 `g4dn.xlarge`;
- region `eu-west-2`;
- NVIDIA Tesla T4 GPU;
- Python 3.13.14;
- PyTorch 2.11.0+cu130 with CUDA acceleration.

Weights & Biases was disabled for the retained final runs. The retained Python version, `pip freeze`, AWS instance note and `nvidia-smi` snapshot are provided under `environment/`.

## 2. Dataset and task

The project uses RWTH-PHOENIX-Weather 2014T and follows the inherited MSKA sign-to-gloss recognition pipeline. The retained partition sizes were:

- train: 7,096 samples;
- development: 519 samples;
- test: 642 samples.

The model consumes pre-extracted keypoints rather than RGB frames. Dataset files and the gloss mapping are not redistributed in this repository.

## 3. Stream configuration

The four retained MSKA streams use the original code/configuration keys:

- `left`: 27 selected keypoints;
- `face`: 26 selected keypoints;
- `right`: 27 selected keypoints;
- `body`: 79 selected keypoints.

The `body` set is exactly the union of the selected left, face and right sets. The dissertation therefore calls this fourth representation the **composite stream**, while the original configuration key is preserved here.

## 4. Temporal DWSF gate

The dissertation-specific gate is inserted after stream encoding and keypoint-dimension pooling, before the main four-stream concatenation.

Gate architecture:

```text
LayerNorm(1024)
→ Linear(1024, 64)
→ ReLU
→ Linear(64, 4, bias=False)
→ Softmax
→ ×4
```

The final linear layer is zero-initialised. This gives initial Softmax proportions of 0.25 per stream and applied scales of 1.0 per stream after multiplication by four.

The gate contains 67,904 parameters.

## 5. Controlled gate comparison

The final sequence-level and temporal-level conditions:

- load the same fixed e25 model weights;
- use the same gate architecture and 67,904 gate parameters;
- use the same identity-preserving initialisation;
- adapt for five epochs;
- use batch size 2;
- use Adam with learning rate 0.001, weight decay 0.001 and betas 0.9/0.998;
- use a cosine-annealing scheduler with `T_max=100`;
- use matched seeds `0`, `123`, `3407`;
- restrict gradient-based optimisation to `stream_gate` parameters.

The principal architectural difference is:

```yaml
fusion_level: sequence
```

versus

```yaml
fusion_level: temporal
```

The sequence-level formulation averages each encoded stream across the temporal dimension before gating. Temporal DWSF keeps the encoded temporal dimension and applies the same shared gate independently at each encoded temporal position.

## 6. Gate-only adaptation vs full-model continuation

For the gate-only conditions, model weights are loaded before fresh optimiser and scheduler construction. Non-gate learnable parameters have `requires_grad=False`.

The separate e30 full-model continuation restores model, optimiser, scheduler and epoch state from the e25 training checkpoint and continues full-model optimisation for five additional epochs.

The e30 condition is therefore a contextual full-training reference, not a parameter-matched comparator.

## 7. Checkpoint selection and evaluation

Checkpoint selection for the final runs uses development WER only. The lowest-development-WER checkpoint within each run is retained and reloaded for final development and test evaluation.

The inherited MSKA evaluator decodes the available recognition outputs and reports the lowest corpus-level WER across them. This behaviour was retained for consistency with the baseline and is explicitly treated as an evaluation limitation in the dissertation.

## 8. Seed handling and determinism

The retained training code sets seeds for:

- PyTorch;
- NumPy;
- Python `random`.

It also disables `cudnn.benchmark`. The implementation does not force every CUDA operation to be deterministic, so the final experiments are described as repeated seeded runs rather than bitwise-deterministic executions.

## 9. Known implementation limitations

The following limitations are retained from the final experimental protocol:

- the sequence-level comparator uses an unmasked mean across the encoded temporal dimension;
- non-gate learnable parameters are excluded from gradient-based optimisation, but the full model remains in training mode, so running statistics such as batch-normalisation state are not explicitly frozen;
- the inherited training loop calls `scheduler.step()` before the optimiser updates at the beginning of each epoch;
- earlier developmental variants had prior exposure to the same test partition before the final design was fixed;
- equivalent per-head prediction exports were not retained for every repeated gate run;
- the public repository does not include dataset files or checkpoint binaries.

## 10. Reported final results

Using the inherited evaluation routine:

| Condition | Reported test WER |
|---|---:|
| Fixed e25 reference | 31.7445% |
| Sequence-level gate, mean ± sample SD | 31.5098% ± 0.1242 |
| Temporal DWSF, mean ± sample SD | 31.0558% ± 0.0358 |
| Full-model continuation e30 | 30.6175% |

The mean Temporal DWSF advantage over the matched sequence-level gate is 0.4540 WER percentage points.

## 11. Analysis pipeline and retained tables

The `analysis/` directory contains scripts for:

- repeated-seed aggregation and Figure 4.1;
- parameter-efficiency summaries;
- Appendix A.1 matched-seed table generation;
- zero-masking diagnostics;
- temporal-scale trajectory selection and plotting;
- deterministic Levenshtein error decomposition;
- qualitative sample selection and alignment plotting;
- Sample 3258 keypose export.

Compact source and derived result tables are provided under `data/`, with runnable-path copies of the core result tables under `final_tables/` and `multiseed_data/`. These cover the numerical results reported in the dissertation, including the matched-seed WER comparison, zero-mask results, parameter-efficiency table, mean temporal-scale statistics, qualitative error counts, sample-outcome counts and final qualitative/temporal example selections.

Three retained artefact classes are intentionally not redistributed: the RWTH-PHOENIX-Weather 2014T dataset, checkpoint binaries, and the larger raw diagnostic exports (the full prediction CSVs and approximately 4 MB frame-level temporal-scale CSV). Consequently, the aggregate/selection analyses are inspectable from this repository, while complete regeneration of the qualitative and temporal-trajectory analyses requires the retained local evidence bundle or equivalent raw exports.
