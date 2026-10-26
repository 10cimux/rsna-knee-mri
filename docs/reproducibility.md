# Reproducibility levels

1. **Runnable here:** pairwise AUC decomposition, synthetic shard merging, original
   global rank policy, conservative runtime projection, original EXP08 PyTorch heads
   and tests on synthetic feature tensors. All run without competition access.
2. **Preserved evidence:** aggregate experiment metrics, recorded pilot timing and
   artifact/source hashes. These document past runs; they are not rerun MRI experiments.
3. **Requires separate access:** original five-fold training, DICOM geometry audits,
   pretrained encoder extraction and full offline MRI inference. Data, checkpoints,
   GPU environment and the complete private notebooks are not bundled.

## Data-free demo

Install as described in README. The demo's study names are synthetic and its probabilities
are fabricated examples, not anonymized patient records. AUC comparisons use exact
positive-negative pair counts with half credit for ties. This small-array utility is
quadratic and intended for audit-size examples, not millions of observations.

## Tests

The core suite tests defined/undefined AUC, ties, unknown targets, invalid inputs,
fold/global differences, identity and coverage failures, order-independent shard
completion, global-versus-local ranking and historical runtime floors.
The optional vision suite tests masks, gradients, head shapes, seed initialization,
determinism, state round trips and masked BCE. Synthetic backward/gradient checks
are unit tests; no competition training or hyperparameter search occurs.

Historical GPU runs used their recorded Kaggle environments, not the lighter CPU
demo environment. Bitwise equivalence is a statement about specific tested runs,
not a guarantee across arbitrary hardware or library versions.

## Integrity evidence

`evidence/aggregate_results.json` contains no patient-level records. Source identities
are recorded in `evidence/source_lineage.json`. Generated release-file hashes live
in `release_manifest.json`; do not treat them as hashes of the original trained models.
