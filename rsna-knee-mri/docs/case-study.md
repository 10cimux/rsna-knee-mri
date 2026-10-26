# Technical case study

## Problem and constraints

Each MRI study produces twelve finding scores. There were 58 fully annotated Gold
studies available to this project, partitioned into five original folds of
12, 10, 12, 11 and 13 studies. Other studies did not become negative examples simply
because an annotation was missing. Competition inference had to run offline within nine hours.

## Why evaluation required engineering

Global OOF AUC compares predictions from different fitted models. An unweighted mean
of fold AUCs only measures within-fold ranking, giving folds equal weight regardless
of how many positive-negative pairs they contain. Global AUC also includes cross-fold
pairs and weights every pair equally. Different class composition and score scales
can therefore create a substantial gap.

The executable synthetic demonstration shows fold AUCs of 1.0 but global AUC of 0.75.
This illustrates a possible mechanism; it does not prove calibration was the sole
cause of the project's real performance gap. We retained original probabilities and
did not recalibrate them retrospectively to improve the reported OOF result.

## Computer vision work

EXP04 used multi-planar 2.5D inputs and physical cropping. EXP08 tested whether a
frozen ConvNeXt encoder plus a small finding-specific attention head improved upon
mean pooling. Both heads shared features, folds and fixed seeds. Masks removed absent
tokens; plane embeddings and relative positions preserved acquisition context.
Unknown targets were excluded from BCE rather than converted into negatives.

The public package retains the original head implementation and its synthetic tests.
It does not include the trained encoder, fitted heads, original labels or MRI images.
The attention hypothesis did not succeed on the observed OOF evaluation. That negative
result was retained rather than replaced by the best retrospective seed.

## From model development to attributed reproduction

Training only on a small Gold subset limited the original pipelines. A public Raptor
checkpoint produced a much stronger accepted public score when reproduced. This was
a change in pretrained model provenance and training supervision, not evidence that
our earlier architectural changes caused the gain. The upstream training workflow
excluded Gold from gradients but used it for selection; its Gold diagnostic must not
be described as independent OOF.

## Exact inference parallelism

EXP12 assigned complete studies to one model replica per GPU, merged raw predictions
in original study order, then applied the original global ranking operation once.
Ranking inside each shard would change predictions. The inherited double-argsort
tie policy is preserved for reproducibility; it is not advocated as a universal
statistical ranking convention.

Equivalence covered input hashes, raw probabilities and ranked outputs in the executed
pilot. The local review independently rechecked parallel repeats; serial raw arrays
were not exported, so serial-to-parallel raw parity relies on the executed pilot report.
The visible-three EXP12 raw predictions were independently identical to EXP10 in float32.

## What I would change in a future project

Start with a versioned experiment ledger, canonical preprocessing entry point and
independent selection/validation protocol. Establish label quality and positive-label
coverage before allocating image-training GPU time. Benchmark realistic DICOM sizes,
startup costs and cold/cache variability early. Reserve a submission buffer instead
of spending the final window on additional experiments.

This project supports a portfolio claim about evaluation discipline and implementation,
not clinical usefulness, a winning placement or an original 0.924 model.
