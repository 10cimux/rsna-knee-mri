# Provenance, attribution and publication boundaries

## Project contribution

Décimus Tavares: project participation, experiment execution and decisions, data and
validation audits, pipeline development and inference engineering with AI-assisted coding.
This release is a curated subset of that work. It claims no authorship of the public
Raptor model or of PyTorch, timm, ConvNeXt, ResNet or CoAtNet.

## Public dependencies used in the competition

| Artifact | Source / declared licence | Reproduction boundary |
|---|---|---|
| Raptor inference code | [Dread Development notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model), Apache-2.0 | Upstream author credited; code/weights are not bundled in this release |
| Raptor MRI weights, dataset v2 | [Knee MRI model weights](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense), CC0 declared by uploader | Declaration does not independently certify every training-data right or competition eligibility |
| Raptor training code | [Public training notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model), Apache-2.0 | Reviewed source; exact training-to-weight chain not independently reproduced |
| CoAtNet backbone | [timm model card](https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k), Apache-2.0 | ImageNet-12k / ImageNet-1k pretraining |
| Residual CoAtNet package (EXP11) | [mattiaangeli package](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3), CC0 declared | Runtime pilot failed; no predictive gain claimed |

Raptor weight SHA-256:
`89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded`.
Kaggle accepting/scoring a notebook does not certify dataset licensing or prize eligibility.
Licence descriptions above reflect the reviewed public artifact versions; they are not a legal opinion.

## What is deliberately excluded

Competition images, reports, actual study identifiers, labels, per-study predictions,
saved feature caches, model weights, API credentials and notebooks embedding those
records. Public access to a competition does not itself grant unrestricted republication
of its data. Follow official access conditions to obtain data separately.

## Licence scope

The MIT licence in this release covers project-authored portfolio code and documentation.
It grants no rights to excluded competition data, model weights or third-party software.
Install dependencies under their own licences. No upstream Apache code is redistributed
in this curated release; if that changes, retain its licence and notices separately.
