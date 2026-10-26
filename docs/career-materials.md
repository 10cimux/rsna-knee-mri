# Career materials — factual wording

## CV / LinkedIn project entry

**Knee MRI classification — Data Science & Computer Vision case study**

- Developed a multi-planar 2.5D ResNet18 competition pipeline with an accepted,
  participant-reported public ROC-AUC of 0.732; documented five-fold evaluation on
  58 fully annotated studies and the limitations of a small validation set.
- Audited missing-label handling, DICOM preprocessing, fold identity and global
  versus fold-wise AUC; retained negative experiments and rejected inconclusive gains.
- Reproduced the public Raptor model with attribution (participant-reported public
  score 0.924) and implemented two-GPU study-level inference with exact prediction
  checks in a limited pilot. The optimized version was not submitted before closing.

## Interview walkthrough

1. Explain missing labels and why replacing unknown with zero was wrong.
2. Run the synthetic fold/global AUC example and explain pair weighting.
3. Explain why finding-specific attention was tested, and why its negative result stood.
4. Distinguish original work from public checkpoint adoption.
5. Demonstrate why ranking per shard changes results and why cache-biased timings
   cannot establish a stable hidden-test speedup.
6. Discuss the missed submission buffer and how the next project would be planned.

Use only statements you can explain and support from the code and evidence.
Add your GitHub profile and contact details after confirming them; none are inferred here.
