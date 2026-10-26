# Experiment ledger

Numbers from preserved project artifacts; OOF, holdout, public score and projections
are distinct evidence types. No metrics were recomputed or tuned for this release.

| Stage | Approach / evidence | Decision and lesson |
|---|---|---|
| Early prototypes | Small 3D CNNs; label and data-loading audits | Unknown labels had been treated as negatives; filename ordering and apparent accuracy were misleading. Correct the data contract first. |
| EXP02 | Five-fold 2.5D; stored global OOF about 0.58897 | Mean fold AUC about 0.67550 was not the global OOF AUC. |
| EXP03 | Multi-plane ResNet18; historical global OOF about 0.60898 | Historical versions differ; do not substitute an earlier reported 0.60606 without version evidence. |
| EXP04 | Three planes, three positions, 256 px, 140 mm crop | Audited OOF 0.6329894097; public 0.732 reported accepted. Historical 0.65927 refers to a different reported state. Preserve the executable selector overrides and hashes. |
| EXP05 | Frozen ResNet18 features + logistic regression | Gold-only OOF 0.6271998388; feature and classifier replay validated. |
| EXP06 | Fixed equal-weight EXP04/05 blend | OOF 0.6483034724; delta +0.015314; existing paired 95% interval [-0.0166556, 0.0462102]. Inconclusive. Latest representative conservative runtime about 19.125 h; not an operational improvement. |
| EXP07 | Report-derived weak supervision | Rejected: insufficient reliable positive labels across folds. Masking worked; supervision quality did not justify MRI training. |
| EXP08 | Frozen ConvNeXt-Tiny, shared cached features, mean vs finding-specific attention; seeds 42/1729/2026 | Mean OOF 0.5929395064; attention 0.6002009235. Large training/validation gap and between-fold ranking loss. Closed without retrospective calibration or seed selection. |
| EXP09 | Fixed alternative slice sampling with frozen EXP04 | OOF 0.6264457546; lower than EXP04. Runtime passed but predictive criterion failed. |
| EXP04 diagnostic | Ranking-pair decomposition and displayed-slice audit | No confirmed structural clinical defect; incomplete spatial evidence limits independent geometry claims. Clinical diagnosis was not inferred automatically. |
| EXP10 | Public Raptor CoAtNet reproduction | Public 0.924 reported accepted. Gold diagnostic 0.912694 is not independent OOF: upstream Gold was used for model selection. Credit the checkpoint author. |
| EXP11 | Fixed Raptor / residual-model diversity pilot | Runtime margin failed before predictive evaluation. No EXP11 AUC established. |
| EXP12 | Same EXP10 recipe, study-level execution on two GPUs | 48 pilot cases across 12 unique studies; executed pilot reports exact parity. Visible-three float32 predictions independently match EXP10. Missed official submission deadline. |

The EXP12 aggregate pilot wall times were 138.017 s serial and 52.968 s parallel.
Cache/order effects make their 2.606 ratio unsuitable as a promised production speedup.
The final conservative projection was 4.237 h for an assumed 1,300 studies;
the visible-three run actually took 163.195 s, including substantial startup.
Neither establishes hidden-test completion or an efficiency-track placing.
