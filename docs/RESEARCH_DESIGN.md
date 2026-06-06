# Research Design

## Proposed title

**RealRF-DG: Source-Aware Dual-Domain Learning for Data-Efficient Radio
Fingerprinting Across Real Capture Environments**

## Research question

Can a shared temporal-frequency encoder trained on only 10% of several real
RF corpora learn device-discriminative features without relying on
dataset-specific shortcuts?

## Contribution

1. A checksum-verified, manifest-driven benchmark using only real captures.
2. A dual-domain encoder that fuses raw I/Q temporal features with FFT
   magnitude and phase-step features.
3. Source-specific device heads plus gradient-reversal source confusion.
4. Cross-environment testing on the 2026 WLAN dataset.
5. Reproducible L4 batch/throughput tuning and standard metric exports.

## Primary metrics

- Per-source macro F1 and accuracy.
- Expected calibration error (15 bins).
- Cross-environment WLAN accuracy.
- Throughput, peak GPU memory, mean GPU utilization, and energy proxy
  (power draw integrated over training time).

## Required ablations

- Time branch only.
- Frequency branch only.
- No source-adversarial loss.
- No supervised contrastive loss.
- Shared global classifier instead of source-specific heads.
- Fixed batch size versus L4-tuned batch size.

## Statistical protocol

Run at least three seeds for the final paper. Report mean and standard
deviation. Use paired bootstrap confidence intervals over test predictions.
Do not select the best seed for the headline result.

