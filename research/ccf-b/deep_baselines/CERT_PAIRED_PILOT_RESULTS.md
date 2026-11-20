# CERT 3-seed paired pilot

Date: 2026-11-20.

Status: exploratory. Seeds `20261201-20261203`. This pilot pairs every method
on the same seed and evaluation protocol, but it is not the confirmatory
20-seed matrix.

## Protocol

- CERT r4.2 train/validation/test boundaries and 27-dimensional user-day
  inputs;
- labels only after scores are produced;
- K = 5, 10, 20 and a seven-day cooldown;
- T0 and T4 use the repository's CERT daily pipeline with 30-day causal
  left-padding and the `global_z` mask-residual score;
- Anomaly Transformer uses its documented 64-day CERT adaptation;
- t-NNCLR uses its documented 96-day CERT adaptation and representation bank.

The T0/T4 comparison is a mask-residual control. It does not replace the
canonical T4+PAI-A1 result.

## Test-set means

| Method | PR-AUC | ROC-AUC | K=5 | K=10 | K=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 matched | 0.007776 | 0.809866 | 0.277778 | 0.472222 | 0.625000 |
| T4 full SoftCLT | 0.008215 | 0.816407 | 0.277778 | 0.472222 | 0.569444 |
| Anomaly Transformer | 0.002324 | 0.497332 | 0.027778 | 0.055556 | 0.097222 |
| t-NNCLR | 0.003709 | 0.643231 | 0.222222 | 0.319444 | 0.416667 |

## Paired differences

T4 minus T0:

| Metric | Mean delta | Positive seeds | 95% t interval |
| --- | ---: | ---: | ---: |
| PR-AUC | +0.000439 | 2/3 | [-0.000559, +0.001437] |
| ROC-AUC | +0.006541 | 2/3 | [-0.026193, +0.039275] |
| K=5 recall | 0.000000 | 2 positive, 1 negative | descriptive only |
| K=10 recall | 0.000000 | 2 positive, 1 negative | descriptive only |
| K=20 recall | -0.055556 | 0/3 | [-0.115315, +0.004204] |

T4 has a small positive PR/ROC seed mean, but the three-seed interval crosses
zero and K=20 does not improve. This does not establish an end-to-end T4 gain
over matched T0.

T4 minus t-NNCLR:

| Metric | Mean delta | Positive seeds | 95% t interval |
| --- | ---: | ---: | ---: |
| PR-AUC | +0.004506 | 3/3 | [-0.001474, +0.010486] |
| ROC-AUC | +0.173175 | 3/3 | [-0.075529, +0.421880] |
| K=5 recall | +0.055556 | 1/3 | [-0.455026, +0.566137] |
| K=10 recall | +0.152778 | 2/3 | [-0.336371, +0.641927] |
| K=20 recall | +0.152778 | 3/3 | [-0.239089, +0.544644] |

All T4-minus-t-NNCLR means are positive, but every interval crosses zero.
Report this as a promising direction, not a confirmed advantage.

T4 minus Anomaly Transformer is positive on all seeds and all five endpoints.
Its intervals exclude zero, but the Anomaly Transformer adaptation remains
far from random ranking and the result is specific to this CERT construction.
It is not evidence about the method's native benchmark.

## Interpretation

The pilot resolves the previous unpaired-baseline problem, but it does not
strengthen the T4 end-to-end claim. The strongest defensible statement is:

- T0/T4 are substantially stronger than the current Anomaly Transformer
  adaptation on CERT;
- T4 is directionally above t-NNCLR on point ranking, with uncertain
  three-seed dispersion;
- T4 versus matched T0 remains unresolved and must retain the negative
  K=20 result.

Artifacts:

- `results/cert_paired_pilot_20261201_03_v1/cert_paired_pilot_by_seed.csv`;
- `results/cert_paired_pilot_20261201_03_v1/cert_paired_pilot_summary.csv`;
- `results/cert_paired_pilot_20261201_03_v1/cert_paired_pilot_summary.json`.
