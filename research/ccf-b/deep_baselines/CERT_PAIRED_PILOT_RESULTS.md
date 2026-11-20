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

## Reused-model dual-end scores

The saved T0/T4 weights were reused to compute a training-period Euclidean
representation bank and `PAI-A1 = euc_z + 0.6 * magG_z + 0.4 * T2_z`.
No model was retrained for this comparison.

| Method | PR-AUC | ROC-AUC | K=5 | K=10 | K=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 + Euclidean bank | 0.011702 | 0.873614 | 0.333333 | 0.430556 | 0.708333 |
| T4 + Euclidean bank | 0.014292 | 0.890928 | 0.416667 | 0.486111 | 0.625000 |
| T0 + PAI-A1 | 0.010963 | 0.865138 | 0.361111 | 0.569444 | 0.777778 |
| T4 + PAI-A1 | 0.013672 | 0.877248 | 0.375000 | 0.597222 | 0.805556 |

T4+PAI-A1 minus T0+PAI-A1:

- PR-AUC: `+0.002709`, 3/3 positive, 95% t interval
  `[+0.000796, +0.004621]`;
- ROC-AUC: `+0.012110`, 3/3 positive, interval
  `[-0.002099, +0.026319]`;
- K=10 recall: `+0.027778`, one positive seed and two ties;
- K=20 recall: `+0.027778`, one positive seed and one negative seed.

T4+PAI-A1 minus t-NNCLR:

- PR-AUC: `+0.009963`, interval `[+0.007223, +0.012702]`;
- ROC-AUC: `+0.234017`, interval `[+0.076675, +0.391360]`;
- K=20 recall: `+0.388889`, interval `[+0.056164, +0.721613]`;
- K=10 recall: `+0.277778`, but the three-seed interval crosses zero.

The point-ranking advantage over t-NNCLR is therefore clearer than the
mask-only comparison. Event-level results remain less stable. PAI-A1 improves
K=10/K=20 recall over the Euclidean score but lowers PR-AUC, so the two
evaluation dimensions still conflict.

The official PAI weighting (`0.6:0.4:0.2`) was independently executed through
the upstream loader. Its T4 pilot values were PR-AUC `0.013234`, ROC-AUC
`0.870970`, K=10 `0.597222`, and K=20 `0.791667`. Formula equivalence is
recorded in `../official_pai/code_level_reference_audit.json`.

LAN remains unpaired: no executable implementation was located and its
activity-level input cannot be collapsed to the 27-dimensional daily table
without changing the method. The boundary is logged in
`LAN_REPRODUCIBILITY_AUDIT.md`.

## Interpretation

The pilot resolves the previous unpaired-baseline problem, but it does not
strengthen the T4 end-to-end claim. The strongest defensible statement is:

- T0/T4 are substantially stronger than the current Anomaly Transformer
  adaptation on CERT;
- T4's complete Euclidean/PAI scores are consistently above t-NNCLR on
  PR-AUC and ROC-AUC, with event-level uncertainty;
- T4+PAI-A1 improves PR-AUC over T0+PAI-A1 but does not improve ROC/K=20
  robustly;
- the mask-only T4 versus matched T0 comparison remains unresolved and must
  retain the negative K=20 result.

Artifacts:

- `results/cert_paired_pilot_20261201_03_v1/cert_paired_pilot_by_seed.csv`;
- `results/cert_paired_pilot_20261201_03_v1/cert_paired_pilot_summary.csv`;
- `results/cert_paired_pilot_20261201_03_v1/cert_paired_pilot_summary.json`.
- `results/cert_paired_dual_end_20261201_03_v1/`;
- `results/cert_paired_euclidean_20261201_03_v1/`.
- `results/cert_paired_official_pai_20261201_03_v1/`.
