# SMD 5-seed baseline results

Date: 2026-11-20.

Seeds: `20261201-20261205`.

All values are macro means over the 28 SMD machines. The T0/T4 rows aggregate
the five seed-level macro results.

## Main table

| Method | PR-AUC | ROC-AUC | PA-F1@Top-5 |
| --- | ---: | ---: | ---: |
| Anomaly Transformer-adapted | 0.045713 +/- 0.001236 | 0.521273 +/- 0.005858 | 0.081471 +/- 0.003949 |
| OmniAnomaly-adapted | 0.441056 +/- 0.003970 | 0.813013 +/- 0.001699 | 0.424934 +/- 0.001916 |
| T0-matched + official PAI | 0.453869 +/- 0.002679 | 0.842260 +/- 0.002736 | 0.430910 +/- 0.001979 |
| T0-matched + project PAI | 0.456253 +/- 0.002656 | 0.844038 +/- 0.002730 | 0.431760 +/- 0.002182 |
| T4-full-soft + official PAI | 0.458459 +/- 0.003176 | 0.844654 +/- 0.002648 | 0.434176 +/- 0.002801 |
| T4-full-soft + project PAI | 0.460913 +/- 0.003197 | 0.846348 +/- 0.002631 | 0.435009 +/- 0.002564 |

## Paired comparisons

T4-full-soft + project PAI minus T0-matched + project PAI:

| Metric | Seed mean delta | Positive seeds | Seed t 95% CI | Machine-cluster 95% CI |
| --- | ---: | ---: | ---: | ---: |
| PR-AUC | +0.004661 | 5/5 | [+0.002510, +0.006811] | [+0.001820, +0.007996] |
| ROC-AUC | +0.002310 | 5/5 | [+0.001324, +0.003296] | [+0.000598, +0.004133] |
| PA-F1@Top-5 | +0.003249 | 5/5 | [+0.000571, +0.005928] | [+0.000830, +0.006067] |
| q99 segment recall | -0.007530 | 1/5 | [-0.021017, +0.005957] | [-0.016780, -0.001361] |

The point ranking and top-fraction signal improve, while the q99 segment-recall
tradeoff is negative. This is reported as a metric-specific tradeoff, not
hidden.

T4-full-soft + project PAI minus OmniAnomaly-adapted:

```text
PR-AUC:     +0.019857, 5/5 seeds, machine-cluster CI [-0.017737, +0.056341]
ROC-AUC:    +0.033335, 5/5 seeds, machine-cluster CI [+0.010084, +0.054680]
PA-F1@5:    +0.010075, 5/5 seeds, machine-cluster CI [-0.021565, +0.042465]
```

T4 is higher on all seed means, but only the ROC-AUC machine-cluster interval
excludes zero. The PR and PA-F1 comparisons are not confirmed effects.

T4-full-soft + project PAI minus Anomaly Transformer-adapted:

```text
PR-AUC:     +0.415200, 5/5 seeds, machine-cluster CI [+0.320042, +0.513569]
ROC-AUC:    +0.325075, 5/5 seeds, machine-cluster CI [+0.273803, +0.373203]
PA-F1@5:    +0.353538, 5/5 seeds, machine-cluster CI [+0.269193, +0.443653]
```

Anomaly Transformer uses the official 100-point window, while T0/T4 use 512.
Its poor SMD transfer is therefore an adapted-baseline result, not a claim
about the method's performance on its original benchmark.

## PAI weights

Within the same training representation, project weights beat official scalar
weights on point ranking:

```text
T0 PR:  +0.002383
T0 ROC: +0.001778
T4 PR:  +0.002454
T4 ROC: +0.001694
```

The project and official-weight event results are not available for SMD
because SMD has point labels but no CERT-style event windows. No event claim is
made on SMD.

## Boundary

The supported conclusion is:

- T4-full-soft has a small but consistent point-ranking advantage over matched
  T0 on the new SMD seed range;
- it does not dominate every metric because q99 segment recall declines;
- OmniAnomaly-adapted is close on PR-AUC and is not decisively separated by the
  machine-cluster interval;
- Anomaly Transformer-adapted performs poorly under this SMD adaptation;
- this is not universal SOTA evidence.

Artifacts:

- `results/smd_all_methods_summary.csv`;
- `results/smd_all_methods_by_seed.csv`;
- `results/t4_vs_omni.csv`;
- `results/t4_vs_at.csv`;
- `../smd_multiseed/results/`.
