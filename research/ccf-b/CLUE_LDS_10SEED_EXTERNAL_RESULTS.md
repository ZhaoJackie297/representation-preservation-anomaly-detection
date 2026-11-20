# CLUE-LDS 10-seed external results

Date: 2026-11-19.

Status: external semi-synthetic replication completed.

CLUE-LDS labels are fixed semi-synthetic account-hijack switches, not official
malicious labels. The external claim is limited accordingly.

## 1. Protocol

```text
users:    838
timeline: 1911 days
train:    0-1697
test:     1698-1910
events:   48
seeds:    20261116-20261125
models:   30
```

To make the long CLUE sequences computationally tractable, all methods use the
same random crop limit of 256 days. Crop start positions still span the full
training history.

The fixed matrix and PAI weights are unchanged.

## 2. Test method summary

| Method | PR-AUC | ROC-AUC | K=10 |
| --- | ---: | ---: | ---: |
| A0-official + plain | 0.655168 | 0.490361 | 46.7/48 |
| A0-official + PAI-A1 | 0.685456 | 0.543051 | 47.0/48 |
| A0-matched + PAI-A1 | 0.685989 | 0.544035 | 47.0/48 |
| A1-Cov + plain | 0.662905 | 0.499563 | 45.5/48 |
| **A1-Cov + PAI-A1** | **0.684023** | **0.540662** | **47.0/48** |

K=10 event recall is nearly saturated on CLUE and cannot distinguish methods.

## 3. Primary tests

| Comparison | Metric | Mean delta | 95% seed CI | Positive seeds | Holm p |
| --- | --- | ---: | ---: | ---: | ---: |
| Joint vs official plain | PR-AUC | +0.028855 | [+0.025842, +0.031868] | 10/10 | 1.79e-8 |
| Joint vs official plain | K=10 event recall | +0.006250 | [-0.007888, +0.020388] | 4/10 | 0.687 |
| Coverage effect under PAI | PR-AUC | -0.001965 | [-0.002609, -0.001321] | 0/10 | 2.11e-4 |
| Coverage effect under PAI | K=10 event recall | 0.000000 | [0, 0] | 0/10 | 1.0 |

## 4. Interpretation

Supported externally:

- PAI scoring improves point-level PR-AUC and ROC-AUC on CLUE.

Not supported externally:

- Coverage-aware soft training does not improve point-level metrics on CLUE.
- Its PR-AUC is lower than matched PAI on all ten seeds.
- No event-level gain is observable because K=10 is saturated.
- At K=5, PAI methods detect fewer events than plain scoring, so PAI also does
  not show a universal event-level benefit.

The fixed joint method beats the official plain baseline on point-level metrics,
but that gain is attributable to PAI scoring, not to coverage-aware training.

## 5. Overall status

The CERT results establish a version-stable CERT method. CLUE does not confirm
the coverage component outside CERT.

The correct paper positioning is therefore:

```text
strong CERT empirical result
+ external point-level PAI validation
+ external negative result for coverage training
```

CLUE must not be described as confirming the full dual-end method.

## 6. Artifacts

```text
<external-root>\clue_dual_end_cap256_10seed_v1
<external-root>\clue_dual_end_cap256_10seed_scores_v1
<external-root>\clue_external_analysis_v1

research/ccf-b/clue_lds_10seed_metrics_by_seed.csv
research/ccf-b/clue_lds_10seed_method_summary.csv
research/ccf-b/clue_lds_10seed_paired_tests.csv
research/ccf-b/clue_lds_10seed_analysis_summary.json
```

