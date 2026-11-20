# Official PAI weight comparison

Date: 2026-11-20.

This analysis reuses the locked r4.2 10-seed score components:

- `euc_z`;
- `mag_g_z`;
- `t2_z`.

No model was retrained and no weight was selected on the test split.

## Definitions

Project PAI-A1:

```text
score_project = 1.0 * euc_z + 0.6 * mag_g_z + 0.4 * t2_z
```

Official scalar weights applied to the same local components:

```text
score_official = 0.6 * euc_z + 0.4 * mag_g_z + 0.2 * t2_z
```

The official scorer was executed separately as a code-level reference on the
same three normalized components.

## Code-level reference

- official repository: `pantheon5100/PAI`;
- audited commit: `29cb3bbf8e40b1610be9a3e91f4500c88fd930d3`;
- official formula:
  `0.6*z_euc + 0.4*z_magG + 0.2*z_T2`;
- all six T0/T4 pilot component arrays agree within `9.54e-7`;
- the official loader converts inactive `NaN` values to zero, while the local
  evaluator retains `NaN` and filters with the active mask.

This verifies formula and interface compatibility, not a native TSB-AD
benchmark reproduction. The upstream checkout has no repository-level license
file and remains outside the release.

On the three-seed pilot, the local implementation of the official formula
produced:

| Training | PR-AUC | ROC-AUC | K=10 | K=20 |
| --- | ---: | ---: | ---: | ---: |
| T0 | 0.010459 | 0.858605 | 0.597222 | 0.791667 |
| T4 | 0.013234 | 0.870970 | 0.597222 | 0.791667 |

Project PAI-A1 minus official PAI remained positive on PR-AUC and ROC-AUC on
3/3 seeds for both T0 and T4, while event recall was mixed. These remain
exploratory pilot observations.

## Results

| Training | PAI | PR-AUC | ROC-AUC | K=10 event recall |
| --- | --- | ---: | ---: | ---: |
| A0-official | project | 0.011094 | 0.869042 | 0.5292 |
| A0-official | official weights | 0.010574 | 0.862872 | 0.5500 |
| A0-matched | project | 0.010334 | 0.862044 | 0.5292 |
| A0-matched | official weights | 0.009855 | 0.855612 | 0.5375 |
| A1-Cov | project | 0.012839 | 0.876870 | 0.5792 |
| A1-Cov | official weights | 0.012395 | 0.870761 | 0.5625 |

Project minus official weights:

- PR-AUC: `+0.000520`, `+0.000480`, `+0.000444` for A0-official,
  A0-matched, and A1-Cov respectively; all are positive on 10/10 seeds.
- ROC-AUC: `+0.006170`, `+0.006432`, `+0.006109`; all are positive on
  10/10 seeds.
- Event recall is mixed and changes by less than one event in most K settings.

## Interpretation

The higher project weights are not the sole source of the joint result. They
improve point ranking over the official scalar weights, but they do not create
a stable event-level advantage. The event conclusion therefore remains tied to
the fixed-budget protocol and the small CERT event count.

Artifacts:

- `official_pai_10seed_by_seed.csv`;
- `official_pai_10seed_summary.csv`;
- `official_pai_10seed_paired.csv`;
- `official_pai_10seed_summary.json`.
- `code_level_reference_audit.json`;
- `verify_official_pai_reference.py`.
- `../deep_baselines/results/cert_paired_official_pai_20261201_03_v1/`.
