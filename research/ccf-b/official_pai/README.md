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

The second score is an official-weight sensitivity analysis, not a
byte-identical run of the official PAI repository.

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
