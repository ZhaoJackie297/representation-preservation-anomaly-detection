# Phase 1 lightweight results

Date: 2026-11-19

Only CSV, JSON, and Markdown summaries are stored here. Large `.npz` score
arrays remain under `E:\MarketRisker\work\cert\external`.

## Directories

- `r42_temporal_v1/`: T3_instance_soft vs T4_full_soft, three seeds.
- `r42_coverage_v1/`: A0-official vs A1-Cov on r4.2, three seeds.
- `clue_coverage_v1/`: A0-official vs A1-Cov on CLUE normal behavior, three
  seeds. The fixed eligibility rule realized 54 background users.
- `risk_cases_v1/`: r4.2 10-seed instant/EWMA/CUSUM/propagation and fixed S4
  cases.

Each controlled directory contains:

```text
manifest.json
metrics_by_seed.csv
summary.csv
paired_comparisons.csv
events.csv
```

The injected events are a controlled mechanism benchmark. They are not an
external validation set and must not replace the real CLUE-LDS result.
