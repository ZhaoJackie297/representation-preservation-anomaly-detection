# Supplement experiment delivery (2026-11-21)

This directory contains the lightweight supplement-experiment handoff.

## Archives

| File | Size | SHA256 |
| --- | ---: | --- |
| `supplement_task_package_20261121.zip` | 32,955 bytes | `A810328BD8F37ED6F9069D8EB7D31451925CF378E37D323F56594B3581216AAD` |
| `supplement_results_light.zip` | 357,948 bytes | `D8DB07E1A099E5FA197DF2CC41078808AE6BF98CD6FD4B41AD832A29E93DE281` |

`supplement_task_package_20261121.zip` preserves the seven original files
from the desktop handoff directory:

- `AGENT_HANDOFF.md`
- `supplement_experiment_plan.md`
- `exp_s1_soft_random_4th_cell.py`
- `exp_s2_coverage_dose_and_null_graphs.py`
- `exp_s3_cpu_diagnostics_bootstrap_norm.py`
- `exp_s3b_smd_norm_corr.py`
- `exp_s4_daily_ae_and_hyperparams.py`

`supplement_results_light.zip` contains 133 CSV files, 170 JSON files, and
`RUN_LOG.md` (304 files total, 1,950,619 uncompressed bytes). It excludes all
model weights (`.pth`), score arrays (`.npz`), and optimizer checkpoints.

## Interpretation guardrails

- The nominal S2 `dose_A3_50` condition realizes approximately 0.68 A3
  coverage because the sampler also requires every edge-incident user to be
  exposed at least once. Use the values in `coverage_audit.csv`.
- The S2 soft dose-response is not monotonic. The hard 90% condition shows
  only a small increase over the random condition.
- The post-B random-batch Bootstrap comparisons do not replace A1-Cov. Their
  event-level deltas have 95% intervals crossing zero on both r4.2 and r5.2.

See `RUN_LOG.md` inside the results archive for the complete execution record,
path adaptations, warnings, and the interrupted/resumed soft run.
