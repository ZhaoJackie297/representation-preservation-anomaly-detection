# Supplement experiment run log

Start date: 2026-11-21 (Asia/Shanghai).

## Frozen execution rules

- Existing `results/canonical/`, model weights, and prepared splits were not
  modified.
- New scripts were copied from the handoff package to
  `research/ccf-b/supplement/`.
- New artifacts are written only below
  `research/ccf-b/supplement_results/`.
- Execution order is A -> B -> C -> D.

## Environment

- Python: `C:\Users\Lenovo\.conda\envs\ts2vec\python.exe`
- Python version: 3.10.22
- PyTorch: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB
- SciPy: 1.15.3
- scikit-learn: 1.7.2
- pandas: 2.3.3

## Path preflight

- CERT r4.2 data: linked at
  `runtime/cert/processed_r42/r4.2_sequence_dataset.npz`.
- CERT r5.2 data: linked at
  `runtime/cert/processed_r52/r5.2_sequence_dataset.npz`.
- r4.2 10-seed score root:
  `E:\MarketRisker\work\cert\external\pilot_r42_10seed_scores_v1`.
- r5.2 10-seed score root:
  `E:\MarketRisker\work\cert\external\r52_10seed_scores_v1`.
- Score archives contain the expected keys:
  `plain_l1`, `pai_a1`, `euc_z`, `mag_g_z`, `t2_z`.
- Seed-20261116 model checkpoints for T0, T3, and T4 were found under the
  external experiment root.

## Execution record

Commands, errors, parameter adjustments, and completion status are appended
below as each task finishes.

## A. Zero-retrain diagnostics

Completed:

1. `random-r42`
2. `amp-standalone --dataset r4.2`
3. `amp-standalone --dataset r5.2`
4. `norm-corr --dataset r4.2`
5. Four paired bootstrap runs (official-plain and matched-PAI comparators on
   both r4.2 and r5.2).

Local path adaptation for bootstrap:

- The score archive stores `A0-matched_scores.npz`,
  `A0-official_scores.npz`, and `A1-Cov_scores.npz` together below each
  `seed_2026xxxx` directory.
- The original loader keyed only by seed directory and therefore overwrote
  methods within each seed. It was changed to require exact
  `--file-a`/`--file-b` filenames and to reject duplicate seed tags.
- All four completed reports contain `n_common_seeds=10` and
  `n_valid_resamples=1000`.
- `np.nanmean` emits "Mean of empty slice" for inactive user-day cells that
  are NaN in every seed. The downstream PR/ROC calculations and reported
  resamples are restricted to `active_mask`; no implicit missing-value
  imputation is used.

Norm-correlation correction:

- Corrected the handoff script's `casual=True` spelling to the model API's
  `causal=True`.

## B. Soft-instance + random-batch 4th cell

Completed both locked datasets:

- r4.2: `SoftInst-Rand-10ep`, `SoftInst-Rand-Matched`, and
  `Hard-Rand-Matched`; 30/30 run directories complete.
- r5.2: `SoftInst-Rand-10ep` and `SoftInst-Rand-Matched`; 20/20 run
  directories complete.
- Every run has `metrics.csv`, `hit_ids_k10.json`, and `run_meta.json`.
- Every method/split/K group contains exactly 10 unique seeds.
- r4.2 matched steps were read per seed from
  `research/ccf-b/r4.2_10seed_coverage_summary.csv` and ranged from 1130 to
  1158.
- r5.2 matched steps were read per seed from the existing external matrix
  summary instead of the script's fixed fallback:
  `E:\MarketRisker\work\cert\external\r52_10seed_matrix_v1\summary.json`.
  The actual A1-Cov optimizer steps ranged from 2631 to 2654.
- No run had a missing final loss.

Post-B paired Bootstrap:

- Added the required
  `SoftInst-Rand-Matched + PAI-A1` vs `A1-Cov + PAI-A1` comparison on both
  datasets.
- Both reports contain 10 seeds and 1000 valid user-cluster resamples.
- r4.2: 24 events; event delta `-0.020833` (95% CI
  `[-0.100000, 0.054271]`) and PR-AUC delta `+0.000315` (95% CI
  `[-0.001089, 0.003691]`).
- r5.2: 99 events; event delta `+0.002020` (95% CI
  `[-0.029293, 0.032323]`) and PR-AUC delta `+0.000043` (95% CI
  `[-0.000335, 0.000388]`).

## C. Coverage dose-response and isolated null graphs

Completed:

- soft objective: 30/30 run directories and 30/30 coverage records.
- hard objective: 30/30 run directories and 30/30 coverage records.
- Both objectives use exactly 700 optimizer steps in every arm and seed.
- Both use the same five seeds (`20261116`-`20261120`) and the same batch
  tables for matching conditions.

Command-line correction:

- The `--total-steps` help string contained an unescaped `90%`, causing
  argparse to interpret it as `%c`. It was escaped for display only; no model
  or evaluation behavior changed.

Coverage audit limitation:

- The handoff sampler's gate requires every user incident to a target edge to
  be exposed at least once. This makes the nominal `dose_A3_50` condition stop
  at realized A3 coverage of approximately `0.68`, not exactly `0.50`.
- The nominal 80% and 90% conditions realize approximately `0.801` and `0.901`.
- These are preserved in `coverage_audit.csv`; the arm labels are retained
  only to match the handoff protocol and must not be reported as exact
  realized dose levels.

Execution interruption:

- The first soft invocation stopped externally after completing 22 runs with
  no Python error, OOM message, or partial run file.
- The identical command was resumed without `--force`. It skipped all 22
  completed runs and finished the remaining 8 runs successfully.

## D. Cross-modal and baseline strengthening

### D1. SMD norm-amplitude audit

Completed with seed `20261201`:

- 28 machines, two variants (`T0_hard`, `T4_soft`), 10 training epochs each.
- The script's repository-relative default data directory does not exist on
  this machine. The existing source data was located at
  `E:\MarketRisker\work\cert\OmniAnomaly\ServerMachineDataset` and passed
  explicitly with `--data-dir`.
- The upstream SoftCLT implementation exposes `casual=True` (historical
  spelling); this was verified against
  `references/softclt_repo/softclt_ts2vec/soft_ts2vec.py`.
- Upstream `models/timelags.py` emits `overflow encountered in exp` warnings
  while producing saturated 0/1 temporal weights. No source code, score
  aggregation, or metric was changed.

### D2. Daily AE and causal LSTM baselines

Completed:

- r4.2: 3 seeds (`20261116`-`20261118`), 20 epochs per model.
- r5.2: the same 3 seeds and 20 epochs.
- Each run evaluates both `MLP-AE-27D` and `LSTM-Pred-27D` with the frozen
  K grid, event tolerance, Top-K ranking, and cooldown used by the other
  supplement tasks.

### D3. Soft-instance alpha sweep

Completed:

- r4.2, alpha values `0.25`, `0.50`, `0.75`.
- 3 seeds per alpha (`20261116`-`20261118`).
- Every model uses the same 620-step random batch budget and the same r4.2
  split; each encoder is evaluated with bank cosine, bank Euclidean, and
  PAI-A1.

## Completion status

All A, B, C, and D commands completed. Required lightweight CSV/JSON
artifacts are present under `research/ccf-b/supplement_results/`; model
weights and score `.npz` files remain local and are not part of the
lightweight delivery.
