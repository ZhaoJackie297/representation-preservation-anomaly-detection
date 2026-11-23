# Exp S7 unified 10-seed comparison

Date: 2026-11-22 (Asia/Shanghai).

## Frozen rules

- Seeds: `20261116`-`20261125` for every comparison.
- Existing `results/canonical/`, historical baseline archives, model
  checkpoints, and prepared datasets are read-only.
- New external artifacts are written only below
  `E:\MarketRisker\work\cert\external\unify_10seed_v1`.
- New repository summaries are written only below
  `research\ccf-b\supplement_results\exp_s7_10seed_unified`.

## Execution record

Commands, warnings, resumed runs, and paired-test decisions are appended below
as tasks finish.

### Environment and immutable inputs

- Python: `C:\Users\Lenovo\.conda\envs\ts2vec\python.exe`.
- Locked seeds: `20261116` through `20261125`.
- CERT r4.2 sequence input:
  `runtime\cert\processed_r42\r4.2_sequence_dataset.npz`.
- CERT r5.2 sequence input:
  `runtime\cert\processed_r52\r5.2_sequence_dataset.npz`.
- Paper-facing T3 reference:
  `research\ccf-b\supplement_results\exp_s1_r4.2\metrics_by_seed.csv`.
- No command in S7 wrote to `results\canonical\`, any `processed_*`
  directory, an old deep-baseline archive, or an S1-S6 result directory.

### Task 1: 10-seed deep baselines

Completed for Anomaly Transformer-adapted and t-NNCLR with the locked seeds.
New model artifacts were isolated under:

```text
E:\MarketRisker\work\cert\external\unify_10seed_v1\cert_tnnclr_10seed
E:\MarketRisker\work\cert\external\unify_10seed_v1\cert_anom_transf_10seed
```

Commands:

```powershell
C:\Users\Lenovo\.conda\envs\ts2vec\python.exe research\ccf-b\deep_baselines\tnnclr_cert.py `
  --repo E:\MarketRisker\references\t-nnclr `
  --dataset E:\MarketRisker\work\cert\processed_r42\r4.2_sequence_dataset.npz `
  --labels E:\MarketRisker\work\cert\processed_r42\r4.2_malicious_account_days.csv `
  --incidents E:\MarketRisker\work\cert\processed_r42\r4.2_incidents.csv `
  --output-dir E:\MarketRisker\work\cert\external\unify_10seed_v1\cert_tnnclr_10seed `
  --runtime-dir runtime\cert `
  --seeds 20261116 20261117 20261118 20261119 20261120 20261121 20261122 20261123 20261124 20261125 `
  --window 96 --epochs 10 --batch-size 64 --embed-dim 64 --layers 4 --heads 4 --device cuda

C:\Users\Lenovo\.conda\envs\ts2vec\python.exe research\ccf-b\deep_baselines\anomaly_transformer_cert.py `
  --repo E:\MarketRisker\references\Anomaly-Transformer `
  --dataset E:\MarketRisker\work\cert\processed_r42\r4.2_sequence_dataset.npz `
  --labels E:\MarketRisker\work\cert\processed_r42\r4.2_malicious_account_days.csv `
  --incidents E:\MarketRisker\work\cert\processed_r42\r4.2_incidents.csv `
  --output-dir E:\MarketRisker\work\cert\external\unify_10seed_v1\cert_anom_transf_10seed `
  --runtime-dir runtime\cert `
  --seeds 20261116 20261117 20261118 20261119 20261120 20261121 20261122 20261123 20261124 20261125 `
  --window 64 --epochs 10 --batch-size 32 --d-model 64 --layers 3 --device cuda

C:\Users\Lenovo\.conda\envs\ts2vec\python.exe research\ccf-b\deep_baselines\summarize_cert_baselines.py `
  --tnnclr-root E:\MarketRisker\work\cert\external\unify_10seed_v1\cert_tnnclr_10seed `
  --anomaly-transformer-root E:\MarketRisker\work\cert\external\unify_10seed_v1\cert_anom_transf_10seed `
  --output-dir research\ccf-b\supplement_results\exp_s7_10seed_unified\deep_baselines
```

Outputs:

```text
deep_baselines\cert_deep_baselines_by_seed.csv
deep_baselines\cert_deep_baselines_summary.csv
deep_baselines\cert_deep_baselines_summary.json
```

### Task 2: 10-seed T3/T4

Training command:

```powershell
C:\Users\Lenovo\.conda\envs\ts2vec\python.exe runtime\cert\e2_softclt_instance.py `
  --seeds 20261116 20261117 20261118 20261119 20261120 20261121 20261122 20261123 20261124 20261125 `
  --epochs 10 --batch-size 16 --output-tag r42_10seed_unified
```

Because `e2_softclt_instance.py --output-tag` owns its checkpoint location, the
checkpoints are under:

```text
runtime\cert\experiments\representation_preservation\e2_instance\r42_10seed_unified
```

They were left in place. Moving them would break the paths recorded by the
mandated command.

Evaluation command:

```powershell
C:\Users\Lenovo\.conda\envs\ts2vec\python.exe research\ccf-b\supplement\exp_s7_t4_10seed_eval.py
```

The evaluator loops over T3 and T4 checkpoints for all 10 seeds and five
scorers. Score arrays are under:

```text
E:\MarketRisker\work\cert\external\unify_10seed_v1\cert_t4_10seed_scores
```

Repository-facing outputs:

```text
t4_10seed_by_seed.csv
t4_10seed_summary.csv
```

The final paper-facing T3 row uses the pre-existing S1 10-seed result because
that is the frozen reference named by the handoff. The newly trained T3 is used
only as a reproduction check.

### Task 3: MLP-AE and LSTM-Pred

`exp_s4_daily_ae_and_hyperparams.py` was minimally extended with an optional
`--output-dir` argument. Its default remains
`research\ccf-b\supplement_results\exp_s4_baselines_hp`, so historical behavior
is unchanged.

Commands:

```powershell
C:\Users\Lenovo\.conda\envs\ts2vec\python.exe research\ccf-b\supplement\exp_s4_daily_ae_and_hyperparams.py ae-baseline `
  --dataset r4.2 --epochs 20 `
  --seeds 20261116 20261117 20261118 20261119 20261120 20261121 20261122 20261123 20261124 20261125 `
  --output-dir research\ccf-b\supplement_results\exp_s7_10seed_unified

C:\Users\Lenovo\.conda\envs\ts2vec\python.exe research\ccf-b\supplement\exp_s4_daily_ae_and_hyperparams.py ae-baseline `
  --dataset r5.2 --epochs 20 `
  --seeds 20261116 20261117 20261118 20261119 20261120 20261121 20261122 20261123 20261124 20261125 `
  --output-dir research\ccf-b\supplement_results\exp_s7_10seed_unified
```

Observed per-seed wall times in the final run were approximately 19-25 seconds
for r4.2 and 27-31 seconds for r5.2. All 20 seed-dataset jobs (40 trained
models) completed without an error.

Outputs:

```text
ae_r4.2\ae_baselines_by_seed.csv
ae_r4.2\ae_baselines_summary.csv
ae_r5.2\ae_baselines_by_seed.csv
ae_r5.2\ae_baselines_summary.csv
```

The r5.2 summary is reported separately in
`r5.2_ae_lstm_10seed_summary.csv`; it is not mixed into the r4.2 main table.

### Unified summary and paired-test construction

Builder:

```text
research\ccf-b\supplement\exp_s7_build_unified_tables.py
```

Command:

```powershell
C:\Users\Lenovo\.conda\envs\ts2vec\python.exe research\ccf-b\supplement\exp_s7_build_unified_tables.py
```

The 12-row main table contains four external baselines, four T4 scorer
variants, and four matching T3 scorer variants. `PAI-Mean` denotes the project
weights `1.0:0.6:0.4`, not official PAI.

Paired tests use `candidate - reference` values aligned by seed. External
baselines are compared with T3 PAI-Mean. Each T4 scorer is compared with the
same-scorer T3 result. The five metric families are PR-AUC, ROC-AUC, and Event
Recall at K = 5, 10, and 20. Within each metric family, Holm correction is
applied across all eight planned comparisons, giving 40 tests in total.

For paired deltas with zero variance, the builder records p = 1 when every
delta is zero and p = 0 when every nonzero delta is identical. It never emits
an unexplained `NaN` p-value.

Outputs:

```text
table3_10seed_summary.csv
table3_10seed_paired_tests.csv
reproduction_checks.json
r5.2_ae_lstm_10seed_summary.csv
```

### Reproduction checks

1. Old S4 AE/LSTM versus new S7 AE/LSTM:
   - r4.2: 36 overlapping test rows, exact within `1e-12`.
   - r5.2: 18 overlapping test rows, exact within `1e-12`.
   - Maximum absolute deltas are zero for PR-AUC, ROC-AUC, and event recall.
2. Existing S1 T3 versus newly trained unified T3:
   - The first three seeds are present, but the results are not bitwise equal.
   - Across test rows, maximum absolute deltas are 0.002670 for PR-AUC,
     0.011435 for ROC-AUC, and 0.083333 for event recall.
   - Root cause boundary: the historical S1 implementation trains from a
     precomputed random batch table via `CoverageSoftTS2Vec`, while the unified
     e2 command uses the SoftCLT repository's `TS2Vec.fit`. The two
     implementations share the seed and budget but are not bitwise identical.
   - This is reported as a numerical reproduction check, not claimed as exact
     reproduction.
3. Historical T4 archive:
   - `paired_pilot_20261201_03_v1` uses seeds `20261201-20261203`.
   - It has zero overlap with the locked S7 seeds `20261116-20261118`, so an
     exact first-three comparison is impossible.
4. Historical deep-baseline archives:
   - Anomaly Transformer and t-NNCLR archives also use
     `20261201-20261203`.
   - They have zero overlap with the requested first three S7 seeds.

### Main result boundary

- The new T4 does not establish a consistent advantage over T3.
- No T4-versus-T3 comparison remains significant after Holm correction.
- T4 Bank-Cosine has the largest observed mean PR-AUC and ROC-AUC, but paired
  T4-minus-T3 differences are small and uncertain.
- T4 PAI-Mean improves mean PR-AUC by 0.000151 with 7/10 positive wins, while
  Event Recall@K=10 and K=20 trend slightly downward. This is a scorer/event
  trade-off, not evidence of uniform improvement.
- The frozen T3 PAI-Mean remains ahead of all four external baselines on
  PR-AUC and ROC-AUC after Holm correction. At event level, its advantage is
  metric- and baseline-specific rather than universal.

### Optional Hard-Cov decision

Hard-Cov was not added to the final 12-row table. It is explicitly optional in
the handoff and is not one of the methods requested for
`table3_10seed_summary.csv`. No incomplete 10-seed Hard-Cov row is presented as
a confirmatory comparison.
