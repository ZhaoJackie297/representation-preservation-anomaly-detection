# Exp S6 Tier-2 run log

Date: 2026-11-22 (Asia/Shanghai).

Environment:

- Python: `C:\Users\Lenovo\.conda\envs\ts2vec\python.exe`
- PyTorch: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Script: `research/ccf-b/supplement/exp_s6_tier2.py`

## S6a. CERT r6.2 PAI-Mean / PAI-Official

Completed with zero retraining:

- Reused Hard-Rand and SoftInst-Rand checkpoints for three seeds each.
- Used causal sliding encoding and a 500-cluster training-period bank.
- PAI-Mean uses `euc_z + 0.5*g_z + 0.5*q_z`.
- PAI-Official uses `0.6*euc_z + 0.4*g_z + 0.2*q_z`.
- K grid: 5, 10, 20.

Checkpoint loading note:

- The stored state dictionaries are `AveragedModel` states with `module.*`
  keys and `n_averaged`. The first loader attempt stripped those keys and
  correctly failed before writing results. The loader was fixed to preserve
  the saved state format.

## S6b. Zero-label training-period modality selector

Completed for:

- CERT r4.2: Hard-Rand and SoftInst-Rand, 3 seeds each.
- CERT r5.2: Hard-Rand and SoftInst-Rand, 3 seeds each.
- CERT r6.2: Hard-Rand and SoftInst-Rand, 3 seeds each.
- CLUE-LDS: Hard-Rand and SoftInst-Rand, 3 seeds each.
- SMD: T0_hard and T3_inst_soft, 28 machines x 3 seeds.

Strict zero-label rows use only active cells with `t <= train_end`.

SMD source limitation:

- No SMD model checkpoints or separate training-only diagnostic table were
  retained from S5c.
- SMD rows therefore come from the existing
  `smd_norm_radial_all_machines_3seed.csv`:
  `spearman_norm_vs_xnorm` and `radial_energy_share_normal` use the test
  normal segment, while `norm_cv_normal` also uses test normal rows.
- These rows are marked `zero_label_valid=False` and
  `metric_source=test_normal_segment_existing_s5c_csv`. They are a post-hoc
  routing audit, not evidence that SMD itself satisfies the zero-label
  selector protocol.

SMD mapping correction:

- The first summary attempt mapped every non-T0 SMD variant to T3 and
  incorrectly included T4_soft, producing 168 T3 rows.
- The filter was corrected to T0_hard and T3_inst_soft only. The zero-retrain
  S6b command was rerun and now produces 84 rows per SMD arm.

## S6c. CERT r5.2 explicit L2 regularization transfer

Completed nine new trainings:

- Arms: `Hard-L2Reg-b0.02`, `Hard-L2Reg-b0.20`, `Hard-L2Reg-b1.00`.
- Seeds: `20261116`, `20261117`, `20261118`.
- Exact budget: 1,250 optimizer steps per seed.
- Hook assertion: 1,250 calls for every run.
- Five scorers: Mask L1, Cosine, Euclidean, PAI-Mean, PAI-Official.

Summary reconstruction bug:

- Existing Hard-Rand and SoftInst-Rand control rows initially lacked
  `split="test"`, so the first groupby omitted them.
- The by-seed data and model weights were intact. The script and summary
  builder were corrected, and `s6c-summary` rebuilt the final summary without
  retraining.

## Constraints

- `results/canonical/` was not modified.
- Existing S1-S5 result CSVs were not modified.
- S6c checkpoints were written under
  `research/ccf-b/supplement_results/exp_s6_tier2/s6c_models/` and are not
  part of the lightweight GitHub CSV delivery.
