# Exp S8 Run Log

Date: 2026-11-23 (Asia/Shanghai).

## Fixed protocol

- Dataset: CERT r4.2.
- Seeds: `20261116 20261117 20261118 20261119 20261120`.
- Steps: 620 (10 epochs x 62 steps), batch size 16.
- Scorers: `plain_l1`, `bank_euclidean`, `bank_cosine`, `pai_a1`, `pai_official`.
- Event K: 5, 10, 20 with the canonical 7-day cooldown.
- New writes were restricted to `C:\Users\Lenovo\Documents\Codex\2026-11-07\files-pasted-by-the-user-2026\representation-preservation-anomaly-detection\research\ccf-b\supplement_results\exp_s8_necessity_and_dynamics`. S1-S7, `results/canonical/`, `processed_*`, and historical archives were read-only.

## Command

```powershell
C:\Users\Lenovo\.conda\envs\ts2vec\python.exe research\ccf-b\supplement\exp_s8_necessity_and_dynamics.py --experiments all

# Post-audit correction: evaluate trajectory snapshots from a CPU state clone
# so the live CUDA training path is never touched by measurement.
C:\Users\Lenovo\.conda\envs\ts2vec\python.exe research\ccf-b\supplement\exp_s8_necessity_and_dynamics.py --experiments s8a s8b --force

# Rebuild cross-experiment tables and reports after the correction.
C:\Users\Lenovo\.conda\envs\ts2vec\python.exe research\ccf-b\supplement\exp_s8_necessity_and_dynamics.py --experiments cross finalize
```

## Execution summary

- Primary full-matrix wall time: 39.29 minutes.
- CPU snapshot correction rerun: approximately 35 minutes.
- Final cross-table/report assembly: less than 1 minute.
- Summed elapsed time of the final training runs: 49.34 minutes.
- Total wall time including loading, repeated checkpoint audits, table builds,
  and the corrected trajectory rerun: approximately 75 minutes.
- S8a runs: 15 (`3` temperatures x `5` seeds).
- S8b trajectory seed-arm rows: 30 x 11 snapshots = 330 rows.
- S8c alpha rows: 1050 scorer/K rows across `7` alpha values and `5` seeds.
- S8c tau rows: 750 scorer/K rows across `5` tau_mult values and `5` seeds.

## Hook and gradient assertions

- Exact 620 hierarchy-loss calls on every Soft-Sphere run: True.
- Every Soft-Sphere run has at least 1,240 gradient hook calls.
- Maximum absolute deviation of mean post-normalization norm from 1.0: 7.883e-09.
- Off-diagonal soft mass must be positive on every Soft-Sphere optimization call.

## S5a reproduction

- Fresh S8b training versus stored S5a metrics: maximum absolute delta = 1.667e-01.
- A direct no-trajectory Hard-Rand/20261116 rerun also differs from frozen S5a;
  its exact deltas are recorded in `s5a_direct_rerun_audit.json`.
- S5a checkpoint geometry reevaluation:
  - maximum absolute mean-norm delta = 1.776e-15;
  - maximum absolute effective-rank delta = 1.776e-15.

Per-arm audit:

| Arm | Rows | Max PR delta | Max ROC delta | Max recall delta | Exact |
|---|---:|---:|---:|---:|---|
| Hard-L2Reg-b0.20 | 150 | 6.043e-03 | 1.213e-02 | 1.667e-01 | False |
| Hard-L2Reg-b1.00 | 150 | 1.056e-02 | 5.026e-02 | 1.522e-01 | False |
| Hard-Rand | 150 | 3.107e-03 | 1.447e-02 | 1.250e-01 | False |
| PermSoft-Rand | 150 | 3.089e-03 | 1.592e-02 | 1.667e-01 | False |
| SoftInst-Rand | 150 | 4.728e-03 | 1.013e-02 | 1.667e-01 | False |

Reproduction boundary: a no-trajectory direct rerun of the same S5a command also differs numerically on this GPU. Strict identity is therefore claimed only for checkpoint geometry; causal benchmark metrics use the frozen S5a tables.


## Trajectory RNG isolation

- Snapshot evaluation saves and restores Python `random`, NumPy, CPU Torch, and all CUDA RNG states.
- Each snapshot is copied to a fresh CPU model before encoding, preventing an
  evaluation convolution from changing the live CUDA training path.
- Snapshot points are exactly steps 0, 62, 124, ..., 620.
- Final downstream metrics are evaluated only after training, using the canonical full CERTr4.2 encoding and scorers.

## Artifact boundaries

- No file under `research/ccf-b/supplement_results/exp_s1_r4.2` through `exp_s7_10seed_unified` was written.
- No S5a/S5b/S5c/S5d result or checkpoint was overwritten.
- The two historical l2reg directories were read-only references.
