# Exp S5 supplement run log

Start date: 2026-11-21 (Asia/Shanghai).

## Frozen rules

- Existing `results/canonical/`, historical weights, and prepared splits were
  not modified.
- New scripts were copied from
  `C:\Users\Lenovo\Desktop\表征保全论文_ExpS5补实验包`.
- New artifacts are written only below
  `research/ccf-b/supplement_results/`.
- Execution order is Task A -> Task B -> Task C -> optional Task D.

## Environment

- Python: `C:\Users\Lenovo\.conda\envs\ts2vec\python.exe`
- Python version: 3.10.22
- PyTorch: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB
- NumPy: 2.2.6
- pandas: 2.3.3
- SciPy: 1.15.3
- scikit-learn: 1.7.2

## Path preflight

- CERT r4.2 sequence data: present.
- CERT r5.2 sequence data: present.
- CERT r6.2 processed data: available externally at
  `E:\MarketRisker\work\cert\processed_r62`, but not linked into the runtime
  directory at the start of this run.
- Existing S1/S2 outputs from the previous supplement round are available for
  matched-step and checkpoint reuse.

## Execution record

Commands, warnings, parameter adaptations, timings, and terminal summaries are
appended below as each task finishes.

## Task A. Norm intervention

Completed the full 11-arm x 5-seed matrix: 55 run directories.

- Every intervention seed used exactly 620 optimizer steps.
- `hook_calls=620` for all 50 explicit interventions; `Hard-Rand` correctly
  has `hook_calls=0`.
- Full matrix geometry ranged from raw Hard-Rand norm `3.489` to
  Hard-L2Reg-b1.00 norm `0.339`.
- The preregistered four L2Reg strengths realized norms of approximately
  `2.675`, `1.907`, `1.033`, and `0.339`; none entered the handoff's requested
  `1.2`-`1.6` alignment band.

Additional geometry-alignment arm:

- Because the four original beta values straddled the target band, the handoff
  prescribed one intermediate run. `Hard-L2Reg-b0.2` was run for all five
  seeds in a separate output directory to preserve the original 11-arm
  aggregates:
  `exp_s5a_norm_intervention_l2reg_b0.2/`.
- It completed 5/5 seeds with mean norm `1.374455` and effective rank
  `14.782644`, within the requested band.
- No arm or metric from the original matrix was overwritten by this addition.

Main test K=10 observations:

- Hard-Rand bank cosine PR-AUC `0.011383`; PAI-A1 `0.011247`.
- SoftInst-Rand bank cosine `0.016702`; PAI-A1 `0.014110`.
- PermSoft-Rand bank cosine `0.017274`; PAI-A1 `0.013313`.
- UniformSoft-Rand bank cosine `0.016330`; PAI-A1 `0.013111`.
- These values are reported as-is; no mechanism claim is made in this log.

## Task B. Symmetry, norms, and iForest audit

Completed:

1. `audit-iforest-t0`
2. `hard-r42-10ep`
3. `hard-r52-full`

r4.2 Hard-Rand-10ep:

- 10/10 seeds at 620 steps.
- Norm mean/std: `3.512804 +/- 0.069138`.
- Effective-rank mean/std: `9.277279 +/- 0.284694`.

r5.2 Hard-Rand:

- Hard-Rand-10ep: 10/10 seeds, 1250 steps, norm
  `3.805813 +/- 0.047670`.
- Hard-Rand-Matched: 10/10 seeds, exact per-seed steps
  `2631, 2637, 2640, 2648, 2654` as applicable (mean `2640.1`).
- Matched norm `3.957291 +/- 0.038192`; effective rank
  `11.300946 +/- 0.274430`.
- Matched steps were loaded exactly from the completed S1
  `SoftInst-Rand-Matched` run metadata, not from the fixed fallback.

iForest/T0 audit:

- The audit scanned 19 existing repository files containing relevant
  `0.00744`, iForest, T0, mask, or isolation strings.
- Recomputed iForest PR-AUC was `0.006994` for `random_state=42`,
  `0.007213` for `random_state=0`, and `0.006761` for
  `random_state=2024`.
- These recomputations do not reproduce the canonical historical
  `iForest-27D=0.0074400944975706` exactly. The audit JSON retains the
  found historical rows and the recomputed values verbatim; no historical
  result was altered.

## Task C. Three-seed norm/radial audit

### C1. CERT r4.2

Completed using the Task A checkpoints from seeds `20261116`-`20261118`:

- Hard-Rand (T0) and SoftInst-Rand (T3) each completed 3/3 seeds.
- Hard-Rand mean normal norm: `3.467207`.
- SoftInst-Rand mean normal norm: `1.387792`.
- Hard-Rand radial-energy share on anomalies: `0.032612`.
- SoftInst-Rand radial-energy share on anomalies: `0.065322`.
- The detailed by-seed table and macro summary were written under
  `exp_s5c_norm_corr_3seed/cert_r42_3seed/`.

### C2. SMD 28 machines x 3 seeds

Completed:

- Three seeds (`20261201`-`20261203`) x 28 machines x three variants
  (`T0_hard`, `T3_inst_soft`, `T4_soft`).
- Total machine-level rows: 252.
- `T3_inst_soft` directly aligns the main instance-soft variant to SMD.
- Macro means across all machine/seed rows:
  - T0 normal norm `10.318467`, anomalous radial share `0.450232`.
  - T3 normal norm `9.269292`, anomalous radial share `0.449234`.
  - T4 normal norm `9.180504`, anomalous radial share `0.447158`.
  - Euclidean-bank PR-AUC: T0 `0.439886`, T3 `0.447474`,
    T4 `0.446773`.
  - Cosine-bank PR-AUC: T0 `0.154848`, T3 `0.147885`,
    T4 `0.146938`.
- Upstream `models/timelags.py` again emitted saturated exponential
  warnings; no source or metric logic was changed.

## Task D. Unified CLUE-LDS and CERT r6.2

The handoff package did not include a Task D script. Both datasets were
available locally, so a minimal script was added at
`research/ccf-b/supplement/exp_s5d_unify_clue_r62.py`.

Implementation notes:

- Uses upstream `SoftTS2Vec.fit(..., n_epochs=10)` for the standard random
  batch protocol.
- Uses fixed 256-day training chunks when a sequence is longer, matching the
  existing CLUE cap-256 protocol. Full test sequences are still encoded for
  evaluation.
- The initial custom `fit_coverage_table` implementation was stopped before
  completing a SoftInst seed because it took over 40 minutes on CLUE's
  1911-day sequences. It was replaced by the substantially faster native
  SoftTS2Vec path above. No metric from the abandoned run is used.
- `Hard-Rand-10ep` and `SoftInst-Rand-10ep` share data, chunking, seeds,
  K=5/10/20, 7-day cooldown, and evaluation code.
- Because CLUE has 52 features and r6.2 has 27, the CERT-only PAI component
  contract was not applied. Scores are masked-L1, bank cosine, and bank
  Euclidean, which are available under both schemas.

### D1. CLUE-LDS

- 10 seeds (`20261116`-`20261125`), 20/20 runs complete.
- 838 users, 1911 days, 1698-day train period, 48 switch events.
- Nominal random-batch steps: 3140 over ten epochs after fixed 256-day
  training chunking.
- Test K=10 summary:
  - Hard plain `0.652585`, Hard cosine `0.672140`, Hard Euclidean
    `0.680857` PR-AUC.
  - Soft plain `0.656668`, Soft cosine `0.662671`, Soft Euclidean
    `0.678952` PR-AUC.
  - Event recall is near saturation for all methods (`0.971`-`0.985`),
    so it does not separate them.

### D2. CERT r6.2

- 3 seeds (`20261116`-`20261118`), 6/6 runs complete.
- Training uses the repository's fixed deterministic 1000-user subset;
  evaluation uses all 4000 users, matching the existing r6.2 protocol.
- 620 nominal random-batch steps per run.
- Test K=10 summary:
  - Hard plain `0.000800`, Hard cosine `0.000288`, Hard Euclidean
    `0.000433` PR-AUC.
  - Soft plain `0.000850`, Soft cosine `0.000355`, Soft Euclidean
    `0.000760` PR-AUC.
  - All Soft methods hit `2/5` events at K=10; Hard Euclidean also hits
    `2/5`, while Hard plain/cosine hit `1.667/5` and `1.333/5`
    on average.
- These five-event results remain stress-audit evidence only.

## Completion status

Tasks A, B, C, and D completed. Required lightweight CSV/JSON files and this
run log are ready for packaging into `supplement_s5_results_light.zip`.
