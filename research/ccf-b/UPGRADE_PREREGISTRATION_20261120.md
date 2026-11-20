# Upgrade preregistration

Date: 2026-11-20.

This file freezes the seed ranges and decision rules added after the original
v5 and CCF-B result sets. Existing seeds are not replaced or filtered.

## New seed ranges

- CERT extension seeds: `20261201-20261220`, inclusive.
- SMD extension seeds: `20261201-20261205`, inclusive.
- All seeds included in a frozen core matrix must be run and reported.
- Optional stopping, selecting only favorable seeds, or moving a seed from
  exploratory to confirmatory after seeing its result is prohibited.
- If a code defect invalidates a run, the defect and rerun must be recorded;
  the invalid run is not silently deleted.

Additional seeds reduce training variance. They do not increase the number of
independent CERT events or SMD machines.

The 20-seed CERT range is the target for the final core-method matrix. A new
deep baseline may first run the first three seeds as an explicitly exploratory
adaptation pilot. A pilot is excluded from confirmatory claims and must either
be extended to the frozen range or remain labelled exploratory. The pilot
cannot decide which of the 20 seeds are reported.

## Primary protocol

The existing data splits, features, K grid, seven-day cooldown, robust score
calibration, and test labels remain unchanged.

CERT endpoints:

1. test PR-AUC;
2. test ROC-AUC;
3. canonical K=10 event recall.

SMD is evaluated separately for each of the 28 machines. The primary SMD
endpoint is macro PR-AUC; macro ROC-AUC and PA-F1@Top-5 are secondary. Paired
comparisons are computed on the same seed-machine pair.

## Baseline adaptations

- `T0_matched`: the existing native TS2Vec 512-point-window control.
- `T4_full_soft`: the existing full SoftCLT double-end method.
- `Anomaly Transformer`: official model code, per-machine SMD input with one
  model per machine. CERT adaptation uses one 27-dimensional series per user,
  followed by the same K and cooldown evaluation. Pinned commit:
  `b0ee470c8012bff857fb600462aec6209c4a18d9`.
- `OmniAnomaly`: official architecture or a documented PyTorch port when the
  original TensorFlow 1 environment is not executable. A port must be labelled
  `OmniAnomaly-adapted`, not `official OmniAnomaly`. Official reference commit:
  `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4`.
- Representation baselines such as TS-TCC and T-NNCLR use the same output
  scoring and evaluation protocol as T0/T4. Encoder-only comparisons are not
  presented as end-to-end detector comparisons. Pinned commits:
  `9c6439991202986a2af108798d5092ea34f0e9ed` (t-NNCLR) and
  `15bdfd6eced6aeb6c638839828effe1b4b2ea391` (TS-TCC).

Baselines receive the same data split, preprocessing, maximum tuning budget,
and evaluation code. Adaptation changes and compute limits must be listed in
the appendix.

## Official PAI reporting

The project score remains `PAI-A1` with weights `1.0:0.6:0.4`. The official
PAI scalar weights `0.6:0.4:0.2` are reported separately as a sensitivity
analysis. Neither is tuned on the test split.

## Event-count boundary

CERT r6.2 contains five threat events. It is a stress audit and case study,
not a confirmatory external test. Additional training seeds cannot repair the
five-event limitation.

## Negative-result policy

- temporal-soft remains a conditional or negative end-to-end result;
- magG keeps the point-level versus event-level conflict;
- Coverage keeps the negative CLUE-LDS result;
- r6.2 keeps the five-event limitation;
- the final claim is CERT-focused and does not call the method universal SOTA.

Any change to these rules requires a dated erratum before the corresponding
new run is inspected.
