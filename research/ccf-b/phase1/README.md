# Phase 1 research code

Date: 2026-11-19

This directory implements the controlled anomaly-signal study used to test
the temporal-soft bridge and the point/context signal decomposition.

## Entry points

- `controlled_signal_study.py`: creates deterministic anomaly injections on
  real CERT or CLUE normal sequences, reuses locked checkpoints, and evaluates
  point, context, combined, and iForest scores.
- `risk_case_study.py`: applies instant, EWMA, CUSUM, and training-graph
  propagation on the locked r4.2 PAI scores and exports fixed S4 cases.

The scripts are intentionally separate from training. No checkpoint is tuned
on an injected event.

## Profiles

```text
r42-temporal : T3_instance_soft vs T4_full_soft
r42-coverage : A0-official vs A1-Cov
clue-coverage: A0-official vs A1-Cov
```

## Signal types

```text
point     : one-day multichannel spike
sustained : seven-day shift
drift     : ten-day linear drift
switch    : seven-day blended account-behavior switch
```

All methods receive the same user set, event start dates, intensities, active
mask, causal context, and K/cooldown protocol.

## Smoke example

```powershell
python controlled_signal_study.py `
  --profile r42-temporal `
  --seeds 20261116 `
  --signal-types point `
  --strengths 1.0 `
  --background-users 32 `
  --event-users 8 `
  --device cuda `
  --output-dir E:\MarketRisker\work\cert\external\phase1_r42_temporal_smoke
```

## Tests

```powershell
python -m unittest -v test_controlled_signal_study.py test_risk_case_study.py
```

## Claim boundary

The generated events are a controlled mechanism benchmark built on real normal
logs. They are not new real datasets and cannot be used to claim external
validity. Negative results from CLUE-LDS remain in force even when a controlled
CERT injection produces a positive score separation.
