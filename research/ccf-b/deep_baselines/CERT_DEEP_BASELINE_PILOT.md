# CERT deep-baseline adaptation pilot

Date: 2026-11-20.

Status: exploratory. Seeds `20261201-20261203`. This pilot is not part of the
confirmatory 20-seed family.

The T0/T4 paired extension is reported separately in
`CERT_PAIRED_PILOT_RESULTS.md`.

## Protocol

Both baselines use:

- CERT r4.2 train/validation/test boundaries;
- the same 27-dimensional user-day inputs;
- labels only after scores are produced;
- the repository's K=5/10/20 and seven-day cooldown evaluation.

Anomaly Transformer uses a 64-day window, 64-dimensional embeddings, three
encoder layers, 10 epochs, and one model over all user windows.

t-NNCLR uses the official encoder and NNCLR components with a 96-step window,
10 epochs, and a fixed training-representation bank. It is not a label-trained
classifier.

## Results

| Method | Seeds | PR-AUC | ROC-AUC | K=10 event recall |
| --- | ---: | ---: | ---: | ---: |
| Anomaly Transformer-adapted | 3 | 0.002324 +/- 0.0000003 | 0.497332 +/- 0.003703 | 0.0556 +/- 0.0241 |
| t-NNCLR + representation bank | 3 | 0.003709 +/- 0.001003 | 0.643231 +/- 0.066294 | 0.3194 +/- 0.2055 |

Anomaly Transformer remains at the random point-ranking level even after a
separate 30-epoch seed-20261201 sensitivity run:

```text
PR-AUC = 0.002344
ROC-AUC = 0.499098
K=10 event recall = 0/24
```

## Interpretation

These results do not support a claim that the adapted baselines detect CERT
user-day insider incidents well. They also do not establish that the original
methods are weak on their native benchmarks. The task changes the unit of
analysis from a machine/sensor series to a sparse cross-user daily cohort, and
the window sizes differ from the 512-point T0/T4 SMD comparison.

The baselines must remain in the paper as an explicit adaptation result:
the proposed method was not compared only with iForest. However, paired
inference against T0/T4 would require running T0/T4 on the same new seeds and is
not claimed here.

Artifacts:

- `results/cert_deep_baselines_by_seed.csv`;
- `results/cert_deep_baselines_summary.csv`;
- `results/anomaly_transformer_seed_*.json`;
- `results/tnnclr_seed_*.json`.
