# SMD deep baselines

These scripts adapt two deep anomaly detectors to the repository's SMD
protocol.

The CERT 3-seed paired pilot and its claim boundary are documented in
`CERT_PAIRED_PILOT_RESULTS.md`.

Fetch the pinned external checkouts with:

```powershell
pwsh .\scripts\fetch_supplementary_baselines.ps1
```

## Anomaly Transformer

`anomaly_transformer_smd.py` imports the official model implementation from a
local checkout of `thuml/Anomaly-Transformer`. The adaptation trains one model
per SMD machine, uses the repository's train/test split, and scores the full
test sequence by completing each block with zero padding when necessary.

The public implementation uses a 100-point window. The adaptation retains that
window and uses 64-dimensional embeddings and three encoder layers to fit the
available GPU. This window differs from the 512-point T0/T4 pipeline and is
therefore reported as an architecture-specific adapted baseline rather than an
exactly matched window comparison.

## OmniAnomaly-adapted

The original OmniAnomaly release depends on TensorFlow 1.12 and Python 3.5/3.6.
`omni_anomaly_smd.py` is a PyTorch GRU-VAE reconstruction-probability port. It
must be reported as `OmniAnomaly-adapted`, not as the unmodified official
implementation.

The port keeps the core normal-window reconstruction model: recurrent encoder,
latent Gaussian state, recurrent decoder, and negative Gaussian log-likelihood
as the anomaly score. It does not reproduce the original normalizing flow or
connected Gaussian state-space prior.

## CERT representation baseline

`tnnclr_cert.py` reuses the official t-NNCLR encoder, augmentation,
projection, prediction, queue, and nearest-neighbor contrastive components. A
fixed representation bank and the repository's Top-K/cooldown evaluation are
added after pretraining.

## Commands

```powershell
$CertRoot = "<path-to-cert-runtime>"
$BaselineRoot = (Resolve-Path .\references\external).Path

python anomaly_transformer_smd.py `
  --repo "$BaselineRoot\Anomaly-Transformer" `
  --data-dir "$CertRoot\OmniAnomaly\ServerMachineDataset" `
  --output-dir "$CertRoot\external\smd_anomaly_transformer_v1" `
  --seeds 20261201 --machines machine-1-1

python omni_anomaly_smd.py `
  --data-dir "$CertRoot\OmniAnomaly\ServerMachineDataset" `
  --output-dir "$CertRoot\external\smd_omnianomaly_adapted_v1" `
  --seeds 20261201 --machines machine-1-1
```

Run the unit tests with:

```powershell
python -m unittest -v test_deep_baselines.py
```

Each training entry point merges resumed invocations into `protocol.json`.
The top-level `seeds` field is the completed union, while `seed_invocations`
preserves how that coverage was split across runs.
