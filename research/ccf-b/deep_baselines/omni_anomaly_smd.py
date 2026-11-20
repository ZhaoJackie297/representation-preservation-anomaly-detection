from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from smd_common import (
    block_windows,
    evaluate_machine,
    load_machine,
    machine_names,
    normalize_from_train,
)
from protocol_utils import record_run_protocol


class OmniAnomalyAdapted(nn.Module):
    def __init__(
        self,
        input_dim: int,
        window: int,
        hidden_dim: int = 64,
        latent_dim: int = 3,
    ) -> None:
        super().__init__()
        self.window = window
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
        )
        self.q_mean = nn.Linear(hidden_dim, latent_dim)
        self.q_logvar = nn.Linear(hidden_dim, latent_dim)
        self.repeat = nn.Linear(latent_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, window, latent_dim))
        self.decoder = nn.GRU(
            latent_dim,
            hidden_dim,
            batch_first=True,
        )
        self.x_mean = nn.Linear(hidden_dim, input_dim)
        self.x_logvar = nn.Linear(hidden_dim, input_dim)

    def encode(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.encoder(values)
        hidden = hidden[-1]
        return self.q_mean(hidden), self.q_logvar(hidden)

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decoder_input = latent[:, None, :] + self.position
        hidden_init = torch.tanh(self.repeat(latent))[None, :, :]
        hidden, _ = self.decoder(decoder_input, hidden_init)
        return self.x_mean(hidden), self.x_logvar(hidden)

    def forward(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(values)
        latent = self.reparameterize(mean, logvar)
        x_mean, x_logvar = self.decode(latent)
        return x_mean, x_logvar, mean, logvar


def gaussian_nll(
    values: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    logvar = torch.clamp(logvar, -8.0, 8.0)
    return 0.5 * (
        (values - mean).pow(2) / torch.exp(logvar)
        + logvar
        + float(np.log(2.0 * np.pi))
    ).sum(dim=-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_windows(
    model: OmniAnomalyAdapted,
    windows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(
                windows[start : start + batch_size]
            ).to(device)
            mean, logvar, _, _ = model(batch)
            outputs.append(gaussian_nll(batch, mean, logvar).cpu().numpy())
    return np.concatenate(outputs, axis=0).reshape(-1)


def run_machine(
    *,
    name: str,
    data_dir: Path,
    output_root: Path,
    seed: int,
    window: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    latent_dim: int,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    output = output_root / f"seed_{seed}" / "machines" / f"{name}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        with np.load(output, allow_pickle=False) as data:
            return json.loads(str(data["metrics"]))

    start_time = time.time()
    train, test, label = load_machine(data_dir, name)
    train_norm, test_norm = normalize_from_train(train, test)
    train_windows = block_windows(train_norm, window)
    test_windows = block_windows(test_norm, window)

    set_seed(seed)
    model = OmniAnomalyAdapted(
        input_dim=train.shape[1],
        window=window,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_windows))
        for batch_start in range(0, len(order), batch_size):
            indices = order[batch_start : batch_start + batch_size]
            batch = torch.from_numpy(train_windows[indices]).to(device)
            mean, logvar, q_mean, q_logvar = model(batch)
            reconstruction = gaussian_nll(batch, mean, logvar).sum(dim=-1)
            kl = 0.5 * (
                q_mean.pow(2)
                + q_logvar.exp()
                - 1.0
                - q_logvar
            ).sum(dim=-1)
            loss = (reconstruction + 0.1 * kl).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

    train_scores = score_windows(
        model,
        train_windows,
        batch_size=batch_size,
        device=device,
    )[: len(train_norm)]
    test_scores = score_windows(
        model,
        test_windows,
        batch_size=batch_size,
        device=device,
    )[: len(test_norm)]
    metrics = {
        "machine": name,
        "seed": seed,
        "window": window,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "adaptation": "PyTorch GRU-VAE reconstruction probability",
        **evaluate_machine(label, train_scores, test_scores),
        "elapsed_seconds": time.time() - start_time,
    }
    np.savez_compressed(
        output,
        label=label,
        train_score=train_scores.astype(np.float32),
        test_score=test_scores.astype(np.float32),
        metrics=json.dumps(metrics),
    )
    print(
        f"{name}: Omni-adapted PR={metrics['pr_auc']:.4f}, "
        f"time={metrics['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261201])
    parser.add_argument("--machines", nargs="*", default=[])
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    selected = args.machines or machine_names(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_run_protocol(
        args.output_dir / "protocol.json",
        {
            "machines": selected,
            "window": args.window,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "implementation": "OmniAnomaly-adapted PyTorch GRU-VAE",
        },
        args.seeds,
    )
    for seed in args.seeds:
        rows = [
            run_machine(
                name=name,
                data_dir=args.data_dir,
                output_root=args.output_dir,
                seed=seed,
                window=args.window,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                device=device,
                force=args.force,
            )
            for name in selected
        ]
        frame = pd.DataFrame(rows)
        seed_root = args.output_dir / f"seed_{seed}"
        frame.to_csv(seed_root / "results_by_machine.csv", index=False)
        summary = {
            "seed": seed,
            "machines": len(frame),
            "macro_mean": {
                metric: float(frame[metric].mean())
                for metric in (
                    "pr_auc",
                    "roc_auc",
                    "pa_f1_top5",
                    "f1_q99",
                    "segment_recall_q99",
                )
            },
        }
        (seed_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
