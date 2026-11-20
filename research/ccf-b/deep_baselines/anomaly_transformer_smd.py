from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
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


def my_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    result = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(result, dim=-1), dim=1)


def normalized_prior(prior: torch.Tensor) -> torch.Tensor:
    return prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def association_terms(
    series: list[torch.Tensor],
    prior: list[torch.Tensor],
    temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    series_loss = torch.zeros(
        series[0].shape[0],
        series[0].shape[-1],
        device=series[0].device,
    )
    prior_loss = torch.zeros_like(series_loss)
    for current_series, current_prior in zip(series, prior):
        normalized = normalized_prior(current_prior)
        series_loss = series_loss + (
            my_kl_loss(current_series, normalized.detach())
            + my_kl_loss(normalized.detach(), current_series)
        )
        prior_loss = prior_loss + (
            my_kl_loss(normalized, current_series.detach())
            + my_kl_loss(current_series.detach(), normalized)
        )
    series_loss = series_loss / len(series)
    prior_loss = prior_loss / len(prior)
    if temperature is not None:
        series_loss = series_loss * temperature
        prior_loss = prior_loss * temperature
    return series_loss, prior_loss


def score_windows(
    model: nn.Module,
    windows: np.ndarray,
    *,
    window: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    criterion = nn.MSELoss(reduction="none")
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(
                windows[start : start + batch_size]
            ).to(device)
            reconstruction, series, prior, _ = model(batch)
            point_loss = criterion(batch, reconstruction).mean(dim=-1)
            series_loss, prior_loss = association_terms(
                series,
                prior,
                temperature=50,
            )
            metric = torch.softmax(-series_loss - prior_loss, dim=-1)
            outputs.append(
                (metric[:, None] * point_loss).cpu().numpy().reshape(-1)
            )
    return np.concatenate(outputs, axis=0)


def run_machine(
    *,
    name: str,
    data_dir: Path,
    output_root: Path,
    seed: int,
    window: int,
    epochs: int,
    batch_size: int,
    d_model: int,
    layers: int,
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
    model = AnomalyTransformer(
        win_size=window,
        enc_in=train.shape[1],
        c_out=train.shape[1],
        d_model=d_model,
        n_heads=8,
        e_layers=layers,
        d_ff=d_model * 2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_windows))
        for batch_start in range(0, len(order), batch_size):
            indices = order[batch_start : batch_start + batch_size]
            batch = torch.from_numpy(train_windows[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, series, prior, _ = model(batch)
            series_loss, prior_loss = association_terms(series, prior)
            reconstruction_loss = criterion(reconstruction, batch)
            loss_series = reconstruction_loss - 3.0 * series_loss.mean()
            loss_prior = reconstruction_loss + 3.0 * prior_loss.mean()
            loss_series.backward(retain_graph=True)
            loss_prior.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

    train_scores = score_windows(
        model,
        train_windows,
        window=window,
        batch_size=batch_size,
        device=device,
    )[: len(train_norm)]
    test_scores = score_windows(
        model,
        test_windows,
        window=window,
        batch_size=batch_size,
        device=device,
    )[: len(test_norm)]
    metrics = {
        "machine": name,
        "seed": seed,
        "window": window,
        "epochs": epochs,
        "batch_size": batch_size,
        "d_model": d_model,
        "layers": layers,
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
        f"{name}: AT PR={metrics['pr_auc']:.4f}, "
        f"time={metrics['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261201])
    parser.add_argument("--machines", nargs="*", default=[])
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    global AnomalyTransformer
    from model.AnomalyTransformer import AnomalyTransformer

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    selected = args.machines or machine_names(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_run_protocol(
        args.output_dir / "protocol.json",
        {
            "repo": str(args.repo),
            "machines": selected,
            "window": args.window,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "layers": args.layers,
            "adaptation": "per-machine SMD; full-length block scoring",
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
                d_model=args.d_model,
                layers=args.layers,
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
