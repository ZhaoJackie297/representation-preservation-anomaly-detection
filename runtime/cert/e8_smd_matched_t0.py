from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e8"
    / "smd_matched_t0"
)
WINDOW = 512
SEEDS = [20261116, 20261117, 20261118]

sys.path.insert(0, str(ROOT))
from e7_smd_pai import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    amplitude_scores,
    bank_distances,
    evaluate_machine,
    z_from_train,
)
from smd_pipeline import load_machine, machine_names  # noqa: E402
from train_r42_ts2vec import TS2Vec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        default=SEEDS[0],
    )
    parser.add_argument("--machines", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def make_windows(train: np.ndarray) -> np.ndarray:
    count = len(train) // WINDOW
    if count < 2:
        raise ValueError("At least two complete windows are required.")
    return train[: count * WINDOW].reshape(
        count,
        WINDOW,
        train.shape[1],
    ).astype(np.float32)


def run_machine(name: str, seed: int, force: bool) -> dict:
    output = OUT / f"seed_{seed}" / "machines" / f"{name}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        with np.load(output, allow_pickle=False) as data:
            return json.loads(str(data["metrics"]))
    start = time.time()
    train, test, label = load_machine(name)
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) < 1e-8, 1.0, train.std(axis=0))
    train_norm = ((train - mean) / std).astype(np.float32)
    test_norm = ((test - mean) / std).astype(np.float32)
    windows = make_windows(train_norm)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_id = 0 if torch.cuda.is_available() else "cpu"
    model = TS2Vec(
        input_dims=train_norm.shape[1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device_id,
        batch_size=16,
        lr=0.001,
        max_train_length=None,
    )
    losses = model.fit(
        windows,
        n_epochs=10,
        verbose=False,
    )
    encode_args = {
        "causal": True,
        "sliding_length": 1,
        "sliding_padding": 30,
        "batch_size": 32,
    }
    train_masked = np.asarray(
        model.encode(
            train_norm[None, :, :],
            mask="mask_last",
            **encode_args,
        )
    ).reshape(-1, 64)
    train_plain = np.asarray(
        model.encode(train_norm[None, :, :], **encode_args)
    ).reshape(-1, 64)
    test_masked = np.asarray(
        model.encode(
            test_norm[None, :, :],
            mask="mask_last",
            **encode_args,
        )
    ).reshape(-1, 64)
    test_plain = np.asarray(
        model.encode(test_norm[None, :, :], **encode_args)
    ).reshape(-1, 64)
    train_original = np.abs(train_masked - train_plain).sum(axis=1)
    test_original = np.abs(test_masked - test_plain).sum(axis=1)
    device = torch.device(
        f"cuda:{device_id}" if isinstance(device_id, int) else "cpu"
    )
    train_cos, train_euc, test_cos, test_euc = bank_distances(
        train_plain,
        test_plain,
        device=device,
    )
    train_amp, test_amp = amplitude_scores(train, test)
    _, test_original_z = z_from_train(train_original, test_original)
    _, test_cos_z = z_from_train(train_cos, test_cos)
    train_euc_z, test_euc_z = z_from_train(train_euc, test_euc)
    train_g_z, test_g_z = z_from_train(train_amp[:, 0], test_amp[:, 0])
    train_q_z, test_q_z = z_from_train(train_amp[:, 1], test_amp[:, 1])
    train_pai = (
        train_euc_z + PAI_W_G * train_g_z + PAI_W_Q * train_q_z
    )
    test_pai = (
        test_euc_z + PAI_W_G * test_g_z + PAI_W_Q * test_q_z
    )
    variants = {
        "original": (train_original, test_original),
        "cos": (train_cos, test_cos),
        "euc": (train_euc, test_euc),
        "pai": (train_pai, test_pai),
    }
    metrics = {
        "machine": name,
        "seed": seed,
        "train_windows": int(len(windows)),
        "window": WINDOW,
    }
    arrays = {
        "label": label,
        "loss_first": np.asarray(losses[0]),
        "loss_last": np.asarray(losses[-1]),
    }
    for variant, (train_score, test_score) in variants.items():
        metric = evaluate_machine(
            train,
            test,
            label,
            train_score,
            test_score,
        )
        for key, value in metric.items():
            metrics[f"{variant}_{key}"] = float(value)
        arrays[f"{variant}_train_score"] = train_score.astype(np.float32)
        arrays[f"{variant}_test_score"] = test_score.astype(np.float32)
    metrics["elapsed_seconds"] = time.time() - start
    np.savez_compressed(
        output,
        **arrays,
        metrics=json.dumps(metrics),
    )
    print(
        f"{name}: T0m PR={metrics['pai_pr_auc']:.4f}, "
        f"time={metrics['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return metrics


def main() -> None:
    args = parse_args()
    selected = args.machines or machine_names()
    rows = [
        run_machine(name, args.seed, args.force)
        for name in selected
    ]
    frame = pd.DataFrame(rows)
    seed_root = OUT / f"seed_{args.seed}"
    frame.to_csv(
        seed_root / "results_by_machine.csv",
        index=False,
        encoding="utf-8",
    )
    metrics = [
        "roc_auc",
        "pr_auc",
        "recall_q99",
        "precision_q99",
        "f1_q99",
        "fpr_q99",
        "segment_recall_q99",
        "pa_f1_top5",
    ]
    summary = {
        "machines": int(len(frame)),
        "seed": args.seed,
        "window": WINDOW,
        "macro_mean": {
            variant: {
                metric: float(frame[f"{variant}_{metric}"].mean())
                for metric in metrics
            }
            for variant in ("original", "cos", "euc", "pai")
        },
    }
    (seed_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
