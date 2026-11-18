from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from e2_mechanism_audit import (  # noqa: E402
    MECHANISM_SAMPLE,
    SAMPLE_SEED,
    load_model,
    neighbor_metrics,
)
from quota_relationship_sim import load_r42  # noqa: E402


OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e2_instance"
)
SEEDS = [20261116, 20261117, 20261118]
VARIANTS = ["T3_instance_soft", "T4_full_soft"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance-tag",
        default="missing_safe_v2_20261117",
        help="Corrected model directory under e2_instance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = OUT / args.instance_tag
    dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    values = dataset["values"].astype(np.float32)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    labels = load_r42()["labels"]
    validation_normal = (
        active[:, train_end + 1 : val_end + 1]
        & (labels[:, train_end + 1 : val_end + 1] == 0)
    )
    users, relative_times = np.nonzero(validation_normal)
    times = relative_times + train_end + 1
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = rng.choice(
        len(users),
        size=min(MECHANISM_SAMPLE, len(users)),
        replace=False,
    )
    sample_users = users[chosen]
    sample_times = times[chosen]
    data_features = values[sample_users, sample_times].astype(
        np.float64
    )
    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            model = load_model(
                run_root / variant / f"seed_{seed}" / "model.pth"
            )
            plain = np.asarray(
                model.encode(
                    values,
                    casual=True,
                    sliding_length=1,
                    sliding_padding=30,
                    batch_size=256,
                )
            )
            representation = plain[
                sample_users, sample_times
            ].astype(np.float64)
            metrics = neighbor_metrics(
                data_features,
                representation,
            )
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    **metrics,
                }
            )
            print(
                f"{variant} seed={seed}: "
                f"kNNO={metrics['knn_overlap']:.4f}, "
                f"rho={metrics['local_distance_spearman']:.4f}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        run_root / "mechanism_metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        frame.groupby("variant", as_index=False)
        .agg(
            knn_overlap_mean=("knn_overlap", "mean"),
            knn_overlap_std=("knn_overlap", "std"),
            local_distance_spearman_mean=(
                "local_distance_spearman",
                "mean",
            ),
            effective_rank_mean=("effective_rank", "mean"),
            uniformity_mean=("uniformity", "mean"),
            mean_norm_mean=("mean_norm", "mean"),
        )
    )
    summary.to_csv(
        run_root / "mechanism_summary.csv",
        index=False,
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
