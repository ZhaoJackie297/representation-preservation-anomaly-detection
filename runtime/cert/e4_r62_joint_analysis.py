from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e4_joint"
    / "r62"
)
INSTANCE_ROOT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e2_instance"
    / "r62"
    / "T4_full_soft"
)
PROCESSED = ROOT / "processed_r62"
DATASET = PROCESSED / "r6.2_sequence_dataset.npz"
LABELS = PROCESSED / "r6.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r6.2_incidents.csv"
RAW = PROCESSED / "r6.2_user_day_features.parquet"
SEEDS = [20261116, 20261117, 20261118]
K_GRID = (5, 10, 20)

sys.path.insert(0, str(ROOT))
from e0_e1_representation_preservation import (  # noqa: E402
    bank_scores,
    global_train_z,
)
from e3_multichannel_amplitude import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    aggregate_components,
    build_alerts,
)
from e6_r62_audit import (  # noqa: E402
    align_raw,
    label_matrix,
    point_metrics,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
    )
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument(
        "--instance-root",
        type=Path,
        default=INSTANCE_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
    )
    return parser.parse_args()


def evaluate(
    name: str,
    signal: np.ndarray,
    data: dict,
    active: np.ndarray,
    test_start: int,
    seed: int,
) -> list[dict]:
    rank = signal.astype(np.float32, copy=True)
    rank[~np.isfinite(rank)] = -np.inf
    point = point_metrics(
        signal,
        data["labels"],
        active,
        test_start,
    )
    rows = []
    for k in K_GRID:
        alerts = build_alerts(rank, active, k)
        result = evaluate_alert_system(
            data,
            alerts,
            name,
            k,
            external=True,
        )
        rows.append(
            {
                "seed": seed,
                "method": name,
                "k": k,
                **compact_metric(result, None),
                **point,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = np.load(DATASET, allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    test_start = int(dataset["test_start_index"])
    labels = label_matrix(users, dates)
    incidents = pd.read_csv(
        INCIDENTS,
        parse_dates=[
            "start",
            "end",
            "first_observed_date",
            "last_observed_date",
        ],
    )
    data = {
        "users": users,
        "dates": dates,
        "active_mask": active,
        "labels": labels,
        "incidents": incidents,
        "test_start": test_start,
    }

    raw = align_raw(users, dates)
    g_a1, q_a1 = aggregate_components(
        raw,
        active,
        train_end,
        user_specific=True,
        aggregation="channel_mean",
    )
    g_a1_z = global_train_z(g_a1, active, train_end)
    q_a1_z = global_train_z(q_a1, active, train_end)

    rows = []
    completed = []
    for seed in args.seeds:
        seed_root = args.instance_root / f"seed_{seed}"
        masked_path = seed_root / "masked.npy"
        plain_path = seed_root / "plain.npy"
        if not masked_path.exists() or not plain_path.exists():
            print(f"skip seed={seed}: missing saved encodings", flush=True)
            continue
        print(
            f"r6.2 E4 seed={seed}: loading saved encodings",
            flush=True,
        )
        masked = np.load(masked_path, allow_pickle=False)
        plain = np.load(plain_path, allow_pickle=False)
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        original = np.abs(masked - plain).sum(axis=-1)
        print(f"r6.2 E4 seed={seed}: representation bank", flush=True)
        cosine, euclidean, n_clusters = bank_scores(
            plain,
            active,
            train_end,
            device=device,
        )
        original_z = global_train_z(original, active, train_end)
        cosine_z = global_train_z(cosine, active, train_end)
        euclidean_z = global_train_z(euclidean, active, train_end)
        pai = (
            euclidean_z
            + PAI_W_G * g_a1_z
            + PAI_W_Q * q_a1_z
        )
        variants = {
            "T4_original": original_z,
            "T4_bank_cosine": cosine_z,
            "T4_bank_euclidean": euclidean_z,
            "T4_pai_a1": pai,
        }
        for name, signal in variants.items():
            rows.extend(
                evaluate(
                    name,
                    signal,
                    data,
                    active,
                    test_start,
                    seed,
                )
            )
        np.savez_compressed(
            args.output_dir / f"seed_{seed}_scores.npz",
            **variants,
        )
        completed.append(
            {
                "seed": seed,
                "bank_clusters": n_clusters,
                "encode_batch_size": args.encode_batch_size,
            }
        )
        print(f"r6.2 E4 seed={seed}: done", flush=True)
        del masked, plain
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No completed r6.2 joint model was found.")
    frame = pd.DataFrame(rows)
    frame.to_csv(
        args.output_dir / "metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        frame.groupby(["method", "k"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            precision_at_k_mean=("precision_at_k", "mean"),
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_std=("incident_hits", "std"),
        )
        .sort_values(["k", "method"])
    )
    summary.to_csv(
        args.output_dir / "metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(
            {
                "dataset": "r6.2",
                "events": len(incidents),
                "train_end": train_end,
                "test_start": test_start,
                "completed_models": completed,
                "labels_used_in_training": False,
                "labels_used_in_calibration": False,
                "test_used_for_model_selection": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
