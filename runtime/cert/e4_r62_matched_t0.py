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
    / "r62_matched_t0"
)
PROCESSED = ROOT / "processed_r62"
DATASET = PROCESSED / "r6.2_sequence_dataset.npz"
INCIDENTS = PROCESSED / "r6.2_incidents.csv"
K_GRID = (5, 10, 20)
SEEDS = [20261116, 20261117, 20261118]

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
from train_r42_ts2vec import TS2Vec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
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
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = np.load(DATASET, allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    test_start = int(dataset["test_start_index"])
    rng = np.random.default_rng(2024)
    train_user_indices = np.sort(
        rng.choice(
            len(users),
            size=min(1000, len(users)),
            replace=False,
        )
    )
    train_values = values[
        train_user_indices,
        : train_end + 1,
    ]
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
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device_id = 0 if torch.cuda.is_available() else "cpu"
        device = torch.device(
            f"cuda:{device_id}" if isinstance(device_id, int) else "cpu"
        )
        model = TS2Vec(
            input_dims=values.shape[-1],
            output_dims=64,
            hidden_dims=64,
            depth=5,
            device=device_id,
            batch_size=16,
        )
        model.fit(
            train_values,
            n_epochs=10,
            verbose=False,
        )
        encode_args = {
            "causal": True,
            "sliding_length": 1,
            "sliding_padding": 30,
            "batch_size": 256,
        }
        masked = np.asarray(
            model.encode(values, mask="mask_last", **encode_args),
            dtype=np.float32,
        )
        plain = np.asarray(
            model.encode(values, **encode_args),
            dtype=np.float32,
        )
        original = np.abs(masked - plain).sum(axis=-1)
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
            "T0m_original": original_z,
            "T0m_bank_cosine": cosine_z,
            "T0m_bank_euclidean": euclidean_z,
            "T0m_pai_a1": pai,
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
            OUT / f"seed_{seed}_scores.npz",
            **variants,
        )
        seed_root = OUT / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        torch.save(model.net.state_dict(), seed_root / "model.pth")
        np.save(seed_root / "masked.npy", masked)
        np.save(seed_root / "plain.npy", plain)
        (seed_root / "summary.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "train_users": len(train_user_indices),
                    "train_user_indices": train_user_indices.tolist(),
                    "epochs": 10,
                    "batch_size": 16,
                    "bank_clusters": n_clusters,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"r6.2 matched T0 seed={seed} done", flush=True)
        del masked, plain
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "metrics_by_seed.csv",
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
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_std=("incident_hits", "std"),
        )
        .sort_values(["k", "method"])
    )
    summary.to_csv(
        OUT / "metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
