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
    / "r52"
)
OUT.mkdir(parents=True, exist_ok=True)
INSTANCE_ROOT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e2_instance"
    / "r52"
    / "T4_full_soft"
)
BASE_E5 = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e5"
    / "r52"
)
SEEDS = [20261116, 20261117, 20261118]
K_GRID = (5, 10, 20)

sys.path.insert(0, str(ROOT))
from e0_e1_representation_preservation import (  # noqa: E402
    align_raw_features,
    global_train_z,
)
from e3_multichannel_amplitude import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    aggregate_components,
    build_alerts,
    point_metrics,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r52,
)
from train_r42_ts2vec import TS2Vec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance-tag",
        default="missing_safe_v2_20261117",
        help="Run subdirectory under e2_instance/r52/T4_full_soft.",
    )
    parser.add_argument(
        "--output-tag",
        default="missing_safe_v2_20261117",
        help="Subdirectory under e4_joint/r52 for corrected results.",
    )
    return parser.parse_args()


def load_model(path: Path):
    device_id = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(
        f"cuda:{device_id}" if isinstance(device_id, int) else "cpu"
    )
    model = TS2Vec(
        input_dims=27,
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device_id,
        batch_size=16,
    )
    state = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if key != "n_averaged"
    }
    model.net.module.load_state_dict(state)
    model.net.eval()
    return model, device


def bank_scores(
    train_repr: np.ndarray,
    all_repr: np.ndarray,
    active: np.ndarray,
    train_end: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import MiniBatchKMeans

    train = train_repr[:, : train_end + 1][
        active[:, : train_end + 1]
    ]
    clusters = min(500, max(1, int(round(0.1 * len(train)))))
    clusters = min(clusters, len(train) - 1)
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        init="k-means++",
        random_state=2024,
        batch_size=max(8192, clusters),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    ).fit(train)
    bank = torch.from_numpy(
        kmeans.cluster_centers_.astype(np.float32)
    ).to(device)
    bank_norm = torch.nn.functional.normalize(bank, dim=1)
    users, times = np.nonzero(active)
    query_values = all_repr[users, times]
    cos_values = []
    euc_values = []
    with torch.inference_mode():
        for start in range(0, len(query_values), 4096):
            query = torch.from_numpy(
                query_values[start : start + 4096].astype(np.float32)
            ).to(device)
            query_norm = torch.nn.functional.normalize(query, dim=1)
            cosine = (
                1.0 - query_norm @ bank_norm.T
            ).clamp_min_(0.0)
            euclidean = torch.cdist(query, bank)
            cos_values.append(
                cosine.topk(3, dim=1, largest=False)
                .values.mean(dim=1)
                .cpu()
                .numpy()
            )
            euc_values.append(
                euclidean.topk(3, dim=1, largest=False)
                .values.mean(dim=1)
                .cpu()
                .numpy()
            )
    cosine = np.full(active.shape, np.nan, dtype=np.float32)
    euclidean = np.full(active.shape, np.nan, dtype=np.float32)
    cosine[users, times] = np.concatenate(cos_values)
    euclidean[users, times] = np.concatenate(euc_values)
    return cosine, euclidean


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
    end = len(data["dates"]) - 1
    point = point_metrics(
        signal,
        data,
        active,
        test_start,
        end,
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
    instance_root = INSTANCE_ROOT / args.instance_tag
    out_dir = OUT / args.output_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = np.load(
        ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    test_start = int(dataset["test_start_index"])
    data = load_r52()
    raw = align_raw_features(
        ROOT / "processed_r52" / "r5.2_user_day_features.csv",
        users,
        dates,
    )
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
    for seed in SEEDS:
        seed_root = instance_root / f"seed_{seed}"
        masked_path = seed_root / "masked.npy"
        plain_path = seed_root / "plain.npy"
        if masked_path.exists() and plain_path.exists():
            masked = np.load(masked_path, allow_pickle=False)
            plain = np.load(plain_path, allow_pickle=False)
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            model, device = load_model(seed_root / "model.pth")
            encode_args = {
                "causal": True,
                "sliding_length": 1,
                "sliding_padding": 30,
                "batch_size": 256,
            }
            masked = np.asarray(
                model.encode(
                    values,
                    mask="mask_last",
                    **encode_args,
                ),
                dtype=np.float32,
            )
            plain = np.asarray(
                model.encode(values, **encode_args),
                dtype=np.float32,
            )
        original = np.abs(masked - plain).sum(axis=-1)
        cosine, euclidean = bank_scores(
            plain,
            plain,
            active,
            train_end,
            device,
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
            out_dir / f"seed_{seed}_scores.npz",
            **variants,
        )
        print(f"r5.2 E4 seed={seed} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        out_dir / "metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        frame.groupby(["method", "k"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            precision_at_k_mean=("precision_at_k", "mean"),
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_std=("incident_hits", "std"),
        )
        .sort_values(["k", "method"])
    )
    summary.to_csv(
        out_dir / "metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
