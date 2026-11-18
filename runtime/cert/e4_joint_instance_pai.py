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
)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from e0_e1_representation_preservation import (  # noqa: E402
    align_raw_features,
    global_train_z,
)
from e3_multichannel_amplitude import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    aggregate_components,
    evaluate_variant,
)
from quota_relationship_sim import load_r42  # noqa: E402
from train_r42_ts2vec import TS2Vec  # noqa: E402


SEEDS = [20261116, 20261117, 20261118]
VARIANTS = ["T3_instance_soft", "T4_full_soft"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
    )
    parser.add_argument(
        "--instance-tag",
        default="missing_safe_v2_20261117",
        help="Corrected e2_instance output directory.",
    )
    parser.add_argument(
        "--output-tag",
        default="missing_safe_v2_20261117",
        help="Subdirectory under e4_joint for corrected results.",
    )
    return parser.parse_args()


def load_soft_model(path: Path):
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
    active_users, active_times = np.nonzero(active)
    queries = all_repr[active_users, active_times]
    cosine_values = []
    euclidean_values = []
    with torch.inference_mode():
        for start in range(0, len(queries), 4096):
            query = torch.from_numpy(
                queries[start : start + 4096].astype(np.float32)
            ).to(device)
            query_norm = torch.nn.functional.normalize(query, dim=1)
            cosine = (
                1.0 - query_norm @ bank_norm.T
            ).clamp_min_(0.0)
            euclidean = torch.cdist(query, bank)
            cosine_values.append(
                cosine.topk(3, dim=1, largest=False)
                .values.mean(dim=1)
                .cpu()
                .numpy()
            )
            euclidean_values.append(
                euclidean.topk(3, dim=1, largest=False)
                .values.mean(dim=1)
                .cpu()
                .numpy()
            )
    cosine_score = np.full(active.shape, np.nan, dtype=np.float32)
    euclidean_score = np.full(active.shape, np.nan, dtype=np.float32)
    cosine_score[active_users, active_times] = np.concatenate(
        cosine_values
    )
    euclidean_score[active_users, active_times] = np.concatenate(
        euclidean_values
    )
    return cosine_score, euclidean_score


def main() -> None:
    args = parse_args()
    out_dir = OUT / args.output_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    data = load_r42()
    raw = align_raw_features(
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
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
    for variant in args.variants:
        for seed in SEEDS:
            model, device = load_soft_model(
                ROOT
                / "experiments"
                / "representation_preservation"
                / "e2_instance"
                / args.instance_tag
                / variant
                / f"seed_{seed}"
                / "model.pth"
            )
            args_encode = {
                "causal": True,
                "sliding_length": 1,
                "sliding_padding": 30,
                "batch_size": 256,
            }
            masked = np.asarray(
                model.encode(values, mask="mask_last", **args_encode),
                dtype=np.float32,
            )
            plain = np.asarray(
                model.encode(values, **args_encode),
                dtype=np.float32,
            )
            original = np.abs(masked - plain).sum(axis=-1)
            cos_score, euc_score = bank_scores(
                plain,
                plain,
                active,
                train_end,
                device,
            )
            original_z = global_train_z(original, active, train_end)
            cos_z = global_train_z(cos_score, active, train_end)
            euc_z = global_train_z(euc_score, active, train_end)
            pai = euc_z + PAI_W_G * g_a1_z + PAI_W_Q * q_a1_z
            scores = {
                "original": original_z,
                "cosine": cos_z,
                "euclidean": euc_z,
                "pai_a1": pai,
            }
            for score_name, score in scores.items():
                method = f"{variant}_{score_name}"
                rows.extend(
                    evaluate_variant(
                        method,
                        score,
                        data,
                        active,
                        train_end,
                        val_end,
                        seed,
                    )
                )
            np.savez_compressed(
                out_dir / f"{variant}_seed_{seed}_scores.npz",
                **scores,
            )
            print(f"{variant} seed={seed} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        out_dir / "metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        frame.groupby(["method", "split", "k"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            precision_at_k_mean=("precision_at_k", "mean"),
            incident_hits_mean=("incident_hits", "mean"),
        )
        .sort_values(["split", "k", "method"])
    )
    summary.to_csv(
        out_dir / "metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
