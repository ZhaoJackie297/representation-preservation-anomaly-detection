from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e19_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261111
PCA_DIMS = 16
METRIC_DIMS = 24
COVARIANCE_SHRINKAGE = 0.15
BATCH_USERS = 32
STEPS_PER_EPOCH = 200
EPOCHS = 2
LEARNING_RATE = 1e-3
TRIPLET_MARGIN = 0.2
COMPACT_WEIGHT = 0.1
VARIANCE_WEIGHT = 0.05
COVARIANCE_WEIGHT = 0.01
K = 10
Q_LIMIT = 4

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from e17_multigranular_mask import evaluate_signal  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42,
    load_r52,
)
from risk_topk_r42 import compute_risk  # noqa: E402


class UserMetricProjector(nn.Module):
    def __init__(self, input_dims: int, output_dims: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, output_dims),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def dataset_config(dataset_name: str) -> dict:
    if dataset_name == "r42":
        return {
            "dataset_file": (
                ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
            ),
            "data": load_r42(),
            "external": False,
            "auxiliary_key": (
                "r42_original_v2_aligned_dual_view_iforest"
            ),
        }
    return {
        "dataset_file": (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        ),
        "data": load_r52(),
        "external": True,
        "auxiliary_key": (
            "r52_external_v2_aligned_dual_view_iforest"
        ),
    }


def load_components(
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix = OUT / f"e17_{dataset_name}_M0"
    plain = np.load(f"{prefix}_plain.npy", mmap_mode="r")
    masked_q = np.load(
        f"{prefix}_masked_quantile.npy", mmap_mode="r"
    )
    latent_q = np.load(
        f"{prefix}_latent_quantile.npy", mmap_mode="r"
    )
    channel_q = np.load(
        f"{prefix}_channel_top2_quantile.npy", mmap_mode="r"
    )
    return plain, masked_q, latent_q, channel_q


def fit_global_pca(
    representations: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = (
        active_mask[:, : train_end + 1]
        & np.isfinite(
            representations[:, : train_end + 1]
        ).all(axis=-1)
    )
    rows = np.asarray(
        representations[:, : train_end + 1][train_mask],
        dtype=np.float32,
    )
    if len(rows) <= dimensions:
        raise RuntimeError("Not enough training rows for PCA.")
    mean = rows.mean(axis=0)
    centered = rows - mean
    covariance = centered.T @ centered / max(len(rows) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:dimensions]
    components = eigenvectors[:, order].astype(np.float32)
    explained = eigenvalues[order].astype(np.float32)
    return (
        mean.astype(np.float32),
        components,
        explained,
    )


def project_in_chunks(
    representations: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    *,
    users_per_chunk: int = 128,
) -> np.ndarray:
    users, time_steps, feature_dims = representations.shape
    output = np.empty(
        (users, time_steps, components.shape[1]),
        dtype=np.float32,
    )
    for start in range(0, users, users_per_chunk):
        end = min(start + users_per_chunk, users)
        block = np.asarray(representations[start:end])
        output[start:end] = (
            (block - mean) @ components
        ).astype(np.float32)
    return output


def eligible_users(
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    train_counts = active_mask[:, : train_end + 1].sum(axis=1)
    return np.flatnonzero(train_counts >= 4)


def train_user_metric(
    representations: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    output_path: Path,
    *,
    seed: int = SEED,
) -> tuple[UserMetricProjector, list[dict[str, float]]]:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    feature_dims = representations.shape[-1]
    model = UserMetricProjector(
        feature_dims, METRIC_DIMS
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    users = eligible_users(active_mask, train_end)
    train_days = {
        int(user): np.flatnonzero(
            active_mask[user, : train_end + 1]
            & np.isfinite(representations[user, : train_end + 1]).all(
                axis=-1
            )
        )
        for user in users
    }
    all_days = [
        np.asarray(representations[user, days], dtype=np.float32)
        for user, days in train_days.items()
        if len(days) >= 2
    ]
    usable_users = [
        user
        for user in users
        if len(train_days[int(user)]) >= 2
    ]
    if len(usable_users) < BATCH_USERS:
        raise RuntimeError("Not enough users for metric training.")
    print(
        f"trained users={len(usable_users)}, "
        f"training rows={sum(len(rows) for rows in all_days)}",
        flush=True,
    )

    loss_log: list[dict[str, float]] = []
    for epoch in range(EPOCHS):
        sums = {
            "total": 0.0,
            "triplet": 0.0,
            "compact": 0.0,
            "variance": 0.0,
            "covariance": 0.0,
        }
        for _ in range(STEPS_PER_EPOCH):
            sampled_users = np.random.choice(
                usable_users,
                size=BATCH_USERS,
                replace=False,
            )
            positives_a = []
            positives_b = []
            for user in sampled_users:
                days = train_days[int(user)]
                selected = np.random.choice(
                    days, size=2, replace=False
                )
                positives_a.append(
                    representations[int(user), selected[0]]
                )
                positives_b.append(
                    representations[int(user), selected[1]]
                )
            anchor_raw = torch.from_numpy(
                np.stack(positives_a).astype(np.float32)
            ).to(device)
            positive_raw = torch.from_numpy(
                np.stack(positives_b).astype(np.float32)
            ).to(device)
            anchor = model(anchor_raw)
            positive = model(positive_raw)

            distance_positive = torch.linalg.vector_norm(
                anchor - positive, dim=-1
            )
            pairwise = torch.linalg.vector_norm(
                anchor[:, None, :] - positive[None, :, :],
                dim=-1,
            )
            off_diagonal = ~torch.eye(
                BATCH_USERS, dtype=torch.bool, device=device
            )
            hardest_negative = pairwise.masked_fill(
                ~off_diagonal, -1e6
            ).max(dim=1).values
            triplet = F.softplus(
                TRIPLET_MARGIN
                + distance_positive
                - hardest_negative
            ).mean()
            compact = distance_positive.square().mean()

            combined = torch.cat(
                [anchor, positive], dim=0
            )
            centered = combined - combined.mean(dim=0)
            std = torch.sqrt(
                centered.var(dim=0, unbiased=False) + 1e-6
            )
            variance = F.relu(1.0 - std).mean()
            covariance_matrix = (
                centered.T @ centered
                / max(combined.shape[0] - 1, 1)
            )
            off_diagonal_dims = ~torch.eye(
                METRIC_DIMS, dtype=torch.bool, device=device
            )
            covariance = covariance_matrix[
                off_diagonal_dims
            ].square().mean()
            total = (
                triplet
                + COMPACT_WEIGHT * compact
                + VARIANCE_WEIGHT * variance
                + COVARIANCE_WEIGHT * covariance
            )
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            for key, value in (
                ("total", total),
                ("triplet", triplet),
                ("compact", compact),
                ("variance", variance),
                ("covariance", covariance),
            ):
                sums[key] += float(value.detach())
        row = {
            key: value / STEPS_PER_EPOCH
            for key, value in sums.items()
        }
        row["epoch"] = float(epoch + 1)
        loss_log.append(row)
        print(
            f"  epoch {epoch + 1}: "
            f"total={row['total']:.6f}, "
            f"triplet={row['triplet']:.6f}, "
            f"compact={row['compact']:.6f}, "
            f"var={row['variance']:.6f}, "
            f"cov={row['covariance']:.6f}",
            flush=True,
        )
    torch.save(model.state_dict(), output_path)
    model.eval()
    return model, loss_log


def project_with_model(
    model: UserMetricProjector,
    representations: np.ndarray,
    *,
    users_per_chunk: int = 128,
) -> np.ndarray:
    device = next(model.parameters()).device
    users, time_steps, feature_dims = representations.shape
    output = np.empty(
        (users, time_steps, METRIC_DIMS), dtype=np.float32
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, users, users_per_chunk):
            end = min(start + users_per_chunk, users)
            block = torch.from_numpy(
                np.asarray(representations[start:end])
            ).to(device)
            projected = model(block.reshape(-1, feature_dims))
            output[start:end] = (
                projected.reshape(end - start, time_steps, -1)
                .cpu()
                .numpy()
            )
    return output


def fit_user_manifolds(
    representations: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    *,
    shrinkage: float = COVARIANCE_SHRINKAGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    users, time_steps, dimensions = representations.shape
    train_mask = (
        active_mask[:, : train_end + 1]
        & np.isfinite(
            representations[:, : train_end + 1]
        ).all(axis=-1)
    )
    global_rows = representations[:, : train_end + 1][train_mask]
    global_center = np.median(global_rows, axis=0)
    global_centered = global_rows - global_center
    global_covariance = (
        global_centered.T @ global_centered
        / max(len(global_rows) - 1, 1)
    )
    global_variance = float(
        np.mean(np.diag(global_covariance))
    )
    global_covariance = (
        (1.0 - shrinkage) * global_covariance
        + shrinkage
        * max(global_variance, 1e-6)
        * np.eye(dimensions, dtype=np.float64)
    )
    global_inverse = np.linalg.pinv(
        global_covariance, hermitian=True
    )

    centers = np.empty((users, dimensions), dtype=np.float32)
    inverse_covariances = np.empty(
        (users, dimensions, dimensions), dtype=np.float32
    )
    distances = np.full(
        active_mask.shape, np.nan, dtype=np.float32
    )
    fallback_count = 0
    for user in range(users):
        training = np.flatnonzero(train_mask[user])
        if len(training) >= 4:
            center = np.median(
                representations[user, training], axis=0
            )
            centered = (
                representations[user, training] - center
            )
            covariance = (
                centered.T @ centered / max(len(training) - 1, 1)
            )
            average_variance = float(
                np.mean(np.diag(covariance))
            )
            covariance = (
                (1.0 - shrinkage) * covariance
                + shrinkage
                * max(average_variance, 1e-6)
                * np.eye(dimensions, dtype=np.float64)
            )
            inverse_covariance = np.linalg.pinv(
                covariance, hermitian=True
            )
        else:
            center = global_center
            inverse_covariance = global_inverse
            fallback_count += 1
        centers[user] = center.astype(np.float32)
        inverse_covariances[user] = (
            inverse_covariance.astype(np.float32)
        )

        active = np.flatnonzero(active_mask[user])
        if len(active) == 0:
            continue
        delta = (
            representations[user, active]
            - centers[user]
        ).astype(np.float64)
        squared = np.einsum(
            "ij,jk,ik->i",
            delta,
            inverse_covariance,
            delta,
        )
        distances[user, active] = np.sqrt(
            np.maximum(squared, 0.0)
        ).astype(np.float32)
    diagnostics = {
        "fallback_users": fallback_count,
        "global_center_norm": float(
            np.linalg.norm(global_center)
        ),
        "mean_inverse_covariance_trace": float(
            np.trace(inverse_covariances, axis1=1, axis2=2).mean()
        ),
    }
    return centers, inverse_covariances, distances, diagnostics


def build_signal(
    data: dict,
    masked_q: np.ndarray,
    manifold_q: np.ndarray,
    channel_q: np.ndarray,
) -> np.ndarray:
    triple = (masked_q + manifold_q + channel_q) / 3.0
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        triple,
        data["active_mask"],
    )
    return daily_percentile(
        dynamic, data["active_mask"]
    ).astype(np.float32)


def evaluate_method(
    data: dict,
    signal: np.ndarray,
    auxiliary: np.ndarray,
    method: str,
    *,
    external: bool,
) -> dict:
    return {
        "complete": evaluate_signal(
            data,
            signal,
            auxiliary,
            f"{method}_complete",
            external=external,
            use_iforest=False,
        ),
        "iforest": evaluate_signal(
            data,
            signal,
            auxiliary,
            f"{method}_iforest",
            external=external,
            use_iforest=True,
        ),
    }


def run_dataset(dataset_name: str) -> None:
    config = dataset_config(dataset_name)
    data = config["data"]
    dataset = np.load(
        config["dataset_file"], allow_pickle=False
    )
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    plain, masked_q, latent_q, channel_q = load_components(
        dataset_name
    )
    auxiliary_all = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    auxiliary = auxiliary_all[config["auxiliary_key"]]

    print("Building PCA manifold", flush=True)
    pca_mean, pca_components, pca_explained = fit_global_pca(
        plain,
        active_mask,
        train_end,
        PCA_DIMS,
    )
    pca_representation = project_in_chunks(
        plain, pca_mean, pca_components
    )
    (
        pca_centers,
        pca_inverse,
        pca_raw,
        pca_diagnostics,
    ) = fit_user_manifolds(
        pca_representation,
        active_mask,
        train_end,
    )
    pca_q = per_user_quantile_scores(
        pca_raw, active_mask, train_end
    )

    print("Training learned user metric", flush=True)
    metric_path = MODEL_DIR / f"{dataset_name}_user_metric.pth"
    metric_model, metric_losses = train_user_metric(
        plain,
        active_mask,
        train_end,
        metric_path,
    )
    metric_representation = project_with_model(
        metric_model, plain
    )
    (
        metric_centers,
        metric_inverse,
        metric_raw,
        metric_diagnostics,
    ) = fit_user_manifolds(
        metric_representation,
        active_mask,
        train_end,
    )
    metric_q = per_user_quantile_scores(
        metric_raw, active_mask, train_end
    )

    methods = {
        "A0_latent_control": build_signal(
            data, masked_q, latent_q, channel_q
        ),
        "A1_pca_manifold": build_signal(
            data, masked_q, pca_q, channel_q
        ),
        "A2_learned_manifold": build_signal(
            data, masked_q, metric_q, channel_q
        ),
    }
    results = {}
    for method, signal in methods.items():
        results[method] = evaluate_method(
            data,
            signal,
            auxiliary,
            f"E19_{method}",
            external=config["external"],
        )
        np.save(
            OUT / f"e19_{dataset_name}_{method}_signal.npy",
            signal,
        )
    np.save(
        OUT / f"e19_{dataset_name}_pca_manifold_raw.npy",
        pca_raw,
    )
    np.save(
        OUT / f"e19_{dataset_name}_pca_manifold_quantile.npy",
        pca_q,
    )
    np.save(
        OUT / f"e19_{dataset_name}_learned_manifold_raw.npy",
        metric_raw,
    )
    np.save(
        OUT / f"e19_{dataset_name}_learned_manifold_quantile.npy",
        metric_q,
    )
    np.savez_compressed(
        OUT / f"e19_{dataset_name}_pca_manifold_params.npz",
        mean=pca_mean,
        components=pca_components,
        explained=pca_explained,
        centers=pca_centers,
        inverse_covariances=pca_inverse,
    )
    np.savez_compressed(
        OUT / f"e19_{dataset_name}_learned_manifold_params.npz",
        centers=metric_centers,
        inverse_covariances=metric_inverse,
    )

    summary = {
        "dataset": dataset_name,
        "protocol": {
            "pca_dims": PCA_DIMS,
            "metric_dims": METRIC_DIMS,
            "covariance_shrinkage": COVARIANCE_SHRINKAGE,
            "batch_users": BATCH_USERS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "triplet_margin": TRIPLET_MARGIN,
            "compact_weight": COMPACT_WEIGHT,
            "variance_weight": VARIANCE_WEIGHT,
            "covariance_weight": COVARIANCE_WEIGHT,
            "anomaly_labels_used_for_training": False,
        },
        "pca_explained_variance": pca_explained.tolist(),
        "pca_diagnostics": pca_diagnostics,
        "learned_diagnostics": metric_diagnostics,
        "metric_losses": metric_losses,
        "results": results,
    }
    (OUT / f"e19_{dataset_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["r42", "r52"], default="r42"
    )
    args = parser.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
