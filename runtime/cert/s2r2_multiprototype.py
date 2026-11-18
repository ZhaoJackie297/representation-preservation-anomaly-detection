from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e19_models"
SEEDS = [20261111, 20261112, 20261113]
PROTOTYPES = 3
COVARIANCE_SHRINKAGE = 0.15

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from e19_user_manifold import (  # noqa: E402
    UserMetricProjector,
    project_with_model,
)
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42,
    load_r52,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from v2_m_pipeline import add_utilities, evaluate_block  # noqa: E402


BLOCKS = {
    "r42_original": {
        "family": "r42",
        "plain": OUT / "e17_r42_M0_plain.npy",
        "masked": OUT / "e17_r42_M0_masked_quantile.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r42_original_channel_top2_quantile",
        ),
        "metric_prefix": "r42",
        "feature_file": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        ),
        "external": False,
    },
    "r42_retrained": {
        "family": "r42",
        "plain": OUT / "e19_r42_retrained_plain.npy",
        "masked": OUT / "e19_r42_retrained_masked_quantile.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r42_retrained_channel_top2_quantile",
        ),
        "metric_prefix": "r42_retrained",
        "feature_file": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        ),
        "external": False,
    },
    "r52_external": {
        "family": "r52",
        "plain": OUT / "e17_r52_M0_plain.npy",
        "masked": OUT / "e17_r52_M0_masked_quantile.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r52_external_channel_top2_quantile",
        ),
        "metric_prefix": "r52",
        "feature_file": (
            ROOT / "processed_r52" / "r5.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        ),
        "external": True,
    },
}


def load_metric_model(
    block_name: str,
    seed: int,
    input_dims: int,
    device: torch.device,
) -> UserMetricProjector:
    prefix = BLOCKS[block_name]["metric_prefix"]
    model = UserMetricProjector(input_dims, 24).to(device)
    model.load_state_dict(
        torch.load(
            MODEL_DIR / f"{prefix}_user_metric_{seed}.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    return model


def shrink_covariance(
    centered: np.ndarray,
    shrinkage: float = COVARIANCE_SHRINKAGE,
) -> np.ndarray:
    covariance = (
        centered.T @ centered / max(len(centered) - 1, 1)
    )
    average_variance = float(np.mean(np.diag(covariance)))
    return (
        (1.0 - shrinkage) * covariance
        + shrinkage
        * max(average_variance, 1e-6)
        * np.eye(covariance.shape[0], dtype=np.float64)
    )


def inverse_sqrt(covariance: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    return (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )


def multi_prototype_distances(
    representation: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    *,
    seed: int,
    prototypes: int = PROTOTYPES,
) -> tuple[np.ndarray, dict]:
    users, _, dimensions = representation.shape
    train_mask = (
        active_mask[:, : train_end + 1]
        & np.isfinite(
            representation[:, : train_end + 1]
        ).all(axis=-1)
    )
    distances = np.full(
        active_mask.shape, np.nan, dtype=np.float32
    )
    centers_all = np.full(
        (users, prototypes, dimensions),
        np.nan,
        dtype=np.float32,
    )
    cluster_counts = []
    for user in range(users):
        training = np.flatnonzero(train_mask[user])
        if len(training) < max(12, prototypes * 3):
            continue
        rows = representation[user, training].astype(np.float64)
        center = np.median(rows, axis=0)
        centered = rows - center
        covariance = shrink_covariance(centered)
        whitener = inverse_sqrt(covariance)
        whitened = centered @ whitener
        model = KMeans(
            n_clusters=prototypes,
            n_init=3,
            max_iter=100,
            random_state=seed + user,
            algorithm="lloyd",
        )
        labels = model.fit_predict(whitened)
        prototypes_whitened = model.cluster_centers_
        centers_all[user] = (
            prototypes_whitened @ whitener.T + center
        ).astype(np.float32)
        counts = np.bincount(labels, minlength=prototypes)
        cluster_counts.append(counts)

        active = np.flatnonzero(active_mask[user])
        if len(active) == 0:
            continue
        active_whitened = (
            representation[user, active] - center
        ).astype(np.float64) @ whitener
        pairwise = np.linalg.norm(
            active_whitened[:, None, :]
            - prototypes_whitened[None, :, :],
            axis=-1,
        )
        distances[user, active] = pairwise.min(axis=1).astype(
            np.float32
        )
    counts_array = (
        np.stack(cluster_counts, axis=0)
        if cluster_counts
        else np.zeros((0, prototypes), dtype=np.int64)
    )
    diagnostics = {
        "users_fitted": int(counts_array.shape[0]),
        "empty_cluster_users": int(
            np.any(counts_array == 0, axis=1).sum()
        )
        if len(counts_array)
        else 0,
        "mean_min_cluster_size": float(
            counts_array.min(axis=1).mean()
        )
        if len(counts_array)
        else 0.0,
        "mean_max_cluster_size": float(
            counts_array.max(axis=1).mean()
        )
        if len(counts_array)
        else 0.0,
    }
    return distances, {
        "diagnostics": diagnostics,
        "centers": centers_all,
    }


def build_signal(
    data: dict,
    masked_q: np.ndarray,
    manifold_q: np.ndarray,
    channel_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    triple = (masked_q + manifold_q + channel_q) / 3.0
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        triple,
        data["active_mask"],
    )
    return (
        daily_percentile(
            dynamic, data["active_mask"]
        ).astype(np.float32),
        dynamic,
    )


def run_block(block_name: str) -> None:
    config = BLOCKS[block_name]
    data = load_r52() if config["family"] == "r52" else load_r42()
    dataset = np.load(config["sequence"], allow_pickle=False)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    plain = np.load(config["plain"], mmap_mode="r")
    masked_q = np.load(config["masked"])
    channel_store = np.load(config["channel"][0], allow_pickle=False)
    channel_q = channel_store[config["channel"][1]].astype(np.float32)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    single_quantiles = []
    prototype_quantiles = []
    prototype_centers = {}
    diagnostics = []
    for seed in SEEDS:
        print(
            f"{block_name}: metric seed {seed}",
            flush=True,
        )
        model = load_metric_model(
            block_name, seed, plain.shape[-1], device
        )
        representation = project_with_model(model, plain)
        single_q = np.load(
            OUT
            / (
                f"e19_{config['metric_prefix']}_learned_manifold_"
                f"quantile_{seed}.npy"
            )
        )
        prototype_raw, metadata = multi_prototype_distances(
            representation,
            active_mask,
            train_end,
            seed=seed,
        )
        prototype_q = per_user_quantile_scores(
            prototype_raw, active_mask, train_end
        )
        single_quantiles.append(single_q)
        prototype_quantiles.append(prototype_q)
        prototype_centers[str(seed)] = metadata["centers"]
        diagnostics.append(
            {
                "seed": seed,
                **metadata["diagnostics"],
            }
        )
        del representation, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    single_ensemble_q = np.mean(
        np.stack(single_quantiles, axis=0),
        axis=0,
    ).astype(np.float32)
    prototype_ensemble_q = np.mean(
        np.stack(prototype_quantiles, axis=0),
        axis=0,
    ).astype(np.float32)
    single_signal, single_dynamic = build_signal(
        data, masked_q, single_ensemble_q, channel_q
    )
    prototype_signal, prototype_dynamic = build_signal(
        data, masked_q, prototype_ensemble_q, channel_q
    )
    single_original, single_augmented = fit_view_scores(
        data,
        config["feature_file"],
        single_dynamic,
        data["personal"],
        20261109,
    )
    prototype_original, prototype_augmented = fit_view_scores(
        data,
        config["feature_file"],
        prototype_dynamic,
        data["personal"],
        20261109,
    )
    single_auxiliary = np.maximum(
        single_original, single_augmented
    )
    prototype_auxiliary = np.maximum(
        prototype_original, prototype_augmented
    )
    frames = [
        evaluate_block(
            block_name,
            data,
            single_signal,
            single_auxiliary,
            "S2R2_single",
            external=bool(config["external"]),
        ),
        evaluate_block(
            block_name,
            data,
            prototype_signal,
            prototype_auxiliary,
            "S2R2_multiprototype",
            external=bool(config["external"]),
        ),
    ]
    metrics = add_utilities(
        pd.concat(frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / f"s2r2_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / f"s2r2_{block_name}_components.npz",
        single_signal=single_signal,
        prototype_signal=prototype_signal,
        single_manifold_quantile=single_ensemble_q,
        prototype_manifold_quantile=prototype_ensemble_q,
        single_aligned_iforest=single_auxiliary,
        prototype_aligned_iforest=prototype_auxiliary,
        **{
            f"prototype_centers_seed_{seed}": centers
            for seed, centers in prototype_centers.items()
        },
    )
    summary = {
        "block": block_name,
        "prototypes": PROTOTYPES,
        "seeds": SEEDS,
        "diagnostics": diagnostics,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s2r2_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
                "diagnostics": diagnostics,
                "metrics_k10": metrics[
                    metrics["k"] == 10
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices=sorted(BLOCKS),
        default="r42_original",
    )
    args = parser.parse_args()
    run_block(args.block)


if __name__ == "__main__":
    main()
