from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SOFTCLT_DIR = (
    ROOT.parents[1]
    / "references"
    / "softclt_repo"
    / "softclt_ts2vec"
)
sys.path.insert(0, str(SOFTCLT_DIR))
from soft_ts2vec import TS2Vec as SoftTS2Vec  # noqa: E402

sys.path.insert(0, str(ROOT))
from quota_relationship_sim import load_r42  # noqa: E402


OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e2"
)
SEEDS = [20261116, 20261117, 20261118]
VARIANTS = {
    "T0": "r42_T0_seed{seed}",
    "T1": "r42_T1_seed{seed}",
    "T2_tau1.5": "r42_T2_tau1.5_seed{seed}",
}
NEIGHBORS = 10
MECHANISM_SAMPLE = 2000
SAMPLE_SEED = 2024


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0)
    covariance = centered.T @ centered
    covariance /= max(len(features) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 1e-12:
        return 0.0
    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * np.log(probabilities)).sum()
    return float(np.exp(entropy))


def uniformity(features: np.ndarray) -> float:
    normalized = features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True),
        1e-8,
    )
    rng = np.random.default_rng(SAMPLE_SEED)
    left = rng.integers(0, len(normalized), size=100_000)
    right = rng.integers(0, len(normalized), size=100_000)
    distances = np.square(
        normalized[left] - normalized[right]
    ).sum(axis=1)
    return float(np.log(np.exp(-2.0 * distances).mean() + 1e-12))


def neighbor_metrics(
    data_features: np.ndarray,
    representation: np.ndarray,
) -> dict[str, float]:
    data_dense = pairwise_distances(
        data_features,
        metric="euclidean",
    )
    rep_dense = pairwise_distances(
        representation,
        metric="euclidean",
    )
    data_indices = np.argsort(data_dense, axis=1)[
        :, 1 : NEIGHBORS + 1
    ]
    rep_indices = np.argsort(rep_dense, axis=1)[
        :, 1 : NEIGHBORS + 1
    ]
    overlap = np.mean(
        [
            len(set(left) & set(right)) / NEIGHBORS
            for left, right in zip(data_indices, rep_indices)
        ]
    )
    upper = np.triu_indices(len(data_features), k=1)
    relation_rho = float(
        spearmanr(
            data_dense[upper],
            rep_dense[upper],
        ).statistic
    )
    return {
        "knn_overlap": float(overlap),
        "local_distance_spearman": relation_rho,
        "effective_rank": effective_rank(representation),
        "uniformity": uniformity(representation),
        "mean_norm": float(
            np.linalg.norm(representation, axis=1).mean()
        ),
    }


def load_model(state_path: Path) -> SoftTS2Vec:
    model = SoftTS2Vec(
        input_dims=27,
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device="cpu",
        batch_size=16,
        lambda_=0.5,
        tau_temp=1.5,
        soft_instance=False,
        soft_temporal=False,
    )
    state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=True,
    )
    model.net.load_state_dict(state)
    model.net.eval()
    return model


def main() -> None:
    dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    values = dataset["values"].astype(np.float32)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    data = load_r42()
    labels = data["labels"]
    validation_active = (
        active_mask[:, train_end + 1 : val_end + 1]
        & (labels[:, train_end + 1 : val_end + 1] == 0)
    )
    user_indices, relative_times = np.nonzero(validation_active)
    absolute_times = relative_times + train_end + 1
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = rng.choice(
        len(user_indices),
        size=min(MECHANISM_SAMPLE, len(user_indices)),
        replace=False,
    )
    sample_users = user_indices[chosen]
    sample_times = absolute_times[chosen]
    data_features = values[sample_users, sample_times].astype(
        np.float64
    )
    detail_path = OUT / "mechanism_metrics_by_seed.csv"
    if detail_path.exists():
        frame = pd.read_csv(detail_path)
        rows = frame.to_dict(orient="records")
    else:
        rows = []
    completed = {
        (int(row["seed"]), str(row["variant"]))
        for row in rows
    }
    for seed in SEEDS:
        for variant, pattern in VARIANTS.items():
            if (seed, variant) in completed:
                continue
            run_dir = OUT / "runs" / pattern.format(seed=seed)
            model = load_model(run_dir / "model.pth")
            plain = np.asarray(
                model.encode(
                    values,
                    casual=True,
                    sliding_length=1,
                    sliding_padding=30,
                    batch_size=32,
                ),
                dtype=np.float32,
            )
            representation = plain[
                sample_users, sample_times
            ].astype(np.float64)
            metrics = neighbor_metrics(
                data_features,
                representation,
            )
            row = {
                "seed": seed,
                "variant": variant,
                **metrics,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(
                detail_path,
                index=False,
                encoding="utf-8",
            )
            print(
                f"{variant} seed={seed}: "
                f"kNNO={metrics['knn_overlap']:.4f}, "
                f"rho={metrics['local_distance_spearman']:.4f}",
                flush=True,
            )
            del model, plain, representation
    frame = pd.DataFrame(rows)
    frame.to_csv(
        detail_path,
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
            local_distance_spearman_std=(
                "local_distance_spearman",
                "std",
            ),
            effective_rank_mean=("effective_rank", "mean"),
            uniformity_mean=("uniformity", "mean"),
            mean_norm_mean=("mean_norm", "mean"),
        )
        .sort_values("variant")
    )
    summary.to_csv(
        OUT / "mechanism_metrics_summary.csv",
        index=False,
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
