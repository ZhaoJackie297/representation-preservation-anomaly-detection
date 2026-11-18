from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def profile_matrix(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    profiles = []
    active_fractions = []
    for user in range(values.shape[0]):
        training_active = active_mask[user, : train_end + 1]
        training_values = values[user, : train_end + 1][training_active]
        if training_values.size == 0:
            raise ValueError(f"user {user} has no active training days")
        median = np.nanmedian(training_values, axis=0)
        q90 = np.nanquantile(training_values, 0.9, axis=0)
        profiles.append(np.concatenate([median, q90]))
        active_fractions.append(float(training_active.mean()))
    profile = np.stack(profiles)
    active_fractions_array = np.asarray(
        active_fractions, dtype=np.float64
    )
    profile = np.concatenate(
        [profile, active_fractions_array[:, None]], axis=1
    )
    population_median = np.nanmedian(profile, axis=0)
    population_median = np.where(
        np.isfinite(population_median), population_median, 0.0
    )
    missing_rows, missing_columns = np.nonzero(~np.isfinite(profile))
    profile[missing_rows, missing_columns] = population_median[
        missing_columns
    ]
    center = np.median(profile, axis=0)
    q25 = np.quantile(profile, 0.25, axis=0)
    q75 = np.quantile(profile, 0.75, axis=0)
    scale = q75 - q25
    std = np.std(profile, axis=0)
    scale = np.where(scale < 1e-8, std, scale)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (profile - center) / scale
    return standardized, active_fractions_array


def gaussian_prior(
    profiles: np.ndarray,
    *,
    probability_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    squared_norm = np.sum(profiles * profiles, axis=1)
    distance_squared = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * profiles @ profiles.T
    )
    distances = np.sqrt(np.clip(distance_squared, 0.0, None))
    upper = distances[np.triu_indices(len(distances), k=1)]
    bandwidth = float(np.median(upper))
    if bandwidth < 1e-12:
        bandwidth = 1.0
    prior = np.exp(-(distances * distances) / (2.0 * bandwidth**2))
    prior = np.maximum(prior, probability_floor)
    np.fill_diagonal(prior, 0.0)
    prior /= prior.sum(axis=1, keepdims=True)
    return prior.astype(np.float32), distances.astype(np.float32)


def row_entropy(matrix: np.ndarray) -> np.ndarray:
    values = matrix.astype(np.float64)
    positive_mask = values > 0
    terms = np.zeros_like(values)
    terms[positive_mask] = (
        values[positive_mask] * np.log(values[positive_mask])
    )
    return -np.sum(terms, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probability-floor", type=float, default=1e-12)
    args = parser.parse_args()

    with np.load(args.dataset) as data:
        values = data["values"].astype(np.float64)
        active_mask = data["active_mask"].astype(bool)
        train_end = int(data["train_end_index"])
        users = data["users"].tolist()
    profiles, active_fractions = profile_matrix(
        values, active_mask, train_end
    )
    prior, distances = gaussian_prior(
        profiles, probability_floor=args.probability_floor
    )
    entropy = row_entropy(prior)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "clue_profile_features.npy", profiles)
    np.save(args.output_dir / "clue_profile_prior.npy", prior)
    np.save(args.output_dir / "clue_profile_distances.npy", distances)
    np.save(
        args.output_dir / "clue_profile_entropy.npy", entropy
    )
    metadata = {
        "users": len(users),
        "profile_dimensions": int(profiles.shape[1]),
        "median_pairwise_distance": float(
            np.median(distances[np.triu_indices(len(users), k=1)])
        ),
        "profile_entropy_mean": float(np.mean(entropy)),
        "profile_entropy_median": float(np.median(entropy)),
        "maximum_uniform_entropy": float(np.log(len(users) - 1)),
        "active_fraction_mean": float(np.mean(active_fractions)),
        "probability_floor": args.probability_floor,
        "users_order": users,
    }
    (args.output_dir / "clue_profile_prior_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
