from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

CLUE_DIR = Path(__file__).resolve().parent / "clue_lds"
if str(CLUE_DIR) not in sys.path:
    sys.path.insert(0, str(CLUE_DIR))

from audit_profile_prior_information import profile_prior_information


PROFILE_FEATURES = [
    "logon_count",
    "logoff_count",
    "after_hours_logon_count",
    "other_pc_logon_count",
    "usb_connect_count",
    "usb_disconnect_count",
    "file_copy_count",
    "archive_copy_count",
    "executable_copy_count",
    "unique_file_extensions",
    "email_count",
    "email_recipient_count",
    "external_recipient_count",
    "email_attachment_count",
    "email_size_bytes",
    "after_hours_email_count",
    "http_count",
    "after_hours_http_count",
    "external_http_count",
    "unique_http_domains",
    "first_logon_minute",
    "last_logoff_minute",
    "active_span_minutes",
]


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_profile_features(
    *,
    dataset_path: Path,
    feature_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    dataset = np.load(dataset_path, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {day: index for index, day in enumerate(dates)}
    raw = np.full(
        (len(users), train_end + 1, len(PROFILE_FEATURES)),
        np.nan,
        dtype=np.float64,
    )

    with feature_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "date", *PROFILE_FEATURES}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing feature columns: {sorted(missing)}")
        for row in reader:
            user = user_index.get(row["user_id"])
            day = date_index.get(row["date"])
            if user is None or day is None or day > train_end:
                continue
            raw[user, day] = [
                float(row[feature]) if row[feature] not in ("", "nan", "NaN") else np.nan
                for feature in PROFILE_FEATURES
            ]

    # Match R0's align_raw exactly: count features use log1p, while timing
    # sentinel values below zero are treated as missing.
    count_dimensions = 20
    counts = raw[:, :, :count_dimensions]
    raw[:, :, :count_dimensions] = np.where(
        np.isfinite(counts),
        np.log1p(np.maximum(counts, 0.0)),
        np.nan,
    )
    timing = raw[:, :, count_dimensions:]
    raw[:, :, count_dimensions:] = np.where(
        np.isfinite(timing) & (timing >= 0.0),
        timing,
        np.nan,
    )
    raw = raw.astype(np.float32)

    profiles = []
    for user in range(len(users)):
        user_active = active_mask[user, : train_end + 1]
        values = raw[user, : train_end + 1][user_active]
        if len(values) == 0:
            values = raw[user, : train_end + 1]
        median = np.nanmedian(values, axis=0)
        q90 = np.nanquantile(values, 0.9, axis=0)
        active_fraction = np.asarray([user_active.mean()])
        profiles.append(np.concatenate([median, q90, active_fraction]))
    profile = np.stack(profiles)
    population_median = np.nanmedian(profile, axis=0)
    population_median = np.where(np.isfinite(population_median), population_median, 0.0)
    missing_rows, missing_columns = np.nonzero(~np.isfinite(profile))
    profile[missing_rows, missing_columns] = population_median[missing_columns]
    mean = profile.mean(axis=0)
    std = np.where(profile.std(axis=0) < 1e-8, 1.0, profile.std(axis=0))
    return (
        ((profile - mean) / std).astype(np.float64),
        active_mask,
        users,
        train_end,
    )


def pairwise_distances(features: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(features * features, axis=1)
    squared = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * features @ features.T
    )
    return np.sqrt(np.clip(squared, 0.0, None))


def gaussian_prior(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    prior = np.exp(-(distances * distances) / (2.0 * bandwidth**2))
    np.fill_diagonal(prior, 0.0)
    totals = prior.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("a prior row has zero total mass")
    return (prior / totals).astype(np.float32)


def knn_bandwidth(distances: np.ndarray, k: int = 10) -> float:
    if k <= 0 or k >= distances.shape[0]:
        raise ValueError("k must be between 1 and n_users-1")
    ordered = np.sort(distances, axis=1)
    return float(np.median(ordered[:, k]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--knn-k", type=int, default=10)
    args = parser.parse_args()

    features, _, users, train_end = build_profile_features(
        dataset_path=args.dataset,
        feature_csv=args.feature_csv,
    )
    distances = pairwise_distances(features)
    upper = distances[np.triu_indices(len(distances), k=1)]
    primary_bandwidth = float(np.median(upper))
    alternative_bandwidth = knn_bandwidth(distances, args.knn_k)
    primary = profile_prior_information(
        gaussian_prior(distances, primary_bandwidth),
        top_m=args.top_m,
        flat_ratio=0.95,
        majority_fraction=0.80,
    )
    alternative = profile_prior_information(
        gaussian_prior(distances, alternative_bandwidth),
        top_m=args.top_m,
        flat_ratio=0.95,
        majority_fraction=0.80,
    )
    result = {
        "dataset": str(args.dataset),
        "feature_csv": str(args.feature_csv),
        "dataset_sha256": file_sha256(args.dataset),
        "feature_csv_sha256": file_sha256(args.feature_csv),
        "users": int(len(users)),
        "profile_dimensions": int(features.shape[1]),
        "train_end_index": int(train_end),
        "primary_bandwidth": primary_bandwidth,
        "knn_k": int(args.knn_k),
        "knn_bandwidth": alternative_bandwidth,
        "primary": primary,
        "knn_alternative": alternative,
        "status": (
            "BOTH_PRIORS_FLAT"
            if primary["triggered"] and alternative["triggered"]
            else "PRIMARY_FLAT_ALTERNATIVE_INFORMATIVE"
            if primary["triggered"]
            else "PRIMARY_INFORMATIVE"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
