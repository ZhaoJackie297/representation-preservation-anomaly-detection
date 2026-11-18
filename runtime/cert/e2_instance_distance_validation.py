from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e2_instance"
    / "distance_validation"
)

COUNT_FEATURES = [
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
]
TIMING_FEATURES = [
    "first_logon_minute",
    "last_logoff_minute",
    "active_span_minutes",
]
PROFILE_FEATURES = [*COUNT_FEATURES, *TIMING_FEATURES]

DATASETS = {
    "r42": {
        "processed": ROOT / "processed_r42",
        "dataset": "r4.2_sequence_dataset.npz",
        "raw": "r4.2_user_day_features.parquet",
    },
    "r52": {
        "processed": ROOT / "processed_r52",
        "dataset": "r5.2_sequence_dataset.npz",
        "raw": "r5.2_user_day_features.parquet",
    },
    "r62": {
        "processed": ROOT / "processed_r62",
        "dataset": "r6.2_sequence_dataset.npz",
        "raw": "r6.2_user_day_features.parquet",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-tag",
        default="profile_stability_missing_safe",
        help="Output filename stem under distance_validation.",
    )
    return parser.parse_args()


def align_raw(
    path: Path,
    users: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    frame = pd.read_parquet(
        path,
        columns=["user_id", "date", *PROFILE_FEATURES],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    index = pd.MultiIndex.from_product(
        [users.tolist(), pd.to_datetime(dates)],
        names=["user_id", "date"],
    )
    aligned = frame.set_index(["user_id", "date"]).reindex(index)
    values = aligned[PROFILE_FEATURES].to_numpy(
        dtype=np.float64
    ).reshape(len(users), len(dates), len(PROFILE_FEATURES))
    counts = values[:, :, : len(COUNT_FEATURES)]
    values[:, :, : len(COUNT_FEATURES)] = np.where(
        np.isfinite(counts),
        np.log1p(np.maximum(counts, 0.0)),
        np.nan,
    )
    timing = values[:, :, len(COUNT_FEATURES) :]
    values[:, :, len(COUNT_FEATURES) :] = np.where(
        np.isfinite(timing) & (timing >= 0.0),
        timing,
        np.nan,
    )
    return values.astype(np.float32)


def profile(
    raw: np.ndarray,
    active: np.ndarray,
    start: int,
    end: int,
    dimensions: int,
) -> np.ndarray:
    rows = []
    for user_index in range(raw.shape[0]):
        mask = active[user_index, start : end + 1]
        values = raw[user_index, start : end + 1, :dimensions][mask]
        if len(values) == 0:
            values = raw[user_index, start : end + 1, :dimensions]
        rows.append(
            np.concatenate(
                [
                    np.nanmedian(values, axis=0),
                    np.nanquantile(values, 0.9, axis=0),
                    np.asarray([mask.mean()]),
                ]
            )
        )
    result = np.stack(rows)
    population_median = np.nanmedian(result, axis=0)
    population_median = np.where(
        np.isfinite(population_median),
        population_median,
        0.0,
    )
    missing_rows, missing_columns = np.nonzero(~np.isfinite(result))
    result[missing_rows, missing_columns] = population_median[
        missing_columns
    ]
    mean = result.mean(axis=0)
    std = np.where(result.std(axis=0) < 1e-8, 1.0, result.std(axis=0))
    return ((result - mean) / std).astype(np.float64)


def top_neighbors(distance: np.ndarray, k: int) -> list[np.ndarray]:
    work = distance.copy()
    np.fill_diagonal(work, np.inf)
    return [np.argsort(row)[:k] for row in work]


def stability(
    first: np.ndarray,
    second: np.ndarray,
    dimensions: int,
) -> dict:
    first_distance = pairwise_distances(first, metric="euclidean")
    second_distance = pairwise_distances(second, metric="euclidean")
    upper = np.triu_indices(len(first), k=1)
    rho = float(
        spearmanr(
            first_distance[upper],
            second_distance[upper],
        ).statistic
    )
    first_neighbors = top_neighbors(first_distance, 10)
    second_neighbors = top_neighbors(second_distance, 10)
    overlaps = [
        len(set(left).intersection(right)) / 10.0
        for left, right in zip(first_neighbors, second_neighbors)
    ]
    return {
        "profile_dimensions": dimensions,
        "users": int(len(first)),
        "pairwise_distance_spearman": rho,
        "top10_neighbor_jaccard_mean": float(np.mean(overlaps)),
        "top10_neighbor_jaccard_median": float(np.median(overlaps)),
    }


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, files in DATASETS.items():
        dataset = np.load(
            files["processed"] / files["dataset"],
            allow_pickle=False,
        )
        users = dataset["users"].astype(str)
        dates = dataset["dates"].astype(str)
        active = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
        midpoint = train_end // 2
        raw = align_raw(files["processed"] / files["raw"], users, dates)
        dataset_result = {
            "train_start": 0,
            "first_half_end": midpoint,
            "second_half_start": midpoint + 1,
            "second_half_end": train_end,
            "variants": {},
        }
        for label, dimensions in (
            ("counts_41d", len(COUNT_FEATURES)),
            ("behavior_47d", len(PROFILE_FEATURES)),
        ):
            first = profile(raw, active, 0, midpoint, dimensions)
            second = profile(
                raw,
                active,
                midpoint + 1,
                train_end,
                dimensions,
            )
            dataset_result["variants"][label] = stability(
                first,
                second,
                dimensions,
            )
            print(f"{name} {label}: {dataset_result['variants'][label]}")
        results[name] = dataset_result
    output_path = OUT / f"{args.output_tag}.json"
    output_path.write_text(
        json.dumps(
            {
                "purpose": (
                    "Construct-validity check for the training-only "
                    "user profile distance used by SoftCLT instance weights."
                ),
                "labels_used": False,
                "missing_policy": (
                    "timing sentinel -1 -> NaN; per-user "
                    "nanmedian/nanquantile; fully missing feature -> "
                    "training population median"
                ),
                "datasets": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
