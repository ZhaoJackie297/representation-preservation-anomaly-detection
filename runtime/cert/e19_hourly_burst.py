from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
CHUNK_ROWS = 1_000_000

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from e19_user_manifold import (  # noqa: E402
    dataset_config,
    evaluate_method,
    load_components,
)
from quota_relationship_sim import daily_percentile  # noqa: E402
from risk_topk_r42 import compute_risk  # noqa: E402


CHANNELS = ["logon", "device", "file", "email", "http"]


def source_directory(dataset_name: str) -> Path:
    version = "r4.2" if dataset_name == "r42" else "r5.2"
    return ROOT / f"{version}_extracted" / version


def increment_events(
    path: Path,
    channel_index: int,
    hour_counts: np.ndarray,
    user_index: dict[str, int],
    date_index: dict[str, int],
    *,
    dataset_name: str,
) -> None:
    print(f"  reading {path.name}", flush=True)
    columns = ["date", "user"]
    if dataset_name == "r52" and path.name == "email.csv":
        columns.append("activity")
    if dataset_name == "r52" and path.name == "file.csv":
        columns.extend(["activity", "to_removable_media"])
    processed = 0
    reader = pd.read_csv(
        path,
        usecols=columns,
        dtype="string",
        chunksize=CHUNK_ROWS,
        on_bad_lines="skip",
        low_memory=False,
    )
    for chunk in reader:
        if "activity" in chunk.columns:
            activity = chunk["activity"].fillna("").str.strip().str.lower()
            if path.name == "email.csv":
                chunk = chunk[activity.eq("send")]
            else:
                removable = (
                    chunk["to_removable_media"]
                    .fillna("")
                    .str.strip()
                    .str.lower()
                    .eq("true")
                )
                chunk = chunk[
                    activity.eq("file write") & removable
                ]
        if chunk.empty:
            continue
        timestamps = pd.to_datetime(
            chunk["date"],
            format="%m/%d/%Y %H:%M:%S",
            errors="coerce",
        )
        users = chunk["user"].fillna("").str.strip()
        user_ids = users.map(user_index)
        day_texts = timestamps.dt.strftime("%Y-%m-%d")
        day_ids = day_texts.map(date_index)
        hours = timestamps.dt.hour
        valid = (
            user_ids.notna()
            & day_ids.notna()
            & hours.notna()
        )
        if not bool(valid.any()):
            continue
        np.add.at(
            hour_counts,
            (
                user_ids[valid].to_numpy(dtype=np.int64),
                day_ids[valid].to_numpy(dtype=np.int64),
                hours[valid].to_numpy(dtype=np.int64),
                channel_index,
            ),
            1,
        )
        processed += int(valid.sum())
        if processed and processed % 5_000_000 < CHUNK_ROWS:
            print(
                f"    {path.name}: {processed:,} rows",
                flush=True,
            )


def build_hour_counts(
    dataset_name: str,
    users: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    cache_path = (
        OUT / f"e19_{dataset_name}_hourly_counts.npy"
    )
    users_count = len(users)
    days_count = len(dates)
    if cache_path.exists():
        counts = np.load(cache_path, mmap_mode="r+")
        if counts.shape == (users_count, days_count, 24, 5):
            return counts
    counts = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.uint32,
        shape=(users_count, days_count, 24, 5),
    )
    counts[:] = 0
    user_index = {
        str(user): index for index, user in enumerate(users)
    }
    date_index = {
        str(day)[:10]: index for index, day in enumerate(dates)
    }
    directory = source_directory(dataset_name)
    for channel_index, channel in enumerate(CHANNELS):
        increment_events(
            directory / f"{channel}.csv",
            channel_index,
            counts,
            user_index,
            date_index,
            dataset_name=dataset_name,
        )
    counts.flush()
    return counts


def burst_features(hour_counts: np.ndarray) -> np.ndarray:
    totals = hour_counts.sum(axis=2, dtype=np.float64)
    maxima = hour_counts.max(axis=2).astype(np.float64)
    active_hours = (hour_counts > 0).sum(axis=2).astype(np.float64)
    shares = maxima / np.maximum(totals, 1.0)
    total_hourly = hour_counts.sum(axis=3)
    total_hourly_max = total_hourly.max(axis=2).astype(np.float64)
    total_share = total_hourly_max / np.maximum(
        total_hourly.sum(axis=2), 1.0
    )
    features = np.concatenate(
        [
            np.log1p(maxima),
            shares,
            np.log1p(active_hours),
            np.log1p(total_hourly_max)[:, :, None],
            total_share[:, :, None],
        ],
        axis=-1,
    )
    return features.astype(np.float32)


def burst_deviation(
    features: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, dict]:
    users, time_steps, dimensions = features.shape
    train_mask = (
        active_mask[:, : train_end + 1]
        & np.isfinite(features[:, : train_end + 1]).all(axis=-1)
    )
    global_rows = features[:, : train_end + 1][train_mask]
    global_center = np.median(global_rows, axis=0)
    centers = np.empty((users, dimensions), dtype=np.float64)
    residual_rows = []
    for user in range(users):
        training = np.flatnonzero(train_mask[user])
        if len(training) >= 4:
            centers[user] = np.median(
                features[user, training], axis=0
            )
            residual_rows.append(
                features[user, training] - centers[user]
            )
        else:
            centers[user] = global_center
    residuals = np.concatenate(residual_rows, axis=0)
    scale = np.sqrt(np.mean(np.square(residuals), axis=0))
    scale = np.where(
        np.isfinite(scale) & (scale >= 1e-8),
        scale,
        1.0,
    )
    distances = np.full(
        active_mask.shape, np.nan, dtype=np.float32
    )
    for user in range(users):
        active = np.flatnonzero(active_mask[user])
        if len(active) == 0:
            continue
        standardized = (
            features[user, active] - centers[user]
        ) / scale
        valid = np.isfinite(standardized).all(axis=1)
        distances[user, active[valid]] = np.sqrt(
            np.mean(np.square(standardized[valid]), axis=1)
        )
    return distances, {
        "dimensions": dimensions,
        "mean_scale": float(np.mean(scale)),
    }


def make_signal(data: dict, scores: np.ndarray) -> np.ndarray:
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        scores,
        data["active_mask"],
    )
    return daily_percentile(
        dynamic, data["active_mask"]
    ).astype(np.float32)


def run_dataset(dataset_name: str) -> None:
    config = dataset_config(dataset_name)
    data = config["data"]
    dataset = np.load(
        config["dataset_file"], allow_pickle=False
    )
    users = dataset["users"]
    dates = dataset["dates"]
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    counts = build_hour_counts(dataset_name, users, dates)
    features = burst_features(counts)
    burst_raw, diagnostics = burst_deviation(
        features, active_mask, train_end
    )
    burst_q = per_user_quantile_scores(
        burst_raw, active_mask, train_end
    )
    relation_q = np.load(
        OUT / f"e19_{dataset_name}_relation_quantile.npy"
    )
    _, masked_q, _, channel_q = load_components(dataset_name)
    dual_q = np.load(
        OUT / f"e19_{dataset_name}_dual_mean_manifold_quantile.npy"
    )
    core_q = (masked_q + dual_q + channel_q) / 3.0
    consensus_q = np.minimum(burst_q, relation_q)
    final_q = np.maximum(core_q, consensus_q)
    burst_equal_q = (
        masked_q + dual_q + channel_q + burst_q
    ) / 4.0
    signals = {
        "A3_dual_mean": make_signal(data, core_q),
        "C1_burst_equal_sensitivity": make_signal(
            data, burst_equal_q
        ),
        "C2_burst_relation_consensus": make_signal(
            data, final_q
        ),
    }
    auxiliary_all = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    auxiliary = auxiliary_all[config["auxiliary_key"]]
    results = {}
    for method, signal in signals.items():
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
        OUT / f"e19_{dataset_name}_hourly_burst_raw.npy",
        burst_raw,
    )
    np.save(
        OUT / f"e19_{dataset_name}_hourly_burst_quantile.npy",
        burst_q,
    )
    np.save(
        OUT / f"e19_{dataset_name}_consensus_quantile.npy",
        consensus_q,
    )
    summary = {
        "dataset": dataset_name,
        "protocol": {
            "default": "C2_burst_relation_consensus",
            "fusion": (
                "max(core quantile, min(burst quantile, "
                "relation quantile))"
            ),
            "burst_equal_is_sensitivity_only": True,
            "weights_searched": False,
            "labels_used_for_training": False,
        },
        "burst_diagnostics": diagnostics,
        "results": results,
    }
    path = OUT / f"e19_{dataset_name}_burst_summary.json"
    path.write_text(
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
