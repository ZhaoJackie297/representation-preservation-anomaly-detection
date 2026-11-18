from __future__ import annotations

import csv
import io
import json
import math
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
TS2VEC_ARRAYS = PROCESSED / "r4.2_ts2vec_scores.npz"
ZSCORE_DAILY = PROCESSED / "r4.2_zscore_daily_scores.csv"
IFOREST_ARRAYS = PROCESSED / "r4.2_iforest_raw_scores.npz"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"
ANSWERS = ROOT / "answers.tar.bz2"

RESULTS = PROCESSED / "r4.2_per_user_calibrated_trend.csv"
SUMMARY = PROCESSED / "r4.2_per_user_calibrated_summary.json"
EARLY_RESULTS = PROCESSED / "r4.2_early_horizon_trend.csv"
CALIBRATED_ARRAYS = PROCESSED / "r4.2_calibrated_method_scores.npz"

sys.path.insert(0, str(ROOT))
from risk_topk_r42 import compute_risk  # noqa: E402
from baseline_zscore_r42 import (  # noqa: E402
    FEATURES as ZSCORE_FEATURES,
    LOG1P_FEATURES as ZSCORE_LOG1P_FEATURES,
    MIN_PERSONAL_DAYS,
    robust_location_scale,
)


K_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000]
EARLY_HORIZONS = [1, 3, 7, 14]
FEATURE_CSV = PROCESSED / "r4.2_user_day_features.csv"


def matrix_from_csv(
    path: Path,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    column: str,
) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["user_id", "date", column])
    frame["date"] = pd.to_datetime(frame["date"])
    index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    return (
        frame.set_index(["user_id", "date"])[column]
        .reindex(index)
        .to_numpy(dtype=np.float32)
        .reshape(len(users), len(dates))
    )


def build_unclipped_zscore(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    raw = pd.read_csv(FEATURE_CSV, parse_dates=["date"])
    index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    values = (
        raw.set_index(["user_id", "date"])[ZSCORE_FEATURES]
        .reindex(index)
        .to_numpy(dtype=np.float64)
        .reshape(len(users), len(dates), len(ZSCORE_FEATURES))
    )
    for feature_index, feature in enumerate(ZSCORE_FEATURES):
        if feature in ZSCORE_LOG1P_FEATURES:
            mask = np.isfinite(values[:, :, feature_index])
            values[:, :, feature_index][mask] = np.log1p(
                values[:, :, feature_index][mask]
            )

    train_active = active_mask[:, : train_end + 1]
    global_median, global_scale = robust_location_scale(
        values[:, : train_end + 1][train_active]
    )
    global_scale = np.where(
        (global_scale < 1e-8) | ~np.isfinite(global_scale),
        1.0,
        global_scale,
    )
    deviations = np.full(values.shape[:2], np.nan, dtype=np.float32)
    for user_index in range(len(users)):
        active_training = np.flatnonzero(train_active[user_index])
        if len(active_training) >= MIN_PERSONAL_DAYS:
            median, scale = robust_location_scale(
                values[user_index, : train_end + 1][active_training]
            )
            median = np.where(np.isfinite(median), median, global_median)
            scale = np.where(
                np.isfinite(scale) & (scale >= 1e-8), scale, global_scale
            )
        else:
            median, scale = global_median, global_scale
        for time_index in np.flatnonzero(active_mask[user_index]):
            z = np.abs((values[user_index, time_index] - median) / scale)
            deviations[user_index, time_index] = np.sort(z)[-3:].mean()
    return deviations


def per_user_quantile_scores(
    raw_scores: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    calibrated = np.full(raw_scores.shape, np.nan, dtype=np.float32)
    for user_index in range(raw_scores.shape[0]):
        train_mask = active_mask[user_index, : train_end + 1] & np.isfinite(
            raw_scores[user_index, : train_end + 1]
        )
        train_values = np.sort(raw_scores[user_index, : train_end + 1][train_mask])
        if len(train_values) == 0:
            continue
        n = len(train_values)
        for time_index in np.flatnonzero(active_mask[user_index]):
            value = raw_scores[user_index, time_index]
            rank = int(np.searchsorted(train_values, value, side="right"))
            tail_probability = (n - rank + 1) / (n + 1)
            calibrated[user_index, time_index] = max(
                0.0, -math.log(tail_probability)
            )
    return calibrated


def rank_users(
    scores: np.ndarray,
    raw_scores: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    ranks = np.full(scores.shape, 0, dtype=np.int32)
    for time_index in range(scores.shape[1]):
        active = np.flatnonzero(active_mask[:, time_index])
        ordering = np.lexsort(
            (
                np.arange(len(active)),
                -raw_scores[active, time_index],
                -scores[active, time_index],
            )
        )
        ordered_users = active[ordering]
        ranks[ordered_users, time_index] = np.arange(1, len(ordered_users) + 1)
    return ranks


def label_matrix(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    label_file: Path,
) -> np.ndarray:
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)}
    labels = pd.read_csv(label_file, parse_dates=["date"])
    matrix = np.zeros((len(users), len(dates)), dtype=np.uint8)
    for row in labels.itertuples(index=False):
        account = row.user_id
        date_text = row.date.strftime("%Y-%m-%d")
        if account in user_index and date_text in date_index:
            matrix[user_index[account], date_index[date_text]] = 1
    return matrix


def evaluate_rankings(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    labels: np.ndarray,
    incidents: pd.DataFrame,
    rank_matrices: dict[str, np.ndarray],
    train_end: int,
    val_end: int,
) -> pd.DataFrame:
    validation_start = dates[train_end + 1].normalize()
    test_start = dates[val_end + 1].normalize()
    final_end = dates[-1].normalize()
    split_ranges = {
        "validation": (validation_start, test_start - pd.Timedelta(days=1)),
        "test": (test_start, final_end),
        "validation_plus_test": (validation_start, final_end),
    }
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)}

    incidents = incidents.copy()
    incidents["start_date"] = pd.to_datetime(incidents["start"]).dt.normalize()
    incidents["end_date"] = pd.to_datetime(incidents["end"]).dt.normalize()

    rows = []
    for method, ranks in rank_matrices.items():
        for split, (split_start, split_end) in split_ranges.items():
            split_time_indices = np.array(
                [
                    index
                    for index, date in enumerate(dates)
                    if split_start <= date <= split_end
                ]
            )
            split_mask = (dates >= split_start) & (dates <= split_end)
            split_labels = labels[:, split_mask]
            split_ranks = ranks[:, split_mask]
            split_incidents = incidents[
                (incidents["start_date"] >= split_start)
                & (incidents["start_date"] <= split_end)
            ]
            positive_days = int(split_labels.sum())
            day_count = int(split_mask.sum())

            for k in K_VALUES:
                selected = (split_ranks > 0) & (split_ranks <= k)
                alert_slots = int(selected.sum())
                true_alerts = int((selected & (split_labels == 1)).sum())
                hits = 0
                delays = []
                for incident in split_incidents.itertuples(index=False):
                    accounts = [
                        user_index[account]
                        for account in incident.observed_accounts.split("|")
                        if account in user_index
                    ]
                    incident_dates = [
                        date_index[date.strftime("%Y-%m-%d")]
                        for date in pd.date_range(
                            incident.start_date, incident.end_date, freq="D"
                        )
                        if date.strftime("%Y-%m-%d") in date_index
                    ]
                    first_alert = None
                    for time_index in incident_dates:
                        candidate_ranks = ranks[np.asarray(accounts), time_index]
                        candidate_ranks = candidate_ranks[candidate_ranks > 0]
                        if len(candidate_ranks) and candidate_ranks.min() <= k:
                            first_alert = dates[time_index]
                            break
                    if first_alert is not None:
                        hits += 1
                        delays.append((first_alert.normalize() - incident.start_date).days)
                rows.append(
                    {
                        "method": method,
                        "split": split,
                        "k": k,
                        "alert_slots": alert_slots,
                        "true_alert_slots": true_alerts,
                        "precision_at_k": true_alerts / alert_slots if alert_slots else 0.0,
                        "account_day_recall": true_alerts / positive_days if positive_days else 0.0,
                        "false_alert_slots": alert_slots - true_alerts,
                        "false_alerts_per_day": (
                            (alert_slots - true_alerts) / day_count if day_count else 0.0
                        ),
                        "incidents_in_split": int(len(split_incidents)),
                        "incidents_hit": hits,
                        "incident_recall": (
                            hits / len(split_incidents) if len(split_incidents) else 0.0
                        ),
                        "median_detection_delay_days": (
                            float(np.median(delays)) if delays else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_early_horizons(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    incidents: pd.DataFrame,
    rank_matrices: dict[str, np.ndarray],
    train_end: int,
    val_end: int,
) -> pd.DataFrame:
    validation_start = dates[train_end + 1].normalize()
    test_start = dates[val_end + 1].normalize()
    final_end = dates[-1].normalize()
    split_ranges = {
        "validation": (validation_start, test_start - pd.Timedelta(days=1)),
        "test": (test_start, final_end),
        "validation_plus_test": (validation_start, final_end),
    }
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)}
    incidents = incidents.copy()
    incidents["start_date"] = pd.to_datetime(incidents["start"]).dt.normalize()
    incidents["end_date"] = pd.to_datetime(incidents["end"]).dt.normalize()

    rows = []
    for method, ranks in rank_matrices.items():
        for split, (split_start, split_end) in split_ranges.items():
            split_incidents = incidents[
                (incidents["start_date"] >= split_start)
                & (incidents["start_date"] <= split_end)
            ]
            for k in [value for value in K_VALUES if value <= 200]:
                for horizon in EARLY_HORIZONS:
                    hits = 0
                    delays = []
                    for incident in split_incidents.itertuples(index=False):
                        accounts = [
                            user_index[account]
                            for account in incident.observed_accounts.split("|")
                            if account in user_index
                        ]
                        horizon_end = min(
                            incident.end_date,
                            incident.start_date + pd.Timedelta(days=horizon - 1),
                        )
                        for date in pd.date_range(
                            incident.start_date, horizon_end, freq="D"
                        ):
                            date_text = date.strftime("%Y-%m-%d")
                            if date_text not in date_index:
                                continue
                            time_index = date_index[date_text]
                            candidate_ranks = ranks[np.asarray(accounts), time_index]
                            candidate_ranks = candidate_ranks[candidate_ranks > 0]
                            if len(candidate_ranks) and candidate_ranks.min() <= k:
                                hits += 1
                                delays.append(
                                    (date.normalize() - incident.start_date).days
                                )
                                break
                    rows.append(
                        {
                            "method": method,
                            "split": split,
                            "k": k,
                            "horizon_days": horizon,
                            "incidents_in_split": len(split_incidents),
                            "incidents_hit": hits,
                            "early_recall": (
                                hits / len(split_incidents)
                                if len(split_incidents)
                                else 0.0
                            ),
                            "median_hit_delay_days": (
                                float(np.median(delays)) if delays else None
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    dataset = np.load(DATASET, allow_pickle=False)
    ts2vec = np.load(TS2VEC_ARRAYS, allow_pickle=False)
    iforest = np.load(IFOREST_ARRAYS, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])

    raw_scores = {
        "ts2vec": ts2vec["raw_error"].astype(np.float32),
        "zscore": build_unclipped_zscore(users, dates, active_mask, train_end),
        "iforest": iforest["iforest_raw"].astype(np.float32),
    }
    labels = label_matrix(
        users, dates, PROCESSED / "r4.2_malicious_account_days.csv"
    )
    incidents = pd.read_csv(INCIDENTS)

    calibrated = {}
    dynamic = {}
    ranks = {}
    for name, raw in raw_scores.items():
        calibrated[name] = per_user_quantile_scores(raw, active_mask, train_end)
        calibrated_name = f"{name}_quantile"
        dynamic_name = f"{name}_quantile_dynamic"
        dynamic[name] = compute_risk(users, dates, calibrated[name], active_mask)
        ranks[calibrated_name] = rank_users(calibrated[name], raw, active_mask)
        ranks[dynamic_name] = rank_users(
            dynamic[name], calibrated[name], active_mask
        )

    results = evaluate_rankings(
        users,
        dates,
        active_mask,
        labels,
        incidents,
        ranks,
        train_end,
        val_end,
    )
    results.to_csv(RESULTS, index=False, encoding="utf-8")
    early = evaluate_early_horizons(
        users, dates, incidents, ranks, train_end, val_end
    )
    early.to_csv(EARLY_RESULTS, index=False, encoding="utf-8")
    np.savez_compressed(
        CALIBRATED_ARRAYS,
        users=np.asarray(users, dtype="U32"),
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        **{
            f"{name}_quantile": calibrated[name].astype(np.float32)
            for name in calibrated
        },
        **{
            f"{name}_quantile_dynamic": dynamic[name].astype(np.float32)
            for name in dynamic
        },
    )
    summary = {
        "result_file": str(RESULTS),
        "early_result_file": str(EARLY_RESULTS),
        "calibrated_arrays_file": str(CALIBRATED_ARRAYS),
        "methods": list(ranks),
        "k_values": K_VALUES,
        "calibration": (
            "Per-user empirical tail probability on training-period raw scores; "
            "labels not used."
        ),
        "dynamic_risk_parameters": {
            "window_days": 7,
            "decay": 0.65,
            "threshold": 3.0,
            "weights": [0.7, 0.2, 0.1],
        },
        "labels_used_for_calibration": False,
        "labels_used_for_risk_parameters": False,
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    holdout = results[results["split"] == "validation_plus_test"]
    print(
        holdout.pivot(
            index="k",
            columns="method",
            values="incident_recall",
        ).to_string(),
        flush=True,
    )
    early_holdout = early[early["split"] == "validation_plus_test"]
    print(
        early_holdout.pivot_table(
            index=["k", "horizon_days"],
            columns="method",
            values="early_recall",
        ).to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
