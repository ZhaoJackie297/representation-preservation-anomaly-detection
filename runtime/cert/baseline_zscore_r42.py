from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
FEATURE_CSV = PROCESSED / "r4.2_user_day_features.csv"
TS2VEC_SCORES = PROCESSED / "r4.2_ts2vec_daily_errors.csv"
SCORE_FILE = PROCESSED / "r4.2_zscore_daily_scores.csv"
TOPK_FILE = PROCESSED / "r4.2_zscore_daily_topk.csv"
METRICS_FILE = PROCESSED / "r4.2_zscore_topk_metrics.json"
COMPARISON_FILE = PROCESSED / "r4.2_method_comparison.json"

sys.path.insert(0, str(ROOT))
from risk_topk_r42 import (  # noqa: E402
    K_VALUES,
    build_topk,
    compute_risk,
    evaluate,
    load_incidents,
)


FEATURES = [
    "logon_count",
    "logoff_count",
    "first_logon_minute",
    "last_logoff_minute",
    "active_span_minutes",
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
LOG1P_FEATURES = FEATURES[0:2] + FEATURES[5:]
TOP_FEATURE_COUNT = 3
MIN_PERSONAL_DAYS = 30
MAX_ABS_Z = 20.0


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    scale = 1.4826 * mad
    return median, scale


def main() -> None:
    dataset = np.load(DATASET, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])

    raw = pd.read_csv(FEATURE_CSV, parse_dates=["date"])
    indexed = raw.set_index(["user_id", "date"])[FEATURES]
    full_index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    aligned = indexed.reindex(full_index)
    values = aligned.to_numpy(dtype=np.float64).reshape(
        len(users), len(dates), len(FEATURES)
    )

    for feature_index, feature in enumerate(FEATURES):
        if feature in LOG1P_FEATURES:
            mask = np.isfinite(values[:, :, feature_index])
            values[:, :, feature_index][mask] = np.log1p(
                values[:, :, feature_index][mask]
            )

    train_slice = values[:, : train_end + 1, :]
    train_active = active_mask[:, : train_end + 1]
    global_median, global_scale = robust_location_scale(
        train_slice[train_active]
    )
    global_scale = np.where(
        (global_scale < 1e-8) | ~np.isfinite(global_scale),
        1.0,
        global_scale,
    )

    deviations = np.full(values.shape[:2], np.nan, dtype=np.float32)
    personal_stats = {}
    for user_index, user in enumerate(users):
        active_indices = np.flatnonzero(train_active[user_index])
        count = len(active_indices)
        if count >= MIN_PERSONAL_DAYS:
            median, scale = robust_location_scale(
                train_slice[user_index, active_indices]
            )
            valid_scale = np.isfinite(scale) & (scale >= 1e-8)
            median = np.where(np.isfinite(median), median, global_median)
            scale = np.where(valid_scale, scale, global_scale)
            source = "personal"
        else:
            median = global_median
            scale = global_scale
            source = "global_fallback"
        personal_stats[user] = {
            "active_training_days": count,
            "baseline_source": source,
        }

        for time_index in np.flatnonzero(active_mask[user_index]):
            row = values[user_index, time_index]
            z = np.abs((row - median) / scale)
            z = np.clip(z, 0.0, MAX_ABS_Z)
            deviations[user_index, time_index] = np.sort(z)[-TOP_FEATURE_COUNT:].mean()

    train_deviation_values = deviations[:, : train_end + 1][train_active]
    train_deviation_values = train_deviation_values[
        np.isfinite(train_deviation_values)
    ]
    median, mad = robust_location_scale(train_deviation_values.reshape(-1, 1))
    scale = 1.4826 * mad[0]
    if scale < 1e-8:
        scale = float(np.std(train_deviation_values))
    if scale < 1e-8:
        scale = 1.0
    median_value = float(np.asarray(median).reshape(-1)[0])
    scale_value = float(np.asarray(scale).reshape(-1)[0])
    anomaly_scores = np.clip(
        (deviations - median_value) / scale_value, 0.0, None
    ).squeeze()
    if anomaly_scores.shape != deviations.shape:
        raise RuntimeError(
            f"Unexpected anomaly score shape: {anomaly_scores.shape}"
        )
    print(
        "Z-score matrix shapes:",
        values.shape,
        deviations.shape,
        anomaly_scores.shape,
        flush=True,
    )

    score_rows = []
    for user_index, user in enumerate(users):
        for time_index, date in enumerate(dates):
            if not active_mask[user_index, time_index]:
                continue
            score_rows.append(
                {
                    "user_id": user,
                    "date": date.strftime("%Y-%m-%d"),
                    "split": (
                        "train"
                        if time_index <= train_end
                        else "validation"
                        if time_index <= val_end
                        else "test"
                    ),
                    "robust_z_score": float(anomaly_scores[user_index, time_index]),
                    "top_feature_deviation": float(
                        deviations[user_index, time_index]
                    ),
                }
            )
    score_frame = pd.DataFrame(score_rows)

    ts2vec_scores = pd.read_csv(
        TS2VEC_SCORES,
        dtype={"scenarios": "string", "insider_ids": "string"},
        low_memory=False,
    )
    label_map = {
        (row.user_id, row.date): {
            "scenarios": "" if pd.isna(row.scenarios) else row.scenarios,
            "insider_ids": "" if pd.isna(row.insider_ids) else row.insider_ids,
        }
        for row in ts2vec_scores.itertuples(index=False)
        if row.is_malicious_account_day == 1
    }
    score_frame["is_malicious_account_day"] = [
        int((row.user_id, row.date) in label_map)
        for row in score_frame.itertuples(index=False)
    ]
    score_frame["scenarios"] = [
        label_map.get((row.user_id, row.date), {}).get("scenarios", "")
        for row in score_frame.itertuples(index=False)
    ]
    score_frame["insider_ids"] = [
        label_map.get((row.user_id, row.date), {}).get("insider_ids", "")
        for row in score_frame.itertuples(index=False)
    ]
    score_frame.to_csv(SCORE_FILE, index=False, encoding="utf-8")

    risk = compute_risk(users, dates, anomaly_scores, active_mask)
    topk = build_topk(users, dates, risk, active_mask, label_map)
    topk.to_csv(TOPK_FILE, index=False, encoding="utf-8")
    incidents = load_incidents()
    positive_days_by_split = (
        score_frame.loc[score_frame["is_malicious_account_day"] == 1, "split"]
        .value_counts()
        .astype(int)
        .to_dict()
    )
    metrics = evaluate(
        topk,
        incidents,
        dates,
        train_end,
        val_end,
        positive_days_by_split,
    )

    ts2vec_metrics = json.loads(
        (PROCESSED / "r4.2_topk_metrics.json").read_text(encoding="utf-8")
    )["metrics"]
    comparison = {}
    for split in ("validation", "test"):
        comparison[split] = {}
        for k in K_VALUES:
            key = f"k_{k}"
            comparison[split][key] = {
                "ts2vec": ts2vec_metrics[split][key],
                "robust_zscore": metrics[split][key],
            }

    baseline_summary = {
        "score_file": str(SCORE_FILE),
        "topk_file": str(TOPK_FILE),
        "features": FEATURES,
        "top_features_per_day": TOP_FEATURE_COUNT,
        "max_abs_feature_z": MAX_ABS_Z,
        "min_personal_days": MIN_PERSONAL_DAYS,
        "personal_baseline_users": sum(
            stats["baseline_source"] == "personal"
            for stats in personal_stats.values()
        ),
        "global_fallback_users": sum(
            stats["baseline_source"] == "global_fallback"
            for stats in personal_stats.values()
        ),
        "metrics": metrics,
        "labels_used_in_baseline_construction": False,
        "labels_used_in_calibration": False,
    }
    METRICS_FILE.write_text(
        json.dumps(baseline_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    COMPARISON_FILE.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(baseline_summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
