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
PERSONAL_SCORES = PROCESSED / "r4.2_zscore_daily_scores.csv"
LABELS = PROCESSED / "r4.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"

GROUP_SCORE_FILE = PROCESSED / "r4.2_group_baseline_daily_scores.csv"
DUAL_SCORE_FILE = PROCESSED / "r4.2_dual_baseline_daily_scores.csv"
SELECTION_FILE = PROCESSED / "r4.2_dual_baseline_selection.csv"
RESULTS_FILE = PROCESSED / "r4.2_dual_baseline_results.json"
SUMMARY_FILE = PROCESSED / "r4.2_dual_baseline_summary.json"

sys.path.insert(0, str(ROOT))
from risk_topk_r42 import compute_risk  # noqa: E402
from production_protocol_r42 import (  # noqa: E402
    COOLDOWN_DAYS,
    HORIZONS,
    PRIMARY_K,
    REPORT_K,
    build_alerts,
    evaluate,
    label_matrix,
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
LOG1P_FEATURES = FEATURES[:2] + FEATURES[5:]
MIN_PEER_COUNT = 10
MAX_FEATURE_Z = 100.0


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    scale = 1.4826 * mad
    return median, scale


def calibrate(train_values: np.ndarray, all_values: np.ndarray) -> np.ndarray:
    median, mad = robust_location_scale(train_values.reshape(-1, 1))
    median_value = float(np.asarray(median).reshape(-1)[0])
    scale = float(np.asarray(1.4826 * mad).reshape(-1)[0])
    if scale < 1e-8:
        scale = float(np.std(train_values))
    if scale < 1e-8:
        scale = 1.0
    return np.clip((all_values - median_value) / scale, 0.0, None)


def build_group_baseline(
    values: np.ndarray,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    functional_units: np.ndarray,
    roles: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_active = active_mask[:, : train_end + 1]
    global_median, global_scale = robust_location_scale(
        values[:, : train_end + 1][train_active]
    )
    global_scale = np.where(
        (global_scale < 1e-8) | ~np.isfinite(global_scale),
        1.0,
        global_scale,
    )

    raw_scores = np.full(values.shape[:2], np.nan, dtype=np.float32)
    peer_levels = np.full(values.shape[:2], 0, dtype=np.int8)
    peer_sizes = np.zeros(values.shape[:2], dtype=np.int16)

    for time_index in range(len(dates)):
        active_users = np.flatnonzero(active_mask[:, time_index])
        if len(active_users) == 0:
            continue
        day_median, day_scale = robust_location_scale(
            values[active_users, time_index]
        )
        day_scale = np.where(
            np.isfinite(day_scale) & (day_scale >= 1e-8),
            day_scale,
            global_scale,
        )
        assigned = np.zeros(len(users), dtype=bool)

        def assign_groups(group_labels: np.ndarray, level: int) -> None:
            for group in np.unique(group_labels[active_users]):
                label = str(group)
                if label == "Unknown" or label == "nan":
                    continue
                group_users = active_users[group_labels[active_users] == group]
                if len(group_users) < MIN_PEER_COUNT:
                    continue
                group_users = group_users[~assigned[group_users]]
                if len(group_users) == 0:
                    continue
                group_values = values[group_users, time_index]
                median = np.nanmedian(group_values, axis=0)
                mad = np.nanmedian(np.abs(group_values - median), axis=0)
                scale = 1.4826 * mad
                scale = np.where(
                    np.isfinite(scale) & (scale >= 1e-8),
                    scale,
                    global_scale,
                )
                z = np.abs((group_values - median) / scale)
                z = np.clip(z, 0.0, MAX_FEATURE_Z)
                raw_scores[group_users, time_index] = np.sort(z, axis=1)[:, -3:].mean(axis=1)
                peer_levels[group_users, time_index] = level
                peer_sizes[group_users, time_index] = int(len(group_users))
                assigned[group_users] = True

        assign_groups(functional_units[:, time_index], 1)
        assign_groups(roles[:, time_index], 2)

        remaining = active_users[~assigned[active_users]]
        if len(remaining):
            z = np.abs((values[remaining, time_index] - day_median) / day_scale)
            z = np.clip(z, 0.0, MAX_FEATURE_Z)
            raw_scores[remaining, time_index] = np.sort(z, axis=1)[:, -3:].mean(axis=1)
            peer_levels[remaining, time_index] = 3
            peer_sizes[remaining, time_index] = len(active_users)
    return raw_scores, peer_levels, peer_sizes


def main() -> None:
    dataset = np.load(DATASET, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])

    raw = pd.read_csv(FEATURE_CSV, parse_dates=["date"])
    index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    aligned = raw.set_index(["user_id", "date"]).reindex(index)
    values = aligned[FEATURES].to_numpy(dtype=np.float64).reshape(
        len(users), len(dates), len(FEATURES)
    )
    functional_units = aligned["functional_unit"].fillna("Unknown").to_numpy().reshape(
        len(users), len(dates)
    )
    roles = aligned["role"].fillna("Unknown").to_numpy().reshape(
        len(users), len(dates)
    )
    for feature_index, feature in enumerate(FEATURES):
        if feature in LOG1P_FEATURES:
            mask = np.isfinite(values[:, :, feature_index])
            values[:, :, feature_index][mask] = np.log1p(
                values[:, :, feature_index][mask]
            )

    print("Building peer-group baseline...", flush=True)
    group_raw, peer_levels, peer_sizes = build_group_baseline(
        values,
        users,
        dates,
        active_mask,
        functional_units,
        roles,
        train_end,
    )
    train_active = active_mask[:, : train_end + 1]
    group_score = calibrate(group_raw[:, : train_end + 1][train_active], group_raw)

    personal_frame = pd.read_csv(
        PERSONAL_SCORES,
        usecols=["user_id", "date", "robust_z_score"],
    )
    personal_frame["date"] = pd.to_datetime(personal_frame["date"])
    personal_score = (
        personal_frame.set_index(["user_id", "date"])["robust_z_score"]
        .reindex(index)
        .to_numpy(dtype=np.float32)
        .reshape(len(users), len(dates))
    )

    group_rows = []
    for user_index, user in enumerate(users):
        for time_index, date in enumerate(dates):
            if not active_mask[user_index, time_index]:
                continue
            group_rows.append(
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
                    "group_raw_score": float(group_raw[user_index, time_index]),
                    "group_score": float(group_score[user_index, time_index]),
                    "peer_level": int(peer_levels[user_index, time_index]),
                    "peer_count": int(peer_sizes[user_index, time_index]),
                }
            )
    group_frame = pd.DataFrame(group_rows)
    group_frame.to_csv(GROUP_SCORE_FILE, index=False, encoding="utf-8")

    dual_mean = (personal_score + group_score) / 2.0
    dual_max = np.maximum(personal_score, group_score)
    families = {
        "personal": personal_score,
        "group": group_score,
        "dual_mean": dual_mean,
        "dual_max": dual_max,
    }
    variants = {}
    for family, score in families.items():
        variants[f"{family}_raw"] = score.astype(np.float32)
        variants[f"{family}_dynamic"] = compute_risk(
            users, dates, score.astype(np.float32), active_mask
        )

    labels = label_matrix(users, dates)
    incidents = pd.read_csv(INCIDENTS)
    validation_rows = []
    selected = {}
    for family in families:
        candidates = []
        for aggregation in ("raw", "dynamic"):
            variant = f"{family}_{aggregation}"
            alerts = build_alerts(
                variants[variant], active_mask, PRIMARY_K, COOLDOWN_DAYS
            )
            result = evaluate(
                users,
                dates,
                active_mask,
                alerts,
                labels,
                incidents,
                train_end,
                val_end,
                variant,
                PRIMARY_K,
            )
            validation = result["splits"]["validation"]
            candidates.append(
                {
                    "family": family,
                    "variant": variant,
                    "validation_recall_3d": validation["horizons"]["3"]["event_recall"],
                    "validation_recall_1d": validation["horizons"]["1"]["event_recall"],
                    "validation_precision": validation["precision_at_k"],
                    "validation_false_per_day": validation["false_alerts_per_day"],
                }
            )
        candidates = sorted(
            candidates,
            key=lambda row: (
                -row["validation_recall_3d"],
                -row["validation_recall_1d"],
                -row["validation_precision"],
                row["validation_false_per_day"],
            ),
        )
        selected[family] = candidates[0]["variant"]
        validation_rows.extend(candidates)
    pd.DataFrame(validation_rows).to_csv(SELECTION_FILE, index=False, encoding="utf-8")

    results = []
    for family, variant in selected.items():
        for k in REPORT_K:
            alerts = build_alerts(
                variants[variant], active_mask, k, COOLDOWN_DAYS
            )
            result = evaluate(
                users,
                dates,
                active_mask,
                alerts,
                labels,
                incidents,
                train_end,
                val_end,
                variant,
                k,
            )
            result["family"] = family
            result["base_score"] = variant
            results.append(result)
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dual_frame = pd.DataFrame(
        {
            "user_id": np.repeat(users, len(dates)),
            "date": np.tile(dates.astype(str), len(users)),
            "personal_score": personal_score.reshape(-1),
            "group_score": group_score.reshape(-1),
            "dual_mean": dual_mean.reshape(-1),
            "dual_max": dual_max.reshape(-1),
            "active": active_mask.reshape(-1).astype(np.uint8),
        }
    )
    dual_frame = dual_frame[dual_frame["active"] == 1]
    dual_frame.to_csv(DUAL_SCORE_FILE, index=False, encoding="utf-8")

    peer_distribution = (
        group_frame.groupby("peer_level").size().rename("rows").to_dict()
    )
    summary = {
        "peer_group_rule": {
            "primary": "functional_unit",
            "fallback_1": "role",
            "fallback_2": "all active users on the same day",
            "minimum_functional_or_role_group_size": MIN_PEER_COUNT,
        },
        "peer_level_rows": peer_distribution,
        "selected_variants": selected,
        "selection_file": str(SELECTION_FILE),
        "results_file": str(RESULTS_FILE),
        "group_score_file": str(GROUP_SCORE_FILE),
        "dual_score_file": str(DUAL_SCORE_FILE),
        "labels_used_for_scores": False,
        "risk_parameters_tuned_on_labels": False,
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
