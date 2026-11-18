from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
CALIBRATED = PROCESSED / "r4.2_calibrated_method_scores.npz"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
TS2VEC_ARRAYS = PROCESSED / "r4.2_ts2vec_scores.npz"
ZSCORE_DAILY = PROCESSED / "r4.2_zscore_daily_scores.csv"
IFOREST_ARRAYS = PROCESSED / "r4.2_iforest_raw_scores.npz"
LABELS = PROCESSED / "r4.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"
ALERT_FILE = PROCESSED / "r4.2_production_alerts.csv"
RESULTS_FILE = PROCESSED / "r4.2_production_protocol_results.json"
SELECTION_FILE = PROCESSED / "r4.2_production_variant_selection.csv"
SUMMARY_FILE = PROCESSED / "r4.2_production_protocol_summary.json"

PRIMARY_K = 10
COOLDOWN_DAYS = 7
REPORT_K = [5, 10, 20]
HORIZONS = [1, 3]

sys.path.insert(0, str(ROOT))
from risk_topk_r42 import compute_risk  # noqa: E402


def build_alerts(
    scores: np.ndarray,
    active_mask: np.ndarray,
    k: int,
    cooldown_days: int,
) -> np.ndarray:
    alerts = np.zeros(scores.shape, dtype=np.uint8)
    last_alert = np.full(scores.shape[0], -10_000, dtype=np.int32)
    for time_index in range(scores.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= cooldown_days
        )
        indices = np.flatnonzero(available)
        if len(indices) == 0:
            continue
        ordering = np.lexsort((indices, -scores[indices, time_index]))
        selected = indices[ordering[:k]]
        alerts[selected, time_index] = 1
        last_alert[selected] = time_index
    return alerts


def label_matrix(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)}
    frame = pd.read_csv(LABELS, parse_dates=["date"])
    matrix = np.zeros((len(users), len(dates)), dtype=np.uint8)
    for row in frame.itertuples(index=False):
        key_date = row.date.strftime("%Y-%m-%d")
        if row.user_id in user_index and key_date in date_index:
            matrix[user_index[row.user_id], date_index[key_date]] = 1
    return matrix


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


def evaluate(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    alerts: np.ndarray,
    labels: np.ndarray,
    incidents: pd.DataFrame,
    train_end: int,
    val_end: int,
    method: str,
    k: int,
) -> dict:
    validation_start = dates[train_end + 1].normalize()
    test_start = dates[val_end + 1].normalize()
    final_end = dates[-1].normalize()
    split_ranges = {
        "validation": (validation_start, test_start - pd.Timedelta(days=1)),
        "test": (test_start, final_end),
    }
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)}

    incident_rows = incidents.copy()
    incident_rows["start_date"] = pd.to_datetime(
        incident_rows["start"]
    ).dt.normalize()
    incident_rows["end_date"] = pd.to_datetime(
        incident_rows["end"]
    ).dt.normalize()

    output = {
        "method": method,
        "k": k,
        "cooldown_days": COOLDOWN_DAYS,
        "splits": {},
    }
    for split, (split_start, split_end) in split_ranges.items():
        date_mask = (dates >= split_start) & (dates <= split_end)
        split_alerts = alerts[:, date_mask]
        split_labels = labels[:, date_mask]
        alert_slots = int(split_alerts.sum())
        true_alerts = int((split_alerts & split_labels).sum())
        day_count = int(date_mask.sum())
        positive_days = int(split_labels.sum())
        split_incidents = incident_rows[
            (incident_rows["start_date"] >= split_start)
            & (incident_rows["start_date"] <= split_end)
        ]

        horizon_results = {}
        for horizon in HORIZONS:
            hits = 0
            contact_times = []
            for incident in split_incidents.itertuples(index=False):
                accounts = [
                    user_index[account]
                    for account in incident.observed_accounts.split("|")
                    if account in user_index
                ]
                candidate_dates = pd.date_range(
                    incident.start_date - pd.Timedelta(days=COOLDOWN_DAYS - 1),
                    min(
                        incident.end_date,
                        incident.start_date + pd.Timedelta(days=horizon - 1),
                    ),
                    freq="D",
                )
                found_date = None
                for date in candidate_dates:
                    date_text = date.strftime("%Y-%m-%d")
                    if date_text not in date_index:
                        continue
                    time_index = date_index[date_text]
                    if alerts[np.asarray(accounts), time_index].any():
                        found_date = date.normalize()
                        break
                if found_date is not None:
                    hits += 1
                    contact_times.append(
                        (found_date - incident.start_date).days
                    )
            horizon_results[str(horizon)] = {
                "incidents_hit": hits,
                "incidents_total": int(len(split_incidents)),
                "event_recall": (
                    hits / len(split_incidents) if len(split_incidents) else 0.0
                ),
                "median_contact_time_days": (
                    float(np.median(contact_times)) if contact_times else None
                ),
                "pre_event_contacts": sum(value < 0 for value in contact_times),
            }

        output["splits"][split] = {
            "alert_slots": alert_slots,
            "true_alert_slots": true_alerts,
            "precision_at_k": true_alerts / alert_slots if alert_slots else 0.0,
            "account_day_recall": true_alerts / positive_days if positive_days else 0.0,
            "false_alert_slots": alert_slots - true_alerts,
            "false_alerts_per_day": (
                (alert_slots - true_alerts) / day_count if day_count else 0.0
            ),
            "horizons": horizon_results,
        }
    return output


def main() -> None:
    dataset = np.load(DATASET, allow_pickle=False)
    calibrated = np.load(CALIBRATED, allow_pickle=False)
    ts2vec_arrays = np.load(TS2VEC_ARRAYS, allow_pickle=False)
    iforest_arrays = np.load(IFOREST_ARRAYS, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    labels = label_matrix(users, dates)
    incidents = pd.read_csv(INCIDENTS)

    global_raw = {
        "ts2vec": ts2vec_arrays["anomaly_score"].astype(np.float32),
        "zscore": matrix_from_csv(
            ZSCORE_DAILY, users, dates, "robust_z_score"
        ),
        "iforest": iforest_arrays["iforest_score"].astype(np.float32),
    }
    variants = {}
    for base, score in global_raw.items():
        variants[f"{base}_global_raw"] = score
        variants[f"{base}_global_dynamic"] = compute_risk(
            users, dates, score, active_mask
        )
        variants[f"{base}_personal_quantile_raw"] = calibrated[
            f"{base}_quantile"
        ].astype(np.float32)
        variants[f"{base}_personal_quantile_dynamic"] = calibrated[
            f"{base}_quantile_dynamic"
        ].astype(np.float32)

    validation_rows = []
    selected = {}
    for base in ("ts2vec", "zscore", "iforest"):
        candidates = []
        for calibration_name in ("global", "personal_quantile"):
            for aggregation_name in ("raw", "dynamic"):
                variant = f"{base}_{calibration_name}_{aggregation_name}"
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
                recall_3 = validation["horizons"]["3"]["event_recall"]
                recall_1 = validation["horizons"]["1"]["event_recall"]
                precision = validation["precision_at_k"]
                candidates.append(
                    {
                        "base_method": base,
                        "variant": variant,
                        "k": PRIMARY_K,
                        "validation_recall_3d": recall_3,
                        "validation_recall_1d": recall_1,
                        "validation_precision": precision,
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
        selected[base] = candidates[0]["variant"]
        validation_rows.extend(candidates)
    pd.DataFrame(validation_rows).to_csv(
        SELECTION_FILE, index=False, encoding="utf-8"
    )

    all_alert_frames = []
    results = []
    for base, variant in selected.items():
        for k in REPORT_K:
            alerts = build_alerts(
                variants[variant], active_mask, k, COOLDOWN_DAYS
            )
            alert_rows, alert_time_indices = np.nonzero(alerts)
            for user_index, time_index in zip(alert_rows, alert_time_indices):
                all_alert_frames.append(
                    {
                        "base_method": base,
                        "variant": variant,
                        "k": k,
                        "date": dates[time_index].strftime("%Y-%m-%d"),
                        "user_id": users[user_index],
                    }
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
            result["base_method"] = base
            results.append(result)

    pd.DataFrame(all_alert_frames).to_csv(
        ALERT_FILE, index=False, encoding="utf-8"
    )
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected_primary = [
        result for result in results if result["k"] == PRIMARY_K
    ]
    comparison = {}
    for result in selected_primary:
        comparison[result["base_method"]] = {
            "variant": result["method"],
            "validation": result["splits"]["validation"],
            "test": result["splits"]["test"],
        }
    summary = {
        "protocol": {
            "primary_daily_new_alerts": PRIMARY_K,
            "cooldown_days": COOLDOWN_DAYS,
            "primary_horizon_days": 3,
            "secondary_horizon_days": 1,
            "report_k_values": REPORT_K,
            "configuration_selected_on_validation": True,
            "configuration": selected,
        },
        "comparison": comparison,
        "alert_file": str(ALERT_FILE),
        "result_file": str(RESULTS_FILE),
        "selection_file": str(SELECTION_FILE),
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
