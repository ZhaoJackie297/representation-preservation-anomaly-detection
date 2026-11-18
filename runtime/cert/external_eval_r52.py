from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r52"
DATASET = PROCESSED / "r5.2_sequence_dataset.npz"
FEATURE_CSV = PROCESSED / "r5.2_user_day_features.csv"
LABELS = PROCESSED / "r5.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r5.2_incidents.csv"
TS2VEC_ARRAYS = PROCESSED / "r5.2_ts2vec_scores.npz"

IFOREST_MODEL = PROCESSED / "r5.2_isolation_forest.joblib"
ALERTS_FILE = PROCESSED / "r5.2_external_alerts.csv"
HITS_FILE = PROCESSED / "r5.2_external_hits.csv"
RESULTS_FILE = PROCESSED / "r5.2_external_results.json"
SUMMARY_FILE = PROCESSED / "r5.2_external_summary.json"

PRIMARY_K = 10
REPORT_K = [5, 10, 20]
COOLDOWN_DAYS = 7
HORIZONS = [1, 3]
MIN_PERSONAL_DAYS = 30
MAX_ABS_Z = 20.0

sys.path.insert(0, str(ROOT))
from baseline_isolation_forest_r42 import (  # noqa: E402
    FEATURES,
    LOG1P_FEATURES,
)
from baseline_zscore_r42 import robust_location_scale  # noqa: E402
from deep_compare_per_user_r42 import (  # noqa: E402
    per_user_quantile_scores,
)
from dual_baseline_r42 import build_group_baseline, calibrate  # noqa: E402
from risk_topk_r42 import compute_risk  # noqa: E402


def build_external_alerts(
    scores: np.ndarray,
    active_mask: np.ndarray,
    start_index: int,
    k: int,
    cooldown_days: int,
) -> np.ndarray:
    alerts = np.zeros(scores.shape, dtype=np.uint8)
    last_alert = np.full(scores.shape[0], -10_000, dtype=np.int32)
    for time_index in range(start_index, scores.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= cooldown_days
        )
        indices = np.flatnonzero(available)
        if len(indices) == 0:
            continue
        ordering = np.lexsort(
            (indices, -scores[indices, time_index])
        )
        selected = indices[ordering[:k]]
        alerts[selected, time_index] = 1
        last_alert[selected] = time_index
    return alerts


def personal_zscore(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
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

    deviations = np.full(
        values.shape[:2], np.nan, dtype=np.float32
    )
    for user_index in range(values.shape[0]):
        active_indices = np.flatnonzero(train_active[user_index])
        if len(active_indices) >= MIN_PERSONAL_DAYS:
            median, scale = robust_location_scale(
                train_slice[user_index, active_indices]
            )
            median = np.where(
                np.isfinite(median), median, global_median
            )
            scale = np.where(
                np.isfinite(scale) & (scale >= 1e-8),
                scale,
                global_scale,
            )
        else:
            median, scale = global_median, global_scale

        for time_index in np.flatnonzero(active_mask[user_index]):
            z = np.abs(
                (values[user_index, time_index] - median) / scale
            )
            z = np.clip(z, 0.0, MAX_ABS_Z)
            deviations[user_index, time_index] = np.sort(z)[-3:].mean()

    train_values = deviations[:, : train_end + 1][train_active]
    train_values = train_values[np.isfinite(train_values)]
    median, mad = robust_location_scale(
        train_values.reshape(-1, 1)
    )
    scale = 1.4826 * mad[0]
    if scale < 1e-8:
        scale = float(np.std(train_values))
    if scale < 1e-8:
        scale = 1.0
    median_value = float(np.asarray(median).reshape(-1)[0])
    scale_value = float(np.asarray(scale).reshape(-1)[0])
    return np.clip(
        (deviations - median_value) / scale_value,
        0.0,
        None,
    ).astype(np.float32)


def iforest_score(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, IsolationForest, int]:
    train_active = active_mask[:, : train_end + 1]
    train_features = values[:, : train_end + 1][train_active]
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (values - mean) / std
    train_normalized = normalized[:, : train_end + 1][train_active]

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=20261109,
    )
    model.fit(train_normalized)
    raw_scores = np.full(
        active_mask.shape, np.nan, dtype=np.float32
    )
    raw_scores[active_mask] = -model.score_samples(
        normalized[active_mask]
    )
    return raw_scores, model, len(train_features)


def evaluate_external(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    alerts: np.ndarray,
    labels: np.ndarray,
    incidents: pd.DataFrame,
    start_index: int,
    method: str,
    k: int,
) -> dict:
    end_index = len(dates) - 1
    date_mask = np.zeros(len(dates), dtype=bool)
    date_mask[start_index:] = True
    split_alerts = alerts[:, date_mask]
    split_labels = labels[:, date_mask]
    alert_slots = int(split_alerts.sum())
    true_alerts = int((split_alerts & split_labels).sum())
    day_count = int(date_mask.sum())
    positive_days = int(split_labels.sum())

    evaluation_start = dates[start_index].normalize()
    evaluation_end = dates[end_index].normalize()
    selected_incidents = incidents[
        (incidents["first_observed_date"] >= evaluation_start)
        & (incidents["first_observed_date"] <= evaluation_end)
    ]
    user_index = {
        user: index for index, user in enumerate(users)
    }
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    missing_accounts = sorted(
        {
            account
            for accounts in selected_incidents["observed_accounts"]
            for account in accounts.split("|")
            if account not in user_index
        }
    )
    if missing_accounts:
        raise RuntimeError(
            f"Incident accounts absent from feature users: "
            f"{missing_accounts[:20]}"
        )

    horizon_results = {}
    for horizon in HORIZONS:
        hit_ids = set()
        contact_times = []
        for incident in selected_incidents.itertuples(index=False):
            accounts = [
                user_index[account]
                for account in incident.observed_accounts.split("|")
                if account in user_index
            ]
            if not accounts:
                continue
            candidate_dates = pd.date_range(
                incident.first_observed_date
                - pd.Timedelta(days=COOLDOWN_DAYS - 1),
                min(
                    incident.last_observed_date,
                    incident.first_observed_date
                    + pd.Timedelta(days=horizon - 1),
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
                hit_ids.add(str(incident.incident_id))
                contact_times.append(
                    (
                        found_date
                        - incident.first_observed_date
                    ).days
                )
        horizon_results[str(horizon)] = {
            "incidents_hit": len(hit_ids),
            "incidents_total": int(len(selected_incidents)),
            "event_recall": (
                len(hit_ids) / len(selected_incidents)
                if len(selected_incidents)
                else 0.0
            ),
            "median_contact_time_days": (
                float(np.median(contact_times))
                if contact_times
                else None
            ),
            "pre_event_contacts": sum(
                value < 0 for value in contact_times
            ),
            "incident_ids": sorted(hit_ids),
        }

    return {
        "method": method,
        "k": k,
        "cooldown_days": COOLDOWN_DAYS,
        "external_test": {
            "start": str(evaluation_start.date()),
            "end": str(evaluation_end.date()),
            "days": day_count,
            "alert_slots": alert_slots,
            "true_alert_slots": true_alerts,
            "precision_at_k": (
                true_alerts / alert_slots if alert_slots else 0.0
            ),
            "account_day_recall": (
                true_alerts / positive_days if positive_days else 0.0
            ),
            "false_alert_slots": alert_slots - true_alerts,
            "false_alerts_per_day": (
                (alert_slots - true_alerts) / day_count
                if day_count
                else 0.0
            ),
            "horizons": horizon_results,
        },
    }


def main() -> None:
    with np.load(DATASET, allow_pickle=False) as dataset:
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active_mask = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
    test_start = train_end + 1

    raw = pd.read_csv(FEATURE_CSV, parse_dates=["date"])
    aligned_index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    aligned = raw.set_index(["user_id", "date"]).reindex(
        aligned_index
    )
    values = aligned[FEATURES].to_numpy(
        dtype=np.float64
    ).reshape(len(users), len(dates), len(FEATURES))
    for feature_index, feature in enumerate(FEATURES):
        if feature in LOG1P_FEATURES:
            finite = np.isfinite(values[:, :, feature_index])
            values[:, :, feature_index][finite] = np.log1p(
                values[:, :, feature_index][finite]
            )

    functional_units = (
        aligned["functional_unit"]
        .fillna("Unknown")
        .to_numpy()
        .reshape(len(users), len(dates))
    )
    roles = (
        aligned["role"]
        .fillna("Unknown")
        .to_numpy()
        .reshape(len(users), len(dates))
    )

    print("Building personal robust baseline...", flush=True)
    zscore_raw = personal_zscore(values, active_mask, train_end)
    zscore_dynamic = compute_risk(
        users, dates, zscore_raw, active_mask
    )

    print("Training Isolation Forest...", flush=True)
    iforest_raw, iforest_model, iforest_training_rows = iforest_score(
        values, active_mask, train_end
    )
    joblib.dump(iforest_model, IFOREST_MODEL)
    iforest_quantile = per_user_quantile_scores(
        iforest_raw, active_mask, train_end
    )
    iforest_dynamic = compute_risk(
        users, dates, iforest_quantile, active_mask
    )

    print("Building peer-group baseline...", flush=True)
    group_raw, peer_levels, _ = build_group_baseline(
        values,
        users,
        dates,
        active_mask,
        functional_units,
        roles,
        train_end,
    )
    group_score = calibrate(
        group_raw[:, : train_end + 1][
            active_mask[:, : train_end + 1]
        ],
        group_raw,
    )

    with np.load(TS2VEC_ARRAYS, allow_pickle=False) as ts_arrays:
        if not np.array_equal(ts_arrays["users"].astype(str), users):
            raise RuntimeError(
                "TS2Vec users do not match sequence users."
            )
        if not np.array_equal(
            pd.to_datetime(ts_arrays["dates"].astype(str)), dates
        ):
            raise RuntimeError(
                "TS2Vec dates do not match sequence dates."
            )
        ts_raw = ts_arrays["anomaly_score"].astype(np.float32)
    ts_quantile = per_user_quantile_scores(
        ts_raw, active_mask, train_end
    )
    ts2vec_dynamic = compute_risk(
        users, dates, ts_quantile, active_mask
    )

    methods = {
        "ts2vec_personal_quantile_dynamic": ts2vec_dynamic,
        "robust_zscore_global_dynamic": zscore_dynamic,
        "isolation_forest_personal_quantile_dynamic": iforest_dynamic,
        "group_baseline_training_calibrated_raw": group_score,
    }
    for name, score in methods.items():
        active_external = active_mask[:, test_start:]
        if np.isnan(score[:, test_start:][active_external]).any():
            raise RuntimeError(
                f"Non-finite external scores for {name}"
            )

    labels_frame = pd.read_csv(LABELS, parse_dates=["date"])
    user_index = {
        user: index for index, user in enumerate(users)
    }
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    labels = np.zeros((len(users), len(dates)), dtype=np.uint8)
    for row in labels_frame.itertuples(index=False):
        user_key = user_index.get(row.user_id)
        date_key = date_index.get(row.date.strftime("%Y-%m-%d"))
        if user_key is not None and date_key is not None:
            labels[user_key, date_key] = 1
    if int(labels[:, :test_start].sum()) != 0:
        raise RuntimeError(
            "Malicious account-days leaked into external training."
        )

    incidents = pd.read_csv(
        INCIDENTS,
        parse_dates=[
            "first_observed_date",
            "last_observed_date",
        ],
    )
    external_incident_count = int(
        (
            incidents["first_observed_date"]
            >= dates[test_start].normalize()
        ).sum()
    )
    if external_incident_count != len(incidents):
        raise RuntimeError(
            "Some incidents start before the clean external cutoff."
        )
    alerts_rows = []
    hit_rows = []
    results = []
    for method, score in methods.items():
        for k in REPORT_K:
            alerts = build_external_alerts(
                score, active_mask, test_start, k, COOLDOWN_DAYS
            )
            row_users, row_times = np.nonzero(alerts)
            for user_index_value, time_index_value in zip(
                row_users, row_times
            ):
                alerts_rows.append(
                    {
                        "method": method,
                        "k": k,
                        "date": dates[time_index_value].strftime(
                            "%Y-%m-%d"
                        ),
                        "user_id": users[user_index_value],
                    }
                )
            result = evaluate_external(
                users,
                dates,
                active_mask,
                alerts,
                labels,
                incidents,
                test_start,
                method,
                k,
            )
            results.append(result)
            for horizon, details in result["external_test"][
                "horizons"
            ].items():
                for incident_id in details["incident_ids"]:
                    hit_rows.append(
                        {
                            "method": method,
                            "k": k,
                            "horizon_days": int(horizon),
                            "incident_id": incident_id,
                        }
                    )

    pd.DataFrame(alerts_rows).to_csv(
        ALERTS_FILE, index=False, encoding="utf-8"
    )
    pd.DataFrame(hit_rows).to_csv(
        HITS_FILE, index=False, encoding="utf-8"
    )
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "dataset_version": "5.2",
        "evaluation_design": (
            "One-shot external replication after a globally clean "
            "pre-incident cutoff; no validation or test-based selection."
        ),
        "train_end_date": str(dates[train_end].date()),
        "external_test_start": str(dates[test_start].date()),
        "external_test_end": str(dates[-1].date()),
        "external_incident_count": int(len(incidents)),
        "training_malicious_account_days": int(
            labels[:, :test_start].sum()
        ),
        "external_malicious_account_days": int(
            labels[:, test_start:].sum()
        ),
        "primary_k": PRIMARY_K,
        "cooldown_days": COOLDOWN_DAYS,
        "report_k": REPORT_K,
        "event_anchor": (
            "First official malicious log date; not the answer-file "
            "scenario metadata start."
        ),
        "pre_event_contact_days": COOLDOWN_DAYS - 1,
        "methods": {
            "ts2vec": "personal training quantile + dynamic risk",
            "robust_zscore": "global robust calibration + dynamic risk",
            "isolation_forest": (
                "personal training quantile + dynamic risk"
            ),
            "group_baseline": "training calibration + raw score",
        },
        "peer_level_rows": {
            str(level): int(count)
            for level, count in zip(
                *np.unique(
                    peer_levels[active_mask],
                    return_counts=True,
                )
            )
        },
        "iforest_training_rows": iforest_training_rows,
        "labels_used_for_model_fit": False,
        "labels_used_for_calibration": False,
        "external_test_used_for_selection": False,
        "alerts_file": str(ALERTS_FILE),
        "hits_file": str(HITS_FILE),
        "results_file": str(RESULTS_FILE),
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
