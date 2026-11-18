from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
CALIBRATED = PROCESSED / "r4.2_calibrated_method_scores.npz"
GROUP_SCORES = PROCESSED / "r4.2_group_baseline_daily_scores.csv"
ZSCORE_DAILY = PROCESSED / "r4.2_zscore_daily_scores.csv"
LABELS = PROCESSED / "r4.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"

SELECTION_FILE = PROCESSED / "r4.2_fusion_selection.csv"
RESULTS_FILE = PROCESSED / "r4.2_fusion_results.json"
SUMMARY_FILE = PROCESSED / "r4.2_fusion_summary.json"
VALIDATION_HITS_FILE = PROCESSED / "r4.2_fusion_validation_hits.csv"

sys.path.insert(0, str(ROOT))
from risk_topk_r42 import compute_risk  # noqa: E402
from production_protocol_r42 import (  # noqa: E402
    COOLDOWN_DAYS,
    PRIMARY_K,
    REPORT_K,
    build_alerts,
    evaluate,
    label_matrix,
)


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


def daily_percentile(
    scores: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    percentiles = np.full(scores.shape, np.nan, dtype=np.float32)
    for time_index in range(scores.shape[1]):
        active = np.flatnonzero(active_mask[:, time_index])
        if len(active) <= 1:
            percentiles[active, time_index] = 1.0
            continue
        values = scores[active, time_index]
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite fusion input at time {time_index}")
        ranks = rankdata(values, method="average")
        percentiles[active, time_index] = (
            (ranks - 1.0) / (len(active) - 1.0)
        )
    return percentiles


def incident_hit_ids(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    alerts: np.ndarray,
    incidents: pd.DataFrame,
    start_index: int,
    end_index: int,
    horizon_days: int,
) -> set[str]:
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)
    }
    start_date = dates[start_index].normalize()
    end_date = dates[end_index].normalize()
    split_incidents = incidents[
        (incidents["start_date"] >= start_date)
        & (incidents["start_date"] <= end_date)
    ]
    hit_ids = set()
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
                incident.start_date + pd.Timedelta(days=horizon_days - 1),
            ),
            freq="D",
        )
        for date in candidate_dates:
            date_text = date.strftime("%Y-%m-%d")
            if date_text not in date_index:
                continue
            time_index = date_index[date_text]
            if alerts[np.asarray(accounts), time_index].any():
                hit_ids.add(str(incident.incident_id))
                break
    return hit_ids


def main() -> None:
    dataset = np.load(DATASET, allow_pickle=False)
    calibrated = np.load(CALIBRATED, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])

    ts2vec_signal = daily_percentile(
        calibrated["ts2vec_quantile_dynamic"].astype(np.float32),
        active_mask,
    )
    personal_score = matrix_from_csv(
        ZSCORE_DAILY, users, dates, "robust_z_score"
    )
    personal_score = compute_risk(
        users, dates, personal_score, active_mask
    )
    personal_signal = daily_percentile(
        personal_score,
        active_mask,
    )
    group_signal = daily_percentile(
        matrix_from_csv(
            GROUP_SCORES, users, dates, "group_score"
        ),
        active_mask,
    )

    combinations = {
        "ts2vec_only": ts2vec_signal,
        "personal_only": personal_signal,
        "group_only": group_signal,
        "ts2vec_personal": (ts2vec_signal + personal_signal) / 2.0,
        "personal_group": (personal_signal + group_signal) / 2.0,
        "ts2vec_personal_group": (
            ts2vec_signal + personal_signal + group_signal
        ) / 3.0,
    }
    labels = label_matrix(users, dates)
    incidents = pd.read_csv(INCIDENTS)
    incidents["start_date"] = pd.to_datetime(
        incidents["start"]
    ).dt.normalize()
    incidents["end_date"] = pd.to_datetime(
        incidents["end"]
    ).dt.normalize()

    validation_rows = []
    validation_hit_rows = []
    selected = None
    for name, score in combinations.items():
        alerts = build_alerts(score, active_mask, PRIMARY_K, COOLDOWN_DAYS)
        hit_ids = incident_hit_ids(
            users,
            dates,
            alerts,
            incidents,
            train_end + 1,
            val_end,
            horizon_days=3,
        )
        for incident_id in sorted(hit_ids):
            validation_hit_rows.append(
                {
                    "combination": name,
                    "incident_id": incident_id,
                    "hit_within_3d": 1,
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
            name,
            PRIMARY_K,
        )
        validation = result["splits"]["validation"]
        row = {
            "combination": name,
            "validation_recall_3d": validation["horizons"]["3"]["event_recall"],
            "validation_recall_1d": validation["horizons"]["1"]["event_recall"],
            "validation_precision": validation["precision_at_k"],
            "validation_false_per_day": validation["false_alerts_per_day"],
        }
        validation_rows.append(row)
        if selected is None:
            selected = row
        else:
            current_key = (
                -row["validation_recall_3d"],
                -row["validation_recall_1d"],
                -row["validation_precision"],
                row["validation_false_per_day"],
            )
            selected_key = (
                -selected["validation_recall_3d"],
                -selected["validation_recall_1d"],
                -selected["validation_precision"],
                selected["validation_false_per_day"],
            )
            if current_key < selected_key:
                selected = row
    pd.DataFrame(validation_rows).to_csv(
        SELECTION_FILE, index=False, encoding="utf-8"
    )
    pd.DataFrame(validation_hit_rows).to_csv(
        VALIDATION_HITS_FILE, index=False, encoding="utf-8"
    )

    selected_name = selected["combination"]
    results = []
    for k in REPORT_K:
        alerts = build_alerts(
            combinations[selected_name], active_mask, k, COOLDOWN_DAYS
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
            selected_name,
            k,
        )
        result["combination"] = selected_name
        results.append(result)
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selection_frame = pd.DataFrame(validation_rows)
    ts2vec_row = selection_frame.loc[
        selection_frame["combination"] == "ts2vec_only"
    ].iloc[0]
    fused_rows = selection_frame[
        ~selection_frame["combination"].isin(
            ["ts2vec_only", "personal_only", "group_only"]
        )
    ].sort_values(
        ["validation_recall_3d", "validation_recall_1d"],
        ascending=False,
    )
    best_fused = fused_rows.iloc[0]
    summary = {
        "fusion_signal": "daily cross-sectional percentile",
        "component_configurations": {
            "ts2vec": {
                "calibration": "per-user training quantile",
                "aggregation": "dynamic risk",
                "array": "ts2vec_quantile_dynamic",
            },
            "personal_baseline": {
                "calibration": "global robust z-score",
                "aggregation": "dynamic risk",
                "column": "robust_z_score",
            },
            "group_baseline": {
                "calibration": "training-period robust calibration",
                "aggregation": "raw score",
                "column": "group_score",
            },
        },
        "configuration_rule": (
            "Use each component's validation-selected production variant; "
            "do not tune component variants on the test split."
        ),
        "combination_scores": list(combinations),
        "selected_on_validation": selected_name,
        "primary_k": PRIMARY_K,
        "cooldown_days": COOLDOWN_DAYS,
        "component_config_selection_split": "validation",
        "fusion_scheme_selection_split": "validation",
        "test_used_for_selection": False,
        "best_fused_on_validation": best_fused["combination"],
        "best_fused_validation_recall_3d": float(
            best_fused["validation_recall_3d"]
        ),
        "ts2vec_validation_recall_3d": float(
            ts2vec_row["validation_recall_3d"]
        ),
        "best_fusion_gain_vs_ts2vec_3d": float(
            best_fused["validation_recall_3d"]
            - ts2vec_row["validation_recall_3d"]
        ),
        "fusion_adopted": selected_name
        not in {"ts2vec_only", "personal_only", "group_only"},
        "selection_file": str(SELECTION_FILE),
        "results_file": str(RESULTS_FILE),
        "validation_hits_file": str(VALIDATION_HITS_FILE),
        "labels_used_for_signal": False,
        "labels_used_for_candidate_selection": "validation only",
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(pd.DataFrame(validation_rows).sort_values(
        ["validation_recall_3d", "validation_recall_1d"],
        ascending=False,
    ).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
