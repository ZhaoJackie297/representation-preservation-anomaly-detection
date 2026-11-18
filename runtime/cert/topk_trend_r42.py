from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
TS2VEC_ARRAYS = PROCESSED / "r4.2_ts2vec_scores.npz"
ZSCORE_SCORES = PROCESSED / "r4.2_zscore_daily_scores.csv"
TS2VEC_SCORES = PROCESSED / "r4.2_ts2vec_daily_errors.csv"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"
RESULTS = PROCESSED / "r4.2_topk_trend.csv"
SUMMARY = PROCESSED / "r4.2_topk_trend_summary.json"

sys.path.insert(0, str(ROOT))
from risk_topk_r42 import compute_risk  # noqa: E402


K_VALUES = [1, 2, 5, 10, 20, 50, 100]
MAX_K = max(K_VALUES)


def matrix_from_daily_csv(
    path: Path,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    value_column: str,
) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["user_id", "date", value_column])
    frame["date"] = pd.to_datetime(frame["date"])
    index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    aligned = (
        frame.set_index(["user_id", "date"])[value_column]
        .reindex(index)
        .to_numpy(dtype=np.float32)
        .reshape(len(users), len(dates))
    )
    return aligned


def build_top100(
    method_name: str,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    scores: np.ndarray,
    active_mask: np.ndarray,
    label_map: dict[tuple[str, str], dict],
) -> pd.DataFrame:
    rows = []
    for time_index, date in enumerate(dates):
        active_indices = np.flatnonzero(active_mask[:, time_index])
        ordering = sorted(
            active_indices,
            key=lambda user_index: (-scores[user_index, time_index], users[user_index]),
        )
        date_text = date.strftime("%Y-%m-%d")
        for rank, user_index in enumerate(ordering[:MAX_K], start=1):
            user = users[user_index]
            label = label_map.get((user, date_text), {})
            rows.append(
                {
                    "method": method_name,
                    "date": date_text,
                    "rank": rank,
                    "user_id": user,
                    "score": float(scores[user_index, time_index]),
                    "is_malicious_account_day": int(bool(label)),
                    "scenarios": label.get("scenarios", ""),
                    "insider_ids": label.get("insider_ids", ""),
                }
            )
    return pd.DataFrame(rows)


def evaluate_curve(
    top100: pd.DataFrame,
    incidents: pd.DataFrame,
    dates: pd.DatetimeIndex,
    train_end: int,
    val_end: int,
    positive_days_by_split: dict[str, int],
) -> pd.DataFrame:
    validation_start = dates[train_end + 1].normalize()
    test_start = dates[val_end + 1].normalize()
    final_end = dates[-1].normalize()
    split_ranges = {
        "validation": (validation_start, test_start - pd.Timedelta(days=1)),
        "test": (test_start, final_end),
        "validation_plus_test": (validation_start, final_end),
    }
    incident_rows = incidents.copy()
    incident_rows["start_date"] = pd.to_datetime(
        incident_rows["start"]
    ).dt.normalize()
    incident_rows["end_date"] = pd.to_datetime(
        incident_rows["end"]
    ).dt.normalize()

    rows = []
    for method in sorted(top100["method"].unique()):
        method_top = top100[top100["method"] == method]
        for split, (split_start, split_end) in split_ranges.items():
            split_top = method_top[
                (method_top["date"] >= str(split_start.date()))
                & (method_top["date"] <= str(split_end.date()))
            ]
            split_incidents = incident_rows[
                (incident_rows["start_date"] >= split_start)
                & (incident_rows["start_date"] <= split_end)
            ]
            for k in K_VALUES:
                alerts_k = split_top[split_top["rank"] <= k]
                true_alerts = int(alerts_k["is_malicious_account_day"].sum())
                alert_slots = int(len(alerts_k))
                positives = positive_days_by_split[split]
                hits = 0
                delays = []
                for incident in split_incidents.itertuples(index=False):
                    accounts = set(incident.observed_accounts.split("|"))
                    candidate = alerts_k[alerts_k["user_id"].isin(accounts)]
                    candidate = candidate[
                        (candidate["date"] >= str(incident.start_date.date()))
                        & (candidate["date"] <= str(incident.end_date.date()))
                    ]
                    if len(candidate):
                        first_alert = pd.to_datetime(candidate["date"]).min()
                        hits += 1
                        delays.append((first_alert - incident.start_date).days)
                days = len(pd.date_range(split_start, split_end))
                rows.append(
                    {
                        "method": method,
                        "split": split,
                        "k": k,
                        "alert_slots": alert_slots,
                        "true_alert_slots": true_alerts,
                        "precision_at_k": true_alerts / alert_slots if alert_slots else 0.0,
                        "account_day_recall": true_alerts / positives if positives else 0.0,
                        "false_alert_slots": alert_slots - true_alerts,
                        "false_alerts_per_day": (
                            (alert_slots - true_alerts) / days if alert_slots else 0.0
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


def main() -> None:
    dataset = np.load(DATASET, allow_pickle=False)
    ts2vec_arrays = np.load(TS2VEC_ARRAYS, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])

    ts2vec_raw = ts2vec_arrays["anomaly_score"].astype(np.float32)
    zscore_raw = matrix_from_daily_csv(
        ZSCORE_SCORES, users, dates, "robust_z_score"
    )

    ts2vec_risk = compute_risk(users, dates, ts2vec_raw, active_mask)
    zscore_risk = compute_risk(users, dates, zscore_raw, active_mask)

    label_source = pd.read_csv(
        TS2VEC_SCORES,
        usecols=[
            "user_id",
            "date",
            "split",
            "is_malicious_account_day",
            "scenarios",
            "insider_ids",
        ],
        dtype={"scenarios": "string", "insider_ids": "string"},
        low_memory=False,
    )
    label_map = {
        (row.user_id, row.date): {
            "scenarios": "" if pd.isna(row.scenarios) else row.scenarios,
            "insider_ids": "" if pd.isna(row.insider_ids) else row.insider_ids,
        }
        for row in label_source.itertuples(index=False)
        if row.is_malicious_account_day == 1
    }
    positive_counts = (
        label_source[label_source["is_malicious_account_day"] == 1]["split"]
        .value_counts()
        .astype(int)
        .to_dict()
    )
    positive_counts["validation_plus_test"] = (
        positive_counts.get("validation", 0) + positive_counts.get("test", 0)
    )

    method_scores = {
        "ts2vec_raw": ts2vec_raw,
        "ts2vec_dynamic": ts2vec_risk,
        "zscore_raw": zscore_raw,
        "zscore_dynamic": zscore_risk,
    }
    top_frames = [
        build_top100(
            method, users, dates, scores, active_mask, label_map
        )
        for method, scores in method_scores.items()
    ]
    top100 = pd.concat(top_frames, ignore_index=True)
    incidents = pd.read_csv(INCIDENTS)
    results = evaluate_curve(
        top100,
        incidents,
        dates,
        train_end,
        val_end,
        positive_counts,
    )
    results.to_csv(RESULTS, index=False, encoding="utf-8")
    summary = {
        "result_file": str(RESULTS),
        "methods": list(method_scores),
        "k_values": K_VALUES,
        "risk_parameters": {
            "window_days": 7,
            "decay": 0.65,
            "threshold": 3.0,
            "severity_weight": 0.7,
            "memory_weight": 0.2,
            "frequency_weight": 0.1,
        },
        "split_incidents": results.groupby("split")["incidents_in_split"].first().to_dict(),
        "labels_used_for_parameters": False,
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(
        results[
            results["split"].isin(["validation", "test"])
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
