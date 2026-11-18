from __future__ import annotations

import csv
import io
import json
import tarfile
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
SCORES = PROCESSED / "r4.2_ts2vec_daily_errors.csv"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
SCORE_ARRAYS = PROCESSED / "r4.2_ts2vec_scores.npz"
ANSWERS = ROOT / "answers.tar.bz2"
RISK_FILE = PROCESSED / "r4.2_user_day_risk.csv"
TOPK_FILE = PROCESSED / "r4.2_daily_topk_alerts.csv"
METRICS_FILE = PROCESSED / "r4.2_topk_metrics.json"
INCIDENTS_FILE = PROCESSED / "r4.2_incidents.csv"

LEGACY_WINDOW_DAYS = 7
LEGACY_DECAY = 0.65
LEGACY_ALERT_THRESHOLD = 3.0
LEGACY_SEVERITY_WEIGHT = 0.7
LEGACY_MEMORY_WEIGHT = 0.2
LEGACY_FREQUENCY_WEIGHT = 0.1
K_VALUES = [5, 10, 20]


def load_incidents() -> pd.DataFrame:
    with tarfile.open(ANSWERS, "r:bz2") as archive:
        names = set(archive.getnames())
        raw_master = archive.extractfile("answers/insiders.csv")
        if raw_master is None:
            raise RuntimeError("Missing answers/insiders.csv")
        text = io.TextIOWrapper(raw_master, encoding="utf-8-sig", newline="")
        master = [
            row
            for row in csv.DictReader(text)
            if row["dataset"].strip() == "4.2"
        ]

        rows = []
        for incident in master:
            scenario = incident["scenario"].strip()
            insider_id = incident["user"].strip()
            detail = incident["details"].strip()
            start = datetime.strptime(
                incident["start"].strip(), "%m/%d/%Y %H:%M:%S"
            )
            end = datetime.strptime(
                incident["end"].strip(), "%m/%d/%Y %H:%M:%S"
            )
            candidates = [
                f"answers/{detail}",
                f"answers/r4.2-{scenario}/{detail}",
            ]
            detail_path = next((item for item in candidates if item in names), None)
            if detail_path is None:
                raise FileNotFoundError(detail)
            raw_detail = archive.extractfile(detail_path)
            if raw_detail is None:
                raise RuntimeError(f"Cannot read {detail_path}")
            observed_accounts = set()
            event_dates = []
            event_count = 0
            detail_text = io.TextIOWrapper(
                raw_detail, encoding="utf-8-sig", newline=""
            )
            for event in csv.reader(detail_text):
                if len(event) < 4:
                    continue
                event_time = datetime.strptime(
                    event[2].strip(), "%m/%d/%Y %H:%M:%S"
                )
                observed_accounts.add(event[3].strip())
                event_dates.append(event_time.date())
                event_count += 1
            rows.append(
                {
                    "incident_id": f"{scenario}:{insider_id}:{Path(detail).stem}",
                    "scenario": scenario,
                    "insider_id": insider_id,
                    "observed_accounts": "|".join(sorted(observed_accounts)),
                    "start": start,
                    "end": end,
                    "first_observed_date": min(event_dates),
                    "last_observed_date": max(event_dates),
                    "detail_event_count": event_count,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(INCIDENTS_FILE, index=False, encoding="utf-8")
    return frame


def compute_severity_risk(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    scores: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Current S3 ordering score: calibrated severity only."""
    del users, dates
    risk = np.full(scores.shape, np.nan, dtype=np.float32)
    risk[active_mask] = scores[active_mask]
    return risk


def compute_legacy_weighted_risk(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    scores: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Archived 0.7/0.2/0.1 heuristic; not used by current S3."""
    del users, dates
    risk = np.full(scores.shape, np.nan, dtype=np.float32)
    for user_index in range(scores.shape[0]):
        memory = 0.0
        history: deque[float] = deque(maxlen=LEGACY_WINDOW_DAYS)
        for time_index in range(scores.shape[1]):
            if not active_mask[user_index, time_index]:
                memory = 0.0
                history.clear()
                continue
            score = float(scores[user_index, time_index])
            memory = max(score, LEGACY_DECAY * memory)
            history.append(score)
            frequency = (
                sum(
                    value > LEGACY_ALERT_THRESHOLD
                    for value in history
                )
                / LEGACY_WINDOW_DAYS
            )
            risk[user_index, time_index] = (
                LEGACY_SEVERITY_WEIGHT * score
                + LEGACY_MEMORY_WEIGHT * memory
                + LEGACY_FREQUENCY_WEIGHT * frequency
            )
    return risk


def compute_risk(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    scores: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Default S3 risk. Kept name for pipeline compatibility."""
    return compute_severity_risk(
        users, dates, scores, active_mask
    )


def build_topk(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    risk: np.ndarray,
    active_mask: np.ndarray,
    label_map: dict[tuple[str, str], dict],
) -> pd.DataFrame:
    rows = []
    for time_index, date in enumerate(dates):
        active_indices = np.flatnonzero(active_mask[:, time_index])
        if len(active_indices) == 0:
            continue
        ordering = sorted(
            active_indices,
            key=lambda user_index: (-risk[user_index, time_index], users[user_index]),
        )
        date_text = date.strftime("%Y-%m-%d")
        for k in K_VALUES:
            for rank, user_index in enumerate(ordering[:k], start=1):
                user = users[user_index]
                label = label_map.get((user, date_text), {})
                rows.append(
                    {
                        "k": k,
                        "date": date_text,
                        "rank": rank,
                        "user_id": user,
                        "risk_score": float(risk[user_index, time_index]),
                        "is_malicious_account_day": int(bool(label)),
                        "scenarios": label.get("scenarios", ""),
                        "insider_ids": label.get("insider_ids", ""),
                    }
                )
    return pd.DataFrame(rows)


def evaluate(
    topk: pd.DataFrame,
    incidents: pd.DataFrame,
    dates: pd.DatetimeIndex,
    train_end: int,
    val_end: int,
    positive_days_by_split: dict[str, int],
) -> dict:
    start_date = dates[0].normalize()
    validation_start = dates[train_end + 1].normalize()
    test_start = dates[val_end + 1].normalize()
    final_end = dates[-1].normalize()
    split_ranges = {
        "validation": (validation_start, test_start - pd.Timedelta(days=1)),
        "test": (test_start, final_end),
    }

    incident_rows = incidents.copy()
    incident_rows["start_date"] = pd.to_datetime(incident_rows["start"]).dt.normalize()
    incident_rows["end_date"] = pd.to_datetime(incident_rows["end"]).dt.normalize()

    metrics = {}
    for split, (split_start, split_end) in split_ranges.items():
        metrics[split] = {}
        split_alerts = topk[
            (topk["date"] >= str(split_start.date()))
            & (topk["date"] <= str(split_end.date()))
        ]
        split_incidents = incident_rows[
            (incident_rows["start_date"] >= split_start)
            & (incident_rows["start_date"] <= split_end)
        ]
        for k in K_VALUES:
            alerts_k = split_alerts[split_alerts["k"] == k]
            true_alerts = int(alerts_k["is_malicious_account_day"].sum())
            alert_slots = int(len(alerts_k))
            precision = true_alerts / alert_slots if alert_slots else 0.0

            positive_days = positive_days_by_split[split]

            hits = []
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
                    hits.append(incident.incident_id)
                    delays.append((first_alert - incident.start_date).days)
            incident_recall = len(hits) / len(split_incidents) if len(split_incidents) else 0.0
            metrics[split][f"k_{k}"] = {
                "alert_slots": alert_slots,
                "true_alert_slots": true_alerts,
                "precision_at_k": precision,
                "account_day_recall": (
                    true_alerts / positive_days if positive_days else 0.0
                ),
                "false_alert_slots": alert_slots - true_alerts,
                "false_alerts_per_day": (
                    (alert_slots - true_alerts) / len(pd.date_range(split_start, split_end))
                    if alert_slots
                    else 0.0
                ),
                "incidents_in_split": int(len(split_incidents)),
                "incidents_hit": int(len(hits)),
                "incident_recall": incident_recall,
                "median_detection_delay_days": (
                    float(np.median(delays)) if delays else None
                ),
            }
    return metrics


def main() -> None:
    dataset = np.load(DATASET, allow_pickle=False)
    score_arrays = np.load(SCORE_ARRAYS, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    anomaly_scores = score_arrays["anomaly_score"].astype(np.float32)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])

    score_frame = pd.read_csv(SCORES, dtype={"scenarios": "string", "insider_ids": "string"})
    label_map = {
        (row.user_id, row.date): {
            "scenarios": "" if pd.isna(row.scenarios) else row.scenarios,
            "insider_ids": "" if pd.isna(row.insider_ids) else row.insider_ids,
        }
        for row in score_frame.itertuples(index=False)
        if row.is_malicious_account_day == 1
    }
    positive_days_by_split = (
        score_frame.loc[score_frame["is_malicious_account_day"] == 1, "split"]
        .value_counts()
        .astype(int)
        .to_dict()
    )

    print("Computing dynamic risk...", flush=True)
    risk = compute_risk(users, dates, anomaly_scores, active_mask)
    risk_rows = []
    for user_index, user in enumerate(users):
        for time_index, date in enumerate(dates):
            if not active_mask[user_index, time_index]:
                continue
            risk_rows.append(
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
                    "anomaly_score": float(anomaly_scores[user_index, time_index]),
                    "risk_score": float(risk[user_index, time_index]),
                }
            )
    pd.DataFrame(risk_rows).to_csv(RISK_FILE, index=False, encoding="utf-8")

    print("Building daily Top-K queues...", flush=True)
    incidents = load_incidents()
    topk = build_topk(users, dates, risk, active_mask, label_map)
    topk.to_csv(TOPK_FILE, index=False, encoding="utf-8")

    print("Evaluating Top-K queues...", flush=True)
    metrics = evaluate(
        topk, incidents, dates, train_end, val_end, positive_days_by_split
    )
    summary = {
        "risk_file": str(RISK_FILE),
        "topk_file": str(TOPK_FILE),
        "incidents_file": str(INCIDENTS_FILE),
        "risk_parameters": {
            "risk_mode": "severity_only",
            "legacy_weighted_risk_available": True,
            "legacy_formula": {
                "severity_weight": LEGACY_SEVERITY_WEIGHT,
                "memory_weight": LEGACY_MEMORY_WEIGHT,
                "frequency_weight": LEGACY_FREQUENCY_WEIGHT,
            },
            "k_values": K_VALUES,
            "parameters_tuned_on_labels": False,
        },
        "metrics": metrics,
        "labels_used_in_risk_construction": False,
    }
    METRICS_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
