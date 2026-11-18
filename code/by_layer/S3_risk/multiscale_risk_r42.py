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
ZSCORE_DAILY = PROCESSED / "r4.2_zscore_daily_scores.csv"
GROUP_SCORES = PROCESSED / "r4.2_group_baseline_daily_scores.csv"
CALIBRATED = PROCESSED / "r4.2_calibrated_method_scores.npz"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"

SELECTION_FILE = PROCESSED / "r4.2_multiscale_selection.csv"
WEIGHTS_FILE = PROCESSED / "r4.2_dynamic_weight_selection.csv"
RESULTS_FILE = PROCESSED / "r4.2_multiscale_dynamic_results.json"
RISK_FILE = PROCESSED / "r4.2_multiscale_dynamic_risk.npz"
EVIDENCE_FILE = PROCESSED / "r4.2_multiscale_alert_evidence.csv"
SUMMARY_FILE = PROCESSED / "r4.2_multiscale_dynamic_summary.json"

PRIMARY_K = 10
REPORT_K = [5, 10, 20]
COOLDOWN_DAYS = 7
HIGH_QUANTILE = 0.95
DURATION_THRESHOLD = 0.90
MIN_PEER_COUNT = 10

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import (  # noqa: E402
    per_user_quantile_scores,
)
from fusion_r42 import daily_percentile, matrix_from_csv  # noqa: E402
from production_protocol_r42 import (  # noqa: E402
    build_alerts,
    evaluate,
    label_matrix,
)


def build_collective_signal(
    personal_percentile: np.ndarray,
    personal_quantile: np.ndarray,
    active_mask: np.ndarray,
    functional_units: np.ndarray,
    roles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high_threshold = -np.log(1.0 - HIGH_QUANTILE)
    personal_high = personal_quantile >= high_threshold
    peer_fraction = np.full(personal_percentile.shape, np.nan)
    peer_levels = np.zeros(personal_percentile.shape, dtype=np.int8)
    peer_sizes = np.zeros(personal_percentile.shape, dtype=np.int16)

    for time_index in range(personal_percentile.shape[1]):
        active = np.flatnonzero(active_mask[:, time_index])
        if len(active) == 0:
            continue
        assigned = np.zeros(personal_percentile.shape[0], dtype=bool)

        def assign(labels: np.ndarray, level: int) -> None:
            for group in np.unique(labels[active]):
                label = str(group)
                if label in {"Unknown", "nan"}:
                    continue
                members = active[labels[active] == group]
                members = members[~assigned[members]]
                if len(members) < MIN_PEER_COUNT:
                    continue
                high_count = personal_high[members, time_index].sum()
                for user_index in members:
                    other_count = max(len(members) - 1, 1)
                    peer_fraction[user_index, time_index] = (
                        high_count
                        - int(personal_high[user_index, time_index])
                    ) / other_count
                    peer_levels[user_index, time_index] = level
                    peer_sizes[user_index, time_index] = len(members)
                assigned[members] = True

        assign(functional_units[:, time_index], 1)
        assign(roles[:, time_index], 2)

        remaining = active[~assigned[active]]
        if len(remaining):
            high_count = personal_high[remaining, time_index].sum()
            for user_index in remaining:
                peer_fraction[user_index, time_index] = (
                    high_count
                    - int(personal_high[user_index, time_index])
                ) / max(len(remaining) - 1, 1)
                peer_levels[user_index, time_index] = 3
                peer_sizes[user_index, time_index] = len(remaining)

    collective_raw = personal_percentile * peer_fraction
    collective = daily_percentile(collective_raw, active_mask)
    return collective, peer_levels, peer_sizes


def dynamic_components(
    severity: np.ndarray,
    component_percentiles: list[np.ndarray],
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency = np.full(severity.shape, np.nan, dtype=np.float32)
    duration = np.full(severity.shape, np.nan, dtype=np.float32)
    for user_index in range(severity.shape[0]):
        history = []
        streak = 0
        for time_index in range(severity.shape[1]):
            if not active_mask[user_index, time_index]:
                history = []
                streak = 0
                continue
            value = float(severity[user_index, time_index])
            history.append(value > HIGH_QUANTILE)
            history = history[-7:]
            frequency[user_index, time_index] = sum(history) / 7.0
            streak = streak + 1 if value >= DURATION_THRESHOLD else 0
            duration[user_index, time_index] = min(streak / 7.0, 1.0)

    agreement = np.full(severity.shape, np.nan, dtype=np.float32)
    for time_index in range(severity.shape[1]):
        active = np.flatnonzero(active_mask[:, time_index])
        if len(active) == 0:
            continue
        agreement[active, time_index] = np.mean(
            [
                component[active, time_index]
                >= DURATION_THRESHOLD
                for component in component_percentiles
            ],
            axis=0,
        )
    return frequency, duration, agreement


def combine_risk(
    severity: np.ndarray,
    frequency: np.ndarray,
    duration: np.ndarray,
    agreement: np.ndarray,
    weights: tuple[float, float, float, float],
) -> np.ndarray:
    w1, w2, w3, w4 = weights
    return (
        w1 * severity
        + w2 * frequency
        + w3 * duration
        + w4 * agreement
    ).astype(np.float32)


def event_metrics(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    alerts: np.ndarray,
    labels: np.ndarray,
    incidents: pd.DataFrame,
    start_index: int,
    end_index: int,
) -> dict:
    date_mask = np.zeros(len(dates), dtype=bool)
    date_mask[start_index : end_index + 1] = True
    split_alerts = alerts[:, date_mask]
    split_labels = labels[:, date_mask]
    alert_slots = int(split_alerts.sum())
    tp = int((split_alerts & split_labels).sum())
    fp = alert_slots - tp
    positive_days = int(split_labels.sum())
    negative_days = int((1 - split_labels).sum())
    precision = tp / alert_slots if alert_slots else 0.0
    recall = tp / positive_days if positive_days else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / negative_days if negative_days else 0.0

    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    split_start = dates[start_index].normalize()
    split_end = dates[end_index].normalize()
    selected = incidents[
        (incidents["start_date"] >= split_start)
        & (incidents["start_date"] <= split_end)
    ]
    horizon_results = {}
    for horizon in (1, 3):
        hit_ids = set()
        contact_times = []
        for incident in selected.itertuples(index=False):
            accounts = [
                user_index[account]
                for account in incident.observed_accounts.split("|")
                if account in user_index
            ]
            candidate_dates = pd.date_range(
                incident.start_date
                - pd.Timedelta(days=COOLDOWN_DAYS - 1),
                min(
                    incident.end_date,
                    incident.start_date
                    + pd.Timedelta(days=horizon - 1),
                ),
                freq="D",
            )
            for date in candidate_dates:
                text = date.strftime("%Y-%m-%d")
                if text not in date_index:
                    continue
                if alerts[np.asarray(accounts), date_index[text]].any():
                    hit_ids.add(str(incident.incident_id))
                    contact_times.append(
                        (date.normalize() - incident.start_date).days
                    )
                    break
        horizon_results[str(horizon)] = {
            "incidents_hit": len(hit_ids),
            "incidents_total": int(len(selected)),
            "event_recall": (
                len(hit_ids) / len(selected)
                if len(selected)
                else 0.0
            ),
            "pre_event_contacts": sum(
                value < 0 for value in contact_times
            ),
            "median_contact_time_days": (
                float(np.median(contact_times))
                if contact_times
                else None
            ),
        }
    return {
        "alert_slots": alert_slots,
        "true_positive_days": tp,
        "false_positive_days": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_alerts_per_day": fp / len(dates[start_index : end_index + 1]),
        "horizons": horizon_results,
    }


def warning_levels(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    risk: np.ndarray,
    labels: np.ndarray,
    incidents: pd.DataFrame,
    train_end: int,
    start_index: int,
    end_index: int,
    active_mask: np.ndarray,
) -> tuple[dict, dict]:
    train_values = risk[:, : train_end + 1][
        active_mask[:, : train_end + 1]
    ]
    thresholds = {
        "level_1": float(np.quantile(train_values, 0.95)),
        "level_2": float(np.quantile(train_values, 0.98)),
        "level_3": float(np.quantile(train_values, 0.99)),
    }
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    selected = incidents[
        (incidents["start_date"] >= dates[start_index].normalize())
        & (incidents["start_date"] <= dates[end_index].normalize())
    ]
    summary = {}
    per_event = {}
    for level, threshold in thresholds.items():
        split_risk = risk[:, start_index : end_index + 1]
        split_active = active_mask[:, start_index : end_index + 1]
        split_labels = labels[:, start_index : end_index + 1]
        flagged = split_active & (split_risk >= threshold)
        tp = int((flagged & (split_labels == 1)).sum())
        fp = int((flagged & (split_labels == 0)).sum())
        positive = int(split_labels.sum())
        negative = int((split_active & (split_labels == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positive if positive else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        contacts = []
        leads = []
        for incident in selected.itertuples(index=False):
            accounts = [
                user_index[account]
                for account in incident.observed_accounts.split("|")
                if account in user_index
            ]
            candidate_dates = pd.date_range(
                incident.start_date
                - pd.Timedelta(days=COOLDOWN_DAYS - 1),
                min(
                    incident.end_date,
                    incident.start_date + pd.Timedelta(days=2),
                ),
                freq="D",
            )
            found = None
            for date in candidate_dates:
                text = date.strftime("%Y-%m-%d")
                if text not in date_index:
                    continue
                if (
                    risk[np.asarray(accounts), date_index[text]]
                    >= threshold
                ).any():
                    found = date.normalize()
                    break
            if found is not None:
                contacts.append(str(incident.incident_id))
                leads.append(
                    (incident.start_date - found).days
                )
                per_event.setdefault(
                    str(incident.incident_id), {}
                )[level] = {
                    "contact_date": str(found.date()),
                    "lead_days": int(
                        (incident.start_date - found).days
                    ),
                }
        summary[level] = {
            "threshold": threshold,
            "alerted_account_days": int(flagged.sum()),
            "alerts_per_day": float(flagged.sum() / len(split_risk[0])),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fp / negative if negative else 0.0,
            "events_contacted": len(contacts),
            "events_total": int(len(selected)),
            "event_coverage": (
                len(contacts) / len(selected)
                if len(selected)
                else 0.0
            ),
            "pre_event_contacts": sum(lead > 0 for lead in leads),
            "median_lead_days_among_hits": (
                float(np.median(leads)) if leads else None
            ),
        }
    return summary, per_event


def main() -> None:
    with np.load(DATASET, allow_pickle=False) as dataset:
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active_mask = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
        val_end = int(dataset["val_end_index"])
    with np.load(CALIBRATED, allow_pickle=False) as calibrated:
        ts_context = calibrated["ts2vec_quantile"].astype(np.float32)

    raw = pd.read_csv(FEATURE_CSV, parse_dates=["date"])
    index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    aligned = raw.set_index(["user_id", "date"]).reindex(index)
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

    personal_raw = matrix_from_csv(
        ZSCORE_DAILY, users, dates, "robust_z_score"
    )
    personal_quantile = per_user_quantile_scores(
        personal_raw, active_mask, train_end
    )
    point = daily_percentile(personal_raw, active_mask)
    context = daily_percentile(ts_context, active_mask)
    group = daily_percentile(
        matrix_from_csv(
            GROUP_SCORES,
            users,
            dates,
            "group_score",
        ),
        active_mask,
    )
    collective, peer_levels, peer_sizes = build_collective_signal(
        point,
        personal_quantile,
        active_mask,
        functional_units,
        roles,
    )

    components = {
        "point": point,
        "context": context,
        "group": group,
        "collective": collective,
    }
    combinations = {
        "point_only": point,
        "context_only": context,
        "group_only": group,
        "collective_only": collective,
        "point_context": (point + context) / 2.0,
        "point_context_group": (point + context + group) / 3.0,
        "all_four": (
            point + context + group + collective
        ) / 4.0,
    }
    labels = label_matrix(users, dates)
    incidents = pd.read_csv(INCIDENTS)
    incidents["start_date"] = pd.to_datetime(
        incidents["start"]
    ).dt.normalize()
    incidents["end_date"] = pd.to_datetime(
        incidents["end"]
    ).dt.normalize()

    selection_rows = []
    selected = None
    for name, score in combinations.items():
        alerts = build_alerts(
            score, active_mask, PRIMARY_K, COOLDOWN_DAYS
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
            "validation_recall_3d": validation["horizons"]["3"][
                "event_recall"
            ],
            "validation_recall_1d": validation["horizons"]["1"][
                "event_recall"
            ],
            "validation_precision": validation["precision_at_k"],
            "validation_false_per_day": validation[
                "false_alerts_per_day"
            ],
        }
        selection_rows.append(row)
        if selected is None:
            selected = row
        else:
            current = (
                -row["validation_recall_3d"],
                -row["validation_recall_1d"],
                -row["validation_precision"],
                row["validation_false_per_day"],
            )
            best = (
                -selected["validation_recall_3d"],
                -selected["validation_recall_1d"],
                -selected["validation_precision"],
                selected["validation_false_per_day"],
            )
            if current < best:
                selected = row
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(
        SELECTION_FILE, index=False, encoding="utf-8"
    )
    selected_combination = selected["combination"]
    severity = combinations[selected_combination]
    frequency, duration, agreement = dynamic_components(
        severity,
        [point, context, group, collective],
        active_mask,
    )
    weight_candidates = {
        "static": (1.0, 0.0, 0.0, 0.0),
        "equal": (0.25, 0.25, 0.25, 0.25),
        "draft_balanced": (0.40, 0.20, 0.20, 0.20),
        "severity_heavy": (0.55, 0.15, 0.15, 0.15),
        "accumulation_heavy": (0.40, 0.30, 0.30, 0.00),
        "current_style": (0.70, 0.20, 0.10, 0.00),
    }
    dynamic_scores = {
        name: combine_risk(
            severity,
            frequency,
            duration,
            agreement,
            weights,
        )
        for name, weights in weight_candidates.items()
    }
    weight_rows = []
    selected_weights = None
    for name, score in dynamic_scores.items():
        alerts = build_alerts(
            score, active_mask, PRIMARY_K, COOLDOWN_DAYS
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
            "weight_name": name,
            "weights": weight_candidates[name],
            "validation_recall_3d": validation["horizons"]["3"][
                "event_recall"
            ],
            "validation_recall_1d": validation["horizons"]["1"][
                "event_recall"
            ],
            "validation_precision": validation["precision_at_k"],
            "validation_false_per_day": validation[
                "false_alerts_per_day"
            ],
        }
        weight_rows.append(row)
        if selected_weights is None:
            selected_weights = row
        else:
            current = (
                -row["validation_recall_3d"],
                -row["validation_recall_1d"],
                -row["validation_precision"],
                row["validation_false_per_day"],
            )
            best = (
                -selected_weights["validation_recall_3d"],
                -selected_weights["validation_recall_1d"],
                -selected_weights["validation_precision"],
                selected_weights["validation_false_per_day"],
            )
            if current < best:
                selected_weights = row
    pd.DataFrame(weight_rows).to_csv(
        WEIGHTS_FILE, index=False, encoding="utf-8"
    )
    selected_weight_name = selected_weights["weight_name"]
    final_risk = dynamic_scores[selected_weight_name]

    final_results = []
    for method_name, score in (
        ("static_multiscale", severity),
        (f"dynamic_{selected_weight_name}", final_risk),
    ):
        for k in REPORT_K:
            alerts = build_alerts(
                score, active_mask, k, COOLDOWN_DAYS
            )
            val_metrics = event_metrics(
                users,
                dates,
                alerts,
                labels,
                incidents,
                train_end + 1,
                val_end,
            )
            test_metrics = event_metrics(
                users,
                dates,
                alerts,
                labels,
                incidents,
                val_end + 1,
                len(dates) - 1,
            )
            final_results.append(
                {
                    "method": method_name,
                    "k": k,
                    "weights": (
                        weight_candidates[selected_weight_name]
                        if method_name.startswith("dynamic_")
                        else None
                    ),
                    "validation": val_metrics,
                    "test": test_metrics,
                }
            )
    RESULTS_FILE.write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    warning_summary, warning_events = warning_levels(
        users,
        dates,
        final_risk,
        labels,
        incidents,
        train_end,
        val_end + 1,
        len(dates) - 1,
        active_mask,
    )
    train_risk = final_risk[:, : train_end + 1][
        active_mask[:, : train_end + 1]
    ]
    level_thresholds = {
        "level_1": float(np.quantile(train_risk, 0.95)),
        "level_2": float(np.quantile(train_risk, 0.98)),
        "level_3": float(np.quantile(train_risk, 0.99)),
    }
    alerts = build_alerts(
        final_risk, active_mask, PRIMARY_K, COOLDOWN_DAYS
    )
    evidence_rows = []
    row_users, row_times = np.nonzero(alerts)
    for user_index, time_index in zip(row_users, row_times):
        level = sum(
            final_risk[user_index, time_index] >= threshold
            for threshold in level_thresholds.values()
        )
        evidence_rows.append(
            {
                "date": dates[time_index].strftime("%Y-%m-%d"),
                "user_id": users[user_index],
                "risk": float(final_risk[user_index, time_index]),
                "warning_level": int(level),
                "point_percentile": float(point[user_index, time_index]),
                "context_percentile": float(
                    context[user_index, time_index]
                ),
                "group_percentile": float(group[user_index, time_index]),
                "collective_percentile": float(
                    collective[user_index, time_index]
                ),
                "peer_level": int(peer_levels[user_index, time_index]),
                "peer_count": int(peer_sizes[user_index, time_index]),
                "frequency": float(frequency[user_index, time_index]),
                "duration": float(duration[user_index, time_index]),
                "agreement": float(agreement[user_index, time_index]),
            }
        )
    pd.DataFrame(evidence_rows).to_csv(
        EVIDENCE_FILE, index=False, encoding="utf-8"
    )
    summary = {
        "selected_multiscale_combination": selected_combination,
        "selected_weight_name": selected_weight_name,
        "selected_weights": weight_candidates[selected_weight_name],
        "warning_levels": warning_summary,
        "warning_per_event": warning_events,
        "selection_file": str(SELECTION_FILE),
        "weight_file": str(WEIGHTS_FILE),
        "results_file": str(RESULTS_FILE),
        "evidence_file": str(EVIDENCE_FILE),
        "test_used_for_selection": False,
    }
    np.savez_compressed(
        RISK_FILE,
        users=users,
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        risk=final_risk.astype(np.float32),
        severity=severity.astype(np.float32),
        point=point.astype(np.float32),
        context=context.astype(np.float32),
        group=group.astype(np.float32),
        collective=collective.astype(np.float32),
        frequency=frequency.astype(np.float32),
        duration=duration.astype(np.float32),
        agreement=agreement.astype(np.float32),
        selected_combination=np.asarray(
            selected_combination, dtype="U64"
        ),
        selected_weight_name=np.asarray(
            selected_weight_name, dtype="U64"
        ),
        selected_weights=np.asarray(
            weight_candidates[selected_weight_name],
            dtype=np.float32,
        ),
        warning_thresholds=np.asarray(
            list(level_thresholds.values()), dtype=np.float32
        ),
    )
    summary["risk_file"] = str(RISK_FILE)
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(selection.to_string(index=False), flush=True)
    print(pd.DataFrame(weight_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
