from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r52"
DATASET = PROCESSED / "r5.2_sequence_dataset.npz"
FEATURE_CSV = PROCESSED / "r5.2_user_day_features.csv"
TS_SCORES = PROCESSED / "r5.2_ts2vec_scores.npz"
LABELS = PROCESSED / "r5.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r5.2_incidents.csv"

RESULTS_FILE = PROCESSED / "r5.2_frozen_multiscale_results.json"
SUMMARY_FILE = PROCESSED / "r5.2_frozen_multiscale_summary.json"
RISK_FILE = PROCESSED / "r5.2_frozen_multiscale_risk.npz"
EVIDENCE_FILE = PROCESSED / (
    "r5.2_frozen_multiscale_alert_evidence.csv"
)

PRIMARY_K = 10
REPORT_K = [5, 10, 20]
WEIGHTS = (0.4, 0.3, 0.3, 0.0)

sys.path.insert(0, str(ROOT))
from baseline_isolation_forest_r42 import (  # noqa: E402
    FEATURES,
    LOG1P_FEATURES,
)
from deep_compare_per_user_r42 import (  # noqa: E402
    per_user_quantile_scores,
)
from dual_baseline_r42 import (  # noqa: E402
    build_group_baseline,
    calibrate,
)
from external_eval_r52 import (  # noqa: E402
    build_external_alerts,
    personal_zscore,
)
from fusion_r42 import daily_percentile  # noqa: E402
from multiscale_risk_r42 import (  # noqa: E402
    build_collective_signal,
    combine_risk,
    dynamic_components,
    event_metrics,
    warning_levels,
)


def main() -> None:
    with np.load(DATASET, allow_pickle=False) as dataset:
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active_mask = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
        test_start = train_end + 1
        test_end = len(dates) - 1

    raw = pd.read_csv(FEATURE_CSV, parse_dates=["date"])
    index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    aligned = raw.set_index(["user_id", "date"]).reindex(index)
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

    personal_raw = personal_zscore(
        values, active_mask, train_end
    )
    personal_quantile = per_user_quantile_scores(
        personal_raw, active_mask, train_end
    )
    point = daily_percentile(personal_raw, active_mask)

    with np.load(TS_SCORES, allow_pickle=False) as ts_scores:
        ts_raw = ts_scores["anomaly_score"].astype(np.float32)
    context_quantile = per_user_quantile_scores(
        ts_raw, active_mask, train_end
    )
    context = daily_percentile(
        context_quantile, active_mask
    )
    group_raw, peer_levels, peer_sizes = build_group_baseline(
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
    group = daily_percentile(group_score, active_mask)
    collective, collective_levels, collective_sizes = (
        build_collective_signal(
            point,
            personal_quantile,
            active_mask,
            functional_units,
            roles,
        )
    )

    severity = (point + context) / 2.0
    frequency, duration, agreement = dynamic_components(
        severity,
        [point, context, group, collective],
        active_mask,
    )
    risk = combine_risk(
        severity,
        frequency,
        duration,
        agreement,
        WEIGHTS,
    )

    labels = np.zeros((len(users), len(dates)), dtype=np.uint8)
    user_index = {user: i for i, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): i
        for i, date in enumerate(dates)
    }
    labels_frame = pd.read_csv(LABELS, parse_dates=["date"])
    for row in labels_frame.itertuples(index=False):
        labels[
            user_index[row.user_id],
            date_index[row.date.strftime("%Y-%m-%d")],
        ] = 1
    incidents = pd.read_csv(INCIDENTS)
    incidents["start_date"] = pd.to_datetime(
        incidents["start"]
    ).dt.normalize()
    incidents["end_date"] = pd.to_datetime(
        incidents["end"]
    ).dt.normalize()

    results = []
    for method, score in (
        ("static_point_context", severity),
        ("frozen_dynamic_accumulation", risk),
    ):
        for k in REPORT_K:
            alerts = build_external_alerts(
                score, active_mask, test_start, k, 7
            )
            metrics = event_metrics(
                users,
                dates,
                alerts,
                labels,
                incidents,
                test_start,
                test_end,
            )
            results.append(
                {
                    "method": method,
                    "k": k,
                    "weights": WEIGHTS if "dynamic" in method else None,
                    "external_test": metrics,
                }
            )
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    warning_summary, warning_events = warning_levels(
        users,
        dates,
        risk,
        labels,
        incidents,
        train_end,
        test_start,
        test_end,
        active_mask,
    )
    train_risk = risk[:, : train_end + 1][
        active_mask[:, : train_end + 1]
    ]
    thresholds = {
        "level_1": float(np.quantile(train_risk, 0.95)),
        "level_2": float(np.quantile(train_risk, 0.98)),
        "level_3": float(np.quantile(train_risk, 0.99)),
    }
    alerts = build_external_alerts(
        risk, active_mask, test_start, PRIMARY_K, 7
    )
    evidence_rows = []
    row_users, row_times = np.nonzero(alerts)
    for user, time in zip(row_users, row_times):
        evidence_rows.append(
            {
                "date": dates[time].strftime("%Y-%m-%d"),
                "user_id": users[user],
                "risk": float(risk[user, time]),
                "warning_level": sum(
                    risk[user, time] >= threshold
                    for threshold in thresholds.values()
                ),
                "point_percentile": float(point[user, time]),
                "context_percentile": float(context[user, time]),
                "group_percentile": float(group[user, time]),
                "collective_percentile": float(
                    collective[user, time]
                ),
                "frequency": float(frequency[user, time]),
                "duration": float(duration[user, time]),
                "agreement": float(agreement[user, time]),
                "peer_level": int(collective_levels[user, time]),
                "peer_count": int(collective_sizes[user, time]),
            }
        )
    pd.DataFrame(evidence_rows).to_csv(
        EVIDENCE_FILE, index=False, encoding="utf-8"
    )
    np.savez_compressed(
        RISK_FILE,
        users=users,
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        severity=severity.astype(np.float32),
        risk=risk.astype(np.float32),
        point=point.astype(np.float32),
        context=context.astype(np.float32),
        group=group.astype(np.float32),
        collective=collective.astype(np.float32),
        frequency=frequency.astype(np.float32),
        duration=duration.astype(np.float32),
        agreement=agreement.astype(np.float32),
        weights=np.asarray(WEIGHTS, dtype=np.float32),
    )
    summary = {
        "purpose": (
            "Post-hoc external sensitivity for the r4.2-validation "
            "multiscale and dynamic-risk configuration."
        ),
        "selected_static_combination": "point_context",
        "selected_dynamic_weights": WEIGHTS,
        "warning_levels": warning_summary,
        "warning_per_event": warning_events,
        "test_used_for_selection": False,
        "confirmatory": False,
        "results_file": str(RESULTS_FILE),
        "risk_file": str(RISK_FILE),
        "evidence_file": str(EVIDENCE_FILE),
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
