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
TS2VEC_ARRAYS = PROCESSED / "r5.2_ts2vec_scores.npz"
LABELS = PROCESSED / "r5.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r5.2_incidents.csv"

RESULTS_FILE = PROCESSED / "r5.2_conditional_combination_results.json"
SUMMARY_FILE = PROCESSED / "r5.2_conditional_combination_summary.json"
ALERTS_FILE = PROCESSED / "r5.2_conditional_combination_alerts.csv"
HITS_FILE = PROCESSED / "r5.2_conditional_combination_hits.csv"
SCENARIO_FILE = PROCESSED / "r5.2_conditional_combination_scenarios.csv"
SCORES_FILE = PROCESSED / "r5.2_conditional_combination_scores.npz"

PRIMARY_K = 10
REPORT_K = [5, 10, 20]
COOLDOWN_DAYS = 7
MIN_PERSONAL_DAYS = 30
MAX_ABS_Z = 20.0

sys.path.insert(0, str(ROOT))
from baseline_isolation_forest_r42 import FEATURES, LOG1P_FEATURES  # noqa: E402
from baseline_zscore_r42 import robust_location_scale  # noqa: E402
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from risk_topk_r42 import compute_risk  # noqa: E402
from external_eval_r52 import (  # noqa: E402
    build_external_alerts,
    evaluate_external,
    iforest_score,
)
from conditional_combination_r42 import (  # noqa: E402
    build_rescue_alerts,
    daily_percentile,
)


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
    return np.clip(
        (
            deviations
            - float(np.asarray(median).reshape(-1)[0])
        )
        / float(np.asarray(scale).reshape(-1)[0]),
        0.0,
        None,
    ).astype(np.float32)


def scenario_macro(
    hit_ids: set[str],
    incidents: pd.DataFrame,
) -> tuple[float, dict[str, dict[str, float | int]]]:
    rows: dict[str, dict[str, float | int]] = {}
    recalls = []
    for scenario, group in incidents.groupby("scenario", sort=True):
        total = int(len(group))
        hits = int(
            group["incident_id"].astype(str).isin(hit_ids).sum()
        )
        recall = hits / total if total else 0.0
        rows[str(scenario)] = {
            "hits": hits,
            "total": total,
            "recall": recall,
        }
        recalls.append(recall)
    return float(np.mean(recalls)) if recalls else 0.0, rows


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

    zscore_raw = personal_zscore(values, active_mask, train_end)
    zscore_dynamic = compute_risk(
        users, dates, zscore_raw, active_mask
    )

    iforest_raw, _, _ = iforest_score(values, active_mask, train_end)
    iforest_quantile = per_user_quantile_scores(
        iforest_raw, active_mask, train_end
    )
    iforest_dynamic = compute_risk(
        users, dates, iforest_quantile, active_mask
    )

    with np.load(TS2VEC_ARRAYS, allow_pickle=False) as ts_arrays:
        if not np.array_equal(ts_arrays["users"].astype(str), users):
            raise RuntimeError("TS2Vec users do not match sequence users.")
        if not np.array_equal(
            pd.to_datetime(ts_arrays["dates"].astype(str)), dates
        ):
            raise RuntimeError("TS2Vec dates do not match sequence dates.")
        ts_raw = ts_arrays["anomaly_score"].astype(np.float32)
    ts_quantile = per_user_quantile_scores(
        ts_raw, active_mask, train_end
    )
    ts_dynamic = compute_risk(
        users, dates, ts_quantile, active_mask
    )

    signals = {
        "ts2vec": daily_percentile(ts_dynamic, active_mask),
        "iforest": daily_percentile(iforest_dynamic, active_mask),
        "zscore": daily_percentile(zscore_dynamic, active_mask),
    }
    raw_quantiles = {
        "ts2vec": ts_quantile,
        "iforest": iforest_quantile,
        "zscore": zscore_raw,
    }
    equal_mean = (
        signals["ts2vec"] + signals["iforest"]
    ) / 2.0

    labels_frame = pd.read_csv(LABELS, parse_dates=["date"])
    user_index = {user: index for index, user in enumerate(users)}
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

    incidents = pd.read_csv(
        INCIDENTS,
        parse_dates=["first_observed_date", "last_observed_date"],
    )
    if int(labels[:, :test_start].sum()) != 0:
        raise RuntimeError("Malicious account-days leaked into training.")

    method_alerts: dict[str, np.ndarray] = {}
    rescue_maps: dict[str, np.ndarray] = {}
    for k in REPORT_K:
        method_alerts[f"ts2vec_single_k{k}"] = build_external_alerts(
            signals["ts2vec"], active_mask, test_start, k, COOLDOWN_DAYS
        )
        method_alerts[f"iforest_single_k{k}"] = build_external_alerts(
            signals["iforest"], active_mask, test_start, k, COOLDOWN_DAYS
        )
        method_alerts[f"equal_percentile_mean_k{k}"] = build_external_alerts(
            equal_mean, active_mask, test_start, k, COOLDOWN_DAYS
        )

        for name, gate in (
            ("iforest_primary_ts2vec_rescue_q1_nogate", "none"),
            ("ts2vec_primary_iforest_rescue_q1_nogate", "none"),
            ("ts2vec_primary_iforest_rescue_q1_personal", "personal"),
        ):
            alerts, rescue_flags = build_rescue_alerts(
                primary_score=(
                    signals["iforest"]
                    if name.startswith("iforest_primary")
                    else signals["ts2vec"]
                ),
                secondary_score=(
                    signals["ts2vec"]
                    if name.startswith("iforest_primary")
                    else signals["iforest"]
                ),
                primary_percentile=(
                    signals["iforest"]
                    if name.startswith("iforest_primary")
                    else signals["ts2vec"]
                ),
                secondary_percentile=(
                    signals["ts2vec"]
                    if name.startswith("iforest_primary")
                    else signals["iforest"]
                ),
                secondary_quantile=(
                    ts_quantile
                    if name.startswith("iforest_primary")
                    else iforest_quantile
                ),
                personal_percentile=signals["zscore"],
                active_mask=active_mask,
                k=k,
                q=1,
                gate_name=gate,
                start_index=test_start,
            )
            method_alerts[f"{name}_k{k}"] = alerts
            rescue_maps[f"{name}_k{k}"] = rescue_flags

    results = []
    alert_rows = []
    hit_rows = []
    scenario_rows = []
    for method_key, alerts in method_alerts.items():
        method_name, k_text = method_key.rsplit("_k", 1)
        k = int(k_text)
        result = evaluate_external(
            users,
            dates,
            active_mask,
            alerts,
            labels,
            incidents,
            test_start,
            method_name,
            k,
        )
        hit_ids = set(
            result["external_test"]["horizons"]["3"]["incident_ids"]
        )
        macro, scenario_recalls = scenario_macro(hit_ids, incidents)
        result["external_test"]["scenario_macro_recall_3d"] = macro
        result["external_test"]["scenario_recalls"] = scenario_recalls
        result["rescue_slots"] = (
            int(rescue_maps[method_key].sum())
            if method_key in rescue_maps
            else 0
        )
        results.append(result)

        alert_users, alert_times = np.nonzero(alerts)
        for alert_user, alert_time in zip(alert_users, alert_times):
            alert_rows.append(
                {
                    "method": method_name,
                    "k": k,
                    "date": dates[alert_time].strftime("%Y-%m-%d"),
                    "user_id": users[alert_user],
                    "rescued": (
                        int(rescue_maps[method_key][alert_user, alert_time])
                        if method_key in rescue_maps
                        else 0
                    ),
                }
            )
        for horizon, details in result["external_test"]["horizons"].items():
            for incident_id in details["incident_ids"]:
                hit_rows.append(
                    {
                        "method": method_name,
                        "k": k,
                        "horizon_days": int(horizon),
                        "incident_id": incident_id,
                    }
                )
        for scenario, values in scenario_recalls.items():
            scenario_rows.append(
                {
                    "method": method_name,
                    "k": k,
                    "scenario": scenario,
                    **values,
                }
            )

    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(alert_rows).to_csv(
        ALERTS_FILE, index=False, encoding="utf-8"
    )
    pd.DataFrame(hit_rows).to_csv(
        HITS_FILE, index=False, encoding="utf-8"
    )
    pd.DataFrame(scenario_rows).to_csv(
        SCENARIO_FILE, index=False, encoding="utf-8"
    )
    np.savez_compressed(
        SCORES_FILE,
        users=np.asarray(users, dtype="U32"),
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        ts2vec_dynamic=ts_dynamic.astype(np.float32),
        iforest_dynamic=iforest_dynamic.astype(np.float32),
        zscore_dynamic=zscore_dynamic.astype(np.float32),
        ts2vec_signal=signals["ts2vec"].astype(np.float32),
        iforest_signal=signals["iforest"].astype(np.float32),
        zscore_signal=signals["zscore"].astype(np.float32),
        ts2vec_quantile=ts_quantile.astype(np.float32),
        iforest_quantile=iforest_quantile.astype(np.float32),
    )

    k10_rows = [
        row for row in results if row["k"] == PRIMARY_K
    ]
    summary = {
        "evaluation_design": (
            "External sensitivity only. The selected methods and parameters "
            "were frozen from r4.2 validation; r5.2 was not used to choose "
            "fusion weights, q, gates, or K."
        ),
        "train_end": str(dates[train_end].date()),
        "external_start": str(dates[test_start].date()),
        "incident_count": int(len(incidents)),
        "primary_k": PRIMARY_K,
        "cooldown_days": COOLDOWN_DAYS,
        "report_k": REPORT_K,
        "k10": {
            row["method"]: {
                "hits_3d": row["external_test"]["horizons"]["3"][
                    "incidents_hit"
                ],
                "incidents_total": row["external_test"]["horizons"]["3"][
                    "incidents_total"
                ],
                "recall_3d": row["external_test"]["horizons"]["3"][
                    "event_recall"
                ],
                "recall_1d": row["external_test"]["horizons"]["1"][
                    "event_recall"
                ],
                "precision": row["external_test"]["precision_at_k"],
                "false_per_day": row["external_test"][
                    "false_alerts_per_day"
                ],
                "scenario_macro": row["external_test"][
                    "scenario_macro_recall_3d"
                ],
                "rescue_slots": row["rescue_slots"],
            }
            for row in k10_rows
        },
        "outputs": {
            "results_file": str(RESULTS_FILE),
            "alerts_file": str(ALERTS_FILE),
            "hits_file": str(HITS_FILE),
            "scenario_file": str(SCENARIO_FILE),
            "scores_file": str(SCORES_FILE),
        },
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
