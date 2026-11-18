from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
FEATURE_CSV = PROCESSED / "r4.2_user_day_features.csv"
TS2VEC_SCORES = PROCESSED / "r4.2_ts2vec_daily_errors.csv"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"
EXISTING_TREND = PROCESSED / "r4.2_topk_trend.csv"

IF_RAW_ARRAY = PROCESSED / "r4.2_iforest_raw_scores.npz"
IF_DAILY_SCORES = PROCESSED / "r4.2_iforest_daily_scores.csv"
IF_MODEL = PROCESSED / "r4.2_isolation_forest.joblib"
COMBINED_TREND = PROCESSED / "r4.2_three_method_trend.csv"
HIT_FILE = PROCESSED / "r4.2_incident_hits_by_method.csv"
HIT_SUMMARY = PROCESSED / "r4.2_incident_hit_overlap.csv"
SUMMARY = PROCESSED / "r4.2_iforest_summary.json"

sys.path.insert(0, str(ROOT))
from risk_topk_r42 import compute_risk, load_incidents  # noqa: E402
from topk_trend_r42 import (  # noqa: E402
    K_VALUES,
    build_top100,
    evaluate_curve,
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


def robust_calibrate(train_values: np.ndarray, all_values: np.ndarray) -> np.ndarray:
    median = float(np.median(train_values))
    mad = float(np.median(np.abs(train_values - median)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(np.std(train_values))
    if scale < 1e-8:
        scale = 1.0
    return np.clip((all_values - median) / scale, 0.0, None)


def incident_hits(
    top100: pd.DataFrame,
    incidents: pd.DataFrame,
    dates: pd.DatetimeIndex,
    train_end: int,
    val_end: int,
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
            split_incidents = incident_rows[
                (incident_rows["start_date"] >= split_start)
                & (incident_rows["start_date"] <= split_end)
            ]
            split_top = method_top[
                (method_top["date"] >= str(split_start.date()))
                & (method_top["date"] <= str(split_end.date()))
            ]
            for k in K_VALUES:
                alerts_k = split_top[split_top["rank"] <= k]
                for incident in split_incidents.itertuples(index=False):
                    accounts = set(incident.observed_accounts.split("|"))
                    candidate = alerts_k[alerts_k["user_id"].isin(accounts)]
                    candidate = candidate[
                        (candidate["date"] >= str(incident.start_date.date()))
                        & (candidate["date"] <= str(incident.end_date.date()))
                    ]
                    if len(candidate):
                        rows.append(
                            {
                                "method": method,
                                "split": split,
                                "k": k,
                                "incident_id": incident.incident_id,
                            }
                        )
    return pd.DataFrame(rows)


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
    values = (
        raw.set_index(["user_id", "date"])[FEATURES]
        .reindex(index)
        .to_numpy(dtype=np.float64)
        .reshape(len(users), len(dates), len(FEATURES))
    )
    for index_feature, feature in enumerate(FEATURES):
        if feature in LOG1P_FEATURES:
            mask = np.isfinite(values[:, :, index_feature])
            values[:, :, index_feature][mask] = np.log1p(
                values[:, :, index_feature][mask]
            )

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
    print("Training Isolation Forest...", flush=True)
    model.fit(train_normalized)
    joblib.dump(model, IF_MODEL)

    print("Scoring all active user-days...", flush=True)
    raw_iforest = np.full(active_mask.shape, np.nan, dtype=np.float32)
    active_values = normalized[active_mask]
    active_scores = -model.score_samples(active_values)
    raw_iforest[active_mask] = active_scores.astype(np.float32)
    train_scores = raw_iforest[:, : train_end + 1][train_active]
    anomaly_scores = robust_calibrate(train_scores, raw_iforest)

    np.savez_compressed(
        IF_RAW_ARRAY,
        users=np.asarray(users, dtype="U32"),
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        iforest_raw=raw_iforest,
        iforest_score=anomaly_scores,
        active_mask=active_mask,
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
                    "iforest_raw": float(raw_iforest[user_index, time_index]),
                    "iforest_score": float(anomaly_scores[user_index, time_index]),
                }
            )
    pd.DataFrame(score_rows).to_csv(IF_DAILY_SCORES, index=False, encoding="utf-8")

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

    dynamic = compute_risk(users, dates, anomaly_scores.astype(np.float32), active_mask)
    methods = {
        "iforest_raw": anomaly_scores.astype(np.float32),
        "iforest_dynamic": dynamic,
    }
    top_frames = [
        build_top100(name, users, dates, score, active_mask, label_map)
        for name, score in methods.items()
    ]
    top100 = pd.concat(top_frames, ignore_index=True)
    incidents = pd.read_csv(INCIDENTS)
    results = evaluate_curve(
        top100, incidents, dates, train_end, val_end, positive_counts
    )

    combined = pd.concat(
        [pd.read_csv(EXISTING_TREND), results], ignore_index=True
    )
    combined.to_csv(COMBINED_TREND, index=False, encoding="utf-8")

    # Build hit sets for TS2Vec and z-score from their saved trend by rerunning
    # only the dynamic top-100 queues.
    ts_arrays = np.load(PROCESSED / "r4.2_ts2vec_scores.npz", allow_pickle=False)
    z_frame = pd.read_csv(
        PROCESSED / "r4.2_zscore_daily_scores.csv",
        usecols=["user_id", "date", "robust_z_score"],
    )
    z_frame["date"] = pd.to_datetime(z_frame["date"])
    z_matrix = (
        z_frame.set_index(["user_id", "date"])["robust_z_score"]
        .reindex(index)
        .to_numpy(dtype=np.float32)
        .reshape(len(users), len(dates))
    )
    ts_matrix = ts_arrays["anomaly_score"].astype(np.float32)
    all_scores = {
        "ts2vec_dynamic": compute_risk(users, dates, ts_matrix, active_mask),
        "zscore_dynamic": compute_risk(users, dates, z_matrix, active_mask),
        **methods,
    }
    all_top = pd.concat(
        [
            build_top100(name, users, dates, score, active_mask, label_map)
            for name, score in all_scores.items()
        ],
        ignore_index=True,
    )
    hits = incident_hits(all_top, incidents, dates, train_end, val_end)
    hits.to_csv(HIT_FILE, index=False, encoding="utf-8")

    overlap_rows = []
    holdout = hits[hits["split"] == "validation_plus_test"]
    for k in K_VALUES:
        sets = {
            method: set(
                holdout[
                    (holdout["k"] == k) & (holdout["method"] == method)
                ]["incident_id"]
            )
            for method in all_scores
        }
        union = set().union(*sets.values())
        overlap_rows.append(
            {
                "k": k,
                "ts2vec_hits": len(sets["ts2vec_dynamic"]),
                "zscore_hits": len(sets["zscore_dynamic"]),
                "iforest_hits": len(sets["iforest_dynamic"]),
                "union_hits": len(union),
                "ts2vec_only": len(sets["ts2vec_dynamic"] - set().union(sets["zscore_dynamic"], sets["iforest_dynamic"])),
                "zscore_only": len(sets["zscore_dynamic"] - set().union(sets["ts2vec_dynamic"], sets["iforest_dynamic"])),
                "iforest_only": len(sets["iforest_dynamic"] - set().union(sets["ts2vec_dynamic"], sets["zscore_dynamic"])),
                "all_three": len(sets["ts2vec_dynamic"] & sets["zscore_dynamic"] & sets["iforest_dynamic"]),
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(HIT_SUMMARY, index=False, encoding="utf-8")

    summary = {
        "iforest_model": str(IF_MODEL),
        "iforest_score_file": str(IF_DAILY_SCORES),
        "combined_trend_file": str(COMBINED_TREND),
        "hit_file": str(HIT_FILE),
        "hit_summary_file": str(HIT_SUMMARY),
        "iforest_parameters": {
            "n_estimators": 200,
            "max_samples": "auto",
            "contamination": "auto",
            "max_features": 1.0,
            "bootstrap": False,
            "random_state": 20261109,
        },
        "feature_count": len(FEATURES),
        "training_rows": int(len(train_features)),
        "labels_used_for_fit": False,
        "labels_used_for_calibration": False,
        "holdout_overlap": overlap.to_dict(orient="records"),
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
