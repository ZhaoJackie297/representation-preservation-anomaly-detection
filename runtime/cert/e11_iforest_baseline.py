from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e11_iforest"
)
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from e3_multichannel_amplitude import (  # noqa: E402
    evaluate_variant,
)
from prepare_r42_sequences import (  # noqa: E402
    FEATURES,
    LOG1P_FEATURES,
)
from quota_relationship_sim import load_r42  # noqa: E402


TIME_FEATURES = (
    "first_logon_minute",
    "last_logoff_minute",
    "active_span_minutes",
)


def build_feature_tensor(
    users: np.ndarray,
    dates: np.ndarray,
    active: np.ndarray,
    train_end: int,
) -> np.ndarray:
    raw_path = (
        ROOT
        / "processed_r42"
        / "r4.2_user_day_features.parquet"
    )
    source_columns = [
        feature
        for feature in FEATURES
        if feature not in {"weekday_sin", "weekday_cos"}
    ]
    frame = pd.read_parquet(
        raw_path,
        columns=["user_id", "date", *source_columns],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    index = pd.MultiIndex.from_product(
        [users.tolist(), pd.to_datetime(dates)],
        names=["user_id", "date"],
    )
    aligned = frame.set_index(
        ["user_id", "date"]
    ).reindex(index)

    values = aligned[source_columns].to_numpy(
        dtype=np.float64
    ).reshape(len(users), len(dates), len(source_columns))
    source_index = {
        feature: index
        for index, feature in enumerate(source_columns)
    }
    for feature in TIME_FEATURES:
        channel = source_index[feature]
        values[:, :, channel] = np.where(
            values[:, :, channel] >= 0.0,
            values[:, :, channel],
            np.nan,
        )
    for feature in LOG1P_FEATURES:
        channel = source_index[feature]
        finite = np.isfinite(values[:, :, channel])
        values[:, :, channel][finite] = np.log1p(
            np.maximum(values[:, :, channel][finite], 0.0)
        )

    train_rows = values[:, : train_end + 1, :]
    train_active = active[:, : train_end + 1]
    reference = train_rows[train_active]
    medians = np.nanmedian(reference, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing_users, missing_times, missing_channels = np.nonzero(
        active[:, :, None] & ~np.isfinite(values)
    )
    values[missing_users, missing_times, missing_channels] = medians[
        missing_channels
    ]

    output = np.full(
        (len(users), len(dates), len(FEATURES)),
        np.nan,
        dtype=np.float64,
    )
    for output_index, feature in enumerate(FEATURES):
        if feature in source_index:
            output[:, :, output_index] = values[
                :, :, source_index[feature]
            ]
    weekday = pd.to_datetime(dates).weekday.to_numpy()
    weekday_sin = np.sin(2.0 * np.pi * weekday / 7.0)
    weekday_cos = np.cos(2.0 * np.pi * weekday / 7.0)
    output[:, :, FEATURES.index("weekday_sin")] = np.where(
        active,
        weekday_sin[None, :],
        np.nan,
    )
    output[:, :, FEATURES.index("weekday_cos")] = np.where(
        active,
        weekday_cos[None, :],
        np.nan,
    )
    return output.astype(np.float32)


def main() -> None:
    dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    data = load_r42()
    features = build_feature_tensor(
        users,
        dates,
        active,
        train_end,
    )

    users_index, times_index = np.nonzero(
        active[:, : train_end + 1]
    )
    x_train = features[
        users_index,
        times_index,
    ]
    if not np.isfinite(x_train).all():
        raise RuntimeError("Training matrix still contains missing values.")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    start_time = time.time()
    model = IsolationForest(
        n_estimators=500,
        max_samples="auto",
        contamination="auto",
        random_state=20261116,
        n_jobs=-1,
    )
    model.fit(x_train_scaled)

    active_users, active_times = np.nonzero(active)
    x_all = features[active_users, active_times]
    if not np.isfinite(x_all).all():
        raise RuntimeError("Scoring matrix still contains missing values.")
    scores_vector = -model.score_samples(
        scaler.transform(x_all)
    )
    scores = np.full(active.shape, np.nan, dtype=np.float32)
    scores[active_users, active_times] = (
        scores_vector.astype(np.float32)
    )

    rows = evaluate_variant(
        "E11_iforest_27d",
        scores,
        data,
        active,
        train_end,
        val_end,
        20261116,
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "metrics_by_seed.csv",
        index=False,
    )
    summary = (
        frame.groupby(["method", "split", "k"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            event_recall_3d_std=("event_recall_3d", "std"),
            precision_at_k_mean=("precision_at_k", "mean"),
            precision_at_k_std=("precision_at_k", "std"),
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_std=("incident_hits", "std"),
        )
        .sort_values(["split", "k"])
    )
    summary.to_csv(
        OUT / "metric_summary.csv",
        index=False,
    )
    np.savez_compressed(
        OUT / "iforest_scores.npz",
        scores=scores,
        active_mask=active,
        features=np.asarray(FEATURES),
    )
    audit = {
        "method": "IsolationForest",
        "dataset": "CERT r4.2",
        "feature_dimensions": len(FEATURES),
        "features": FEATURES,
        "train_rows": int(len(x_train)),
        "active_rows_all_splits": int(len(x_all)),
        "train_end_index": train_end,
        "validation_end_index": val_end,
        "n_estimators": 500,
        "max_samples": "auto",
        "contamination": "auto",
        "random_state": 20261116,
        "n_jobs": -1,
        "score_convention": (
            "-score_samples: larger means more anomalous"
        ),
        "preprocessing": {
            "count_features": "log1p",
            "time_sentinel": "-1 -> NaN",
            "missing_fill": "training active-row feature median",
            "scaling": "StandardScaler fit on active training rows",
        },
        "labels_used_for_training": False,
        "validation_used_for_tuning": False,
        "test_used_for_model_selection": False,
        "elapsed_seconds": time.time() - start_time,
    }
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
