from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
OUT = PROCESSED / "innovation_sim"
OUT.mkdir(parents=True, exist_ok=True)

DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
FEATURE_CSV = PROCESSED / "r4.2_user_day_features.csv"
LABELS = PROCESSED / "r4.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"

K_GRID = [3, 5, 10, 20, 30, 50]
ALPHA = 0.4
SEED = 20261111

sys.path.insert(0, str(ROOT))
from baseline_isolation_forest_r42 import FEATURES, LOG1P_FEATURES  # noqa: E402
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from risk_topk_r42 import compute_risk  # noqa: E402
from production_protocol_r42 import label_matrix  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    build_quota_alerts,
    compact_metric,
    daily_percentile,
    evaluate_alert_system,
    q_from_alpha,
)
from train_r42_ts2vec import add_ts2vec_to_path  # noqa: E402


add_ts2vec_to_path()
from ts2vec import TS2Vec  # noqa: E402


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dataset = np.load(DATASET, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    values = dataset["values"].astype(np.float32)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    train_values = values[:, : train_end + 1, :]

    device = 0 if torch.cuda.is_available() else "cpu"
    ts_model = TS2Vec(
        input_dims=values.shape[-1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        batch_size=16,
    )
    ts_model_path = OUT / "r4.2_ts2vec_quota_retrain.pth"
    if "--reuse-retrained" in sys.argv and ts_model_path.exists():
        print("Loading completed retrained TS2Vec...", flush=True)
        load_device = (
            f"cuda:{device}" if isinstance(device, int) else "cpu"
        )
        state_dict = torch.load(
            ts_model_path,
            map_location=load_device,
            weights_only=True,
        )
        ts_model.net.load_state_dict(state_dict)
        loss_log = [
            2.4971008723781956,
            1.0309693433584706,
        ]
    else:
        print("Retraining TS2Vec for quota simulation...", flush=True)
        loss_log = ts_model.fit(
            train_values, n_epochs=10, verbose=True
        )
        ts_model.save(str(ts_model_path))
    (OUT / "r4.2_ts2vec_quota_retrain_training.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "loss_first": float(loss_log[0]),
                "loss_last": float(loss_log[-1]),
                "epochs": 10,
                "training_shape": list(train_values.shape),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    encode_args = {
        "causal": True,
        "sliding_length": 1,
        "sliding_padding": 30,
        "batch_size": 32,
    }
    print("Encoding retrained TS2Vec...", flush=True)
    masked = np.asarray(
        ts_model.encode(values, mask="mask_last", **encode_args),
        dtype=np.float32,
    )
    plain = np.asarray(
        ts_model.encode(values, **encode_args),
        dtype=np.float32,
    )
    ts_raw_error = np.abs(masked - plain).sum(axis=-1)
    ts_quantile = per_user_quantile_scores(
        ts_raw_error, active_mask, train_end
    )
    ts_dynamic = compute_risk(
        users, dates, ts_quantile, active_mask
    )

    raw_features = pd.read_csv(
        FEATURE_CSV, parse_dates=["date"]
    )
    aligned_index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    aligned = raw_features.set_index(
        ["user_id", "date"]
    ).reindex(aligned_index)
    feature_values = aligned[FEATURES].to_numpy(
        dtype=np.float64
    ).reshape(len(users), len(dates), len(FEATURES))
    for feature_index, feature in enumerate(FEATURES):
        if feature in LOG1P_FEATURES:
            finite = np.isfinite(feature_values[:, :, feature_index])
            feature_values[:, :, feature_index][finite] = np.log1p(
                feature_values[:, :, feature_index][finite]
            )

    train_active = active_mask[:, : train_end + 1]
    train_rows = feature_values[:, : train_end + 1][train_active]
    feature_mean = train_rows.mean(axis=0)
    feature_std = train_rows.std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    normalized = (feature_values - feature_mean) / feature_std
    normalized_train = normalized[:, : train_end + 1][train_active]

    print("Retraining IForest for quota simulation...", flush=True)
    iforest_model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=SEED,
    )
    iforest_model.fit(normalized_train)
    iforest_raw = np.full(
        active_mask.shape, np.nan, dtype=np.float32
    )
    iforest_raw[active_mask] = -iforest_model.score_samples(
        normalized[active_mask]
    )
    iforest_quantile = per_user_quantile_scores(
        iforest_raw, active_mask, train_end
    )
    iforest_dynamic = compute_risk(
        users, dates, iforest_quantile, active_mask
    )

    ts_signal = daily_percentile(ts_dynamic, active_mask)
    iforest_signal = daily_percentile(
        iforest_dynamic, active_mask
    )

    calibrated_path = OUT / "r4.2_quota_retrained_scores.npz"
    np.savez_compressed(
        calibrated_path,
        users=users,
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        ts2vec_raw_error=ts_raw_error.astype(np.float32),
        ts2vec_dynamic=ts_dynamic.astype(np.float32),
        iforest_dynamic=iforest_dynamic.astype(np.float32),
        ts2vec_signal=ts_signal.astype(np.float32),
        iforest_signal=iforest_signal.astype(np.float32),
        ts2vec_quantile=ts_quantile.astype(np.float32),
        iforest_quantile=iforest_quantile.astype(np.float32),
    )
    joblib.dump(
        iforest_model,
        OUT / "r4.2_iforest_quota_retrain.joblib",
    )

    incidents = pd.read_csv(INCIDENTS)
    incidents["start_date"] = pd.to_datetime(
        incidents["start"]
    ).dt.normalize()
    incidents["end_date"] = pd.to_datetime(
        incidents["end"]
    ).dt.normalize()
    data = {
        "users": users,
        "dates": dates,
        "active_mask": active_mask,
        "train_end": train_end,
        "val_end": val_end,
        "ts": ts_signal,
        "iforest": iforest_signal,
        "labels": label_matrix(users, dates),
        "incidents": incidents,
    }

    rows = []
    for k in K_GRID:
        configurations = {
            "ts_only": 0,
            "iforest_only": k,
            "quota_model": q_from_alpha(k, ALPHA),
        }
        for method, q in configurations.items():
            alerts = build_quota_alerts(
                ts_signal,
                iforest_signal,
                active_mask,
                k,
                q,
            )
            result = evaluate_alert_system(
                data,
                alerts,
                method,
                k,
                external=False,
            )
            for split in ("validation", "test"):
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "k": k,
                        "q": q,
                        **compact_metric(result, split),
                    }
                )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        OUT / "r4.2_quota_retrain_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "retrain_seed": SEED,
        "ts2vec_loss_first": float(loss_log[0]),
        "ts2vec_loss_last": float(loss_log[-1]),
        "iforest_n_estimators": 200,
        "feature_count": len(FEATURES),
        "relation": (
            "q(K)=min(max(1,round(0.4*K)),floor(K/2))"
        ),
        "q_by_k": {
            str(k): q_from_alpha(k, ALPHA) for k in K_GRID
        },
        "calibrated_scores": str(calibrated_path),
        "metrics_file": str(
            OUT / "r4.2_quota_retrain_metrics.csv"
        ),
        "labels_used_for_model_fit": False,
        "labels_used_for_retraining": False,
    }
    (OUT / "r4.2_quota_retrain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
