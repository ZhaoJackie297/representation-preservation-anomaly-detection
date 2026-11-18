from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
RETRAIN = ROOT / "processed_r42" / "innovation_sim"
SEEDS = [20261109, 20261110, 20261111, 20261112, 20261113]
K_GRID = [3, 5, 10, 20, 30, 50]
WINDOW_DAYS = 30
THRESHOLD_QUANTILE = 0.95
IFOREST_WEIGHT = 0.5

sys.path.insert(0, str(ROOT))
from conditional_combination_r42 import daily_percentile  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    augment,
    load_feature_values,
    standardize_augmented_features,
)
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from risk_topk_r42 import compute_risk  # noqa: E402
from hybrid_dynamic_quota_sim import rolling_threshold  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
)
from adaptive_count_quota import build_adaptive_alerts  # noqa: E402


def train_iforest_raw(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    seed: int,
) -> np.ndarray:
    from sklearn.ensemble import IsolationForest

    normalized = standardize_augmented_features(
        values, active_mask, train_end
    )
    train_rows = normalized[:, : train_end + 1][
        active_mask[:, : train_end + 1]
    ]
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(train_rows)
    raw = np.full(active_mask.shape, np.nan, dtype=np.float32)
    raw[active_mask] = -model.score_samples(normalized[active_mask])
    return raw


def average_raw_scores(
    raw_scores: list[np.ndarray],
    active_mask: np.ndarray,
) -> np.ndarray:
    average = np.nanmean(np.stack(raw_scores, axis=0), axis=0)
    average[~active_mask] = np.nan
    return average.astype(np.float32)


def calibrate_raw_iforest(
    raw: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    quantile = per_user_quantile_scores(
        raw, active_mask, train_end
    )
    dynamic = compute_risk(users, dates, quantile, active_mask)
    return daily_percentile(dynamic, active_mask)


def fit_multiseed_raw_views(
    data: dict,
    feature_path: Path,
    ts_dynamic: np.ndarray,
    personal_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import feature_augmented_iforest_sim as source

    source.users = data["users"]
    source.dates = data["dates"]
    base_values = load_feature_values(data, feature_path)
    augmented_values = augment(
        base_values,
        [ts_dynamic[..., None], personal_base[..., None]],
    )
    train_end = (
        data["train_end"]
        if "train_end" in data
        else data["test_start"] - 1
    )
    original_raw = []
    augmented_raw = []
    for seed in SEEDS:
        print(f"raw seed {seed}", flush=True)
        original_raw.append(
            train_iforest_raw(
                base_values,
                data["active_mask"],
                train_end,
                seed,
            )
        )
        augmented_raw.append(
            train_iforest_raw(
                augmented_values,
                data["active_mask"],
                train_end,
                seed,
            )
        )
    original = calibrate_raw_iforest(
        average_raw_scores(original_raw, data["active_mask"]),
        data["active_mask"],
        train_end,
        data["users"],
        data["dates"],
    )
    augmented = calibrate_raw_iforest(
        average_raw_scores(augmented_raw, data["active_mask"]),
        data["active_mask"],
        train_end,
        data["users"],
        data["dates"],
    )
    return original, augmented


def evaluate_view_model(
    data: dict,
    ts_signal: np.ndarray,
    original: np.ndarray,
    augmented: np.ndarray,
    personal_base: np.ndarray,
    dataset_name: str,
    external: bool,
) -> pd.DataFrame:
    iforest_view = np.maximum(original, augmented)
    h_if = (
        IFOREST_WEIGHT * iforest_view
        + (1.0 - IFOREST_WEIGHT) * personal_base
    ).astype(np.float32)
    threshold = rolling_threshold(
        h_if,
        data["active_mask"],
        WINDOW_DAYS,
        THRESHOLD_QUANTILE,
    )
    rows = []
    for k in K_GRID:
        alerts, rescue_flags = build_adaptive_alerts(
            ts_signal,
            h_if,
            data["active_mask"],
            k,
            threshold,
            start_index=data["test_start"] if external else 0,
        )
        result = evaluate_alert_system(
            data,
            alerts,
            "raw_multiseed_adaptive",
            k,
            external=external,
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            if external:
                used = rescue_flags[:, data["test_start"] :]
            elif split == "validation":
                used = rescue_flags[
                    :, data["train_end"] + 1 : data["val_end"] + 1
                ]
            else:
                used = rescue_flags[:, data["val_end"] + 1 :]
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": "raw_multiseed_adaptive",
                    "split": split,
                    "k": k,
                    "mean_used_q": float(used.sum(axis=0).mean()),
                    **compact_metric(
                        result, None if external else split
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    r42 = load_r42()
    calibrated42 = np.load(
        ROOT / "processed_r42" / "r4.2_calibrated_method_scores.npz",
        allow_pickle=False,
    )
    personal42 = daily_percentile(
        calibrated42["zscore_quantile"].astype(np.float32),
        r42["active_mask"],
    )
    original42, augmented42 = fit_multiseed_raw_views(
        r42,
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        calibrated42["ts2vec_quantile_dynamic"].astype(np.float32),
        personal42,
    )
    metrics42 = evaluate_view_model(
        r42,
        r42["ts"],
        original42,
        augmented42,
        personal42,
        "r4.2_original",
        False,
    )

    r52 = load_r52()
    scores52 = np.load(
        ROOT / "processed_r52" / "r5.2_conditional_combination_scores.npz",
        allow_pickle=False,
    )
    frozen52 = np.load(
        ROOT / "processed_r52" / "r5.2_frozen_multiscale_risk.npz",
        allow_pickle=False,
    )
    personal52 = frozen52["point"].astype(np.float32)
    original52, augmented52 = fit_multiseed_raw_views(
        r52,
        ROOT / "processed_r52" / "r5.2_user_day_features.csv",
        scores52["ts2vec_dynamic"].astype(np.float32),
        personal52,
    )
    metrics52 = evaluate_view_model(
        r52,
        r52["ts"],
        original52,
        augmented52,
        personal52,
        "r5.2_original_external",
        True,
    )

    retrained = np.load(
        RETRAIN / "r4.2_quota_retrained_scores.npz",
        allow_pickle=False,
    )
    original_retrain, augmented_retrain = fit_multiseed_raw_views(
        r42,
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        retrained["ts2vec_dynamic"].astype(np.float32),
        personal42,
    )
    retrain_data = dict(r42)
    retrain_data["ts"] = retrained["ts2vec_signal"].astype(np.float32)
    metrics_retrain = evaluate_view_model(
        retrain_data,
        retrain_data["ts"],
        original_retrain,
        augmented_retrain,
        personal42,
        "r4.2_retrained",
        False,
    )

    final = pd.concat(
        [metrics42, metrics52, metrics_retrain],
        ignore_index=True,
    )
    final.to_csv(
        OUT / "multiseed_raw_iforest_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "design": (
            "Average raw IForest anomaly scores across five seeds before "
            "per-user calibration and dynamic risk."
        ),
        "seeds": SEEDS,
        "adaptive_q": (
            "q_t(K)=min(floor(K/2), IF-only candidates above past-30-day "
            "95th percentile of H_IF)"
        ),
        "metrics": str(
            OUT / "multiseed_raw_iforest_metrics.csv"
        ),
    }
    (OUT / "multiseed_raw_iforest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
