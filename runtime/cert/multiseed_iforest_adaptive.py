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
from feature_augmented_iforest_sim import (  # noqa: E402
    augment,
    build_unique_quota_alerts,
    load_feature_values,
    train_iforest,
)
from adaptive_count_quota import build_adaptive_alerts  # noqa: E402
from hybrid_dynamic_quota_sim import rolling_threshold  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)


def fit_multiseed_views(
    data: dict,
    feature_file: Path,
    ts_dynamic: np.ndarray,
    personal_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import feature_augmented_iforest_sim as source

    source.users = data["users"]
    source.dates = data["dates"]
    base_values = load_feature_values(data, feature_file)
    augmented_values = augment(
        base_values,
        [ts_dynamic[..., None], personal_signal[..., None]],
    )
    train_end = (
        data["train_end"]
        if "train_end" in data
        else data["test_start"] - 1
    )
    original_scores = []
    augmented_scores = []
    for seed in SEEDS:
        print(f"training seed {seed}", flush=True)
        original_scores.append(
            train_iforest(
                base_values,
                data["active_mask"],
                train_end,
                seed,
            )
        )
        augmented_scores.append(
            train_iforest(
                augmented_values,
                data["active_mask"],
                train_end,
                seed,
            )
        )
    original_avg = np.mean(original_scores, axis=0).astype(np.float32)
    augmented_avg = np.mean(augmented_scores, axis=0).astype(np.float32)
    return original_avg, augmented_avg


def build_scores(
    ts_signal: np.ndarray,
    original_iforest: np.ndarray,
    augmented_iforest: np.ndarray,
    personal_base: np.ndarray,
) -> dict[str, np.ndarray]:
    view_max = np.maximum(original_iforest, augmented_iforest)
    view_mean = (original_iforest + augmented_iforest) / 2.0
    return {
        "ts_only": ts_signal,
        "iforest_original_5seed": original_iforest,
        "iforest_augmented_5seed": augmented_iforest,
        "iforest_view_max_5seed": view_max,
        "iforest_view_mean_5seed": view_mean,
        "hif_view_max": (
            IFOREST_WEIGHT * view_max
            + (1.0 - IFOREST_WEIGHT) * personal_base
        ).astype(np.float32),
    }


def evaluate_dataset(
    data: dict,
    scores: dict[str, np.ndarray],
    dataset_name: str,
    external: bool,
) -> pd.DataFrame:
    threshold = rolling_threshold(
        scores["hif_view_max"],
        data["active_mask"],
        WINDOW_DAYS,
        THRESHOLD_QUANTILE,
    )
    rows = []
    for k in K_GRID:
        methods = {}
        alerts_ts = build_unique_quota_alerts(
            scores["ts_only"],
            scores["ts_only"],
            data["active_mask"],
            k,
            0,
        )
        methods["ts_only"] = alerts_ts
        alerts_if = build_unique_quota_alerts(
            scores["iforest_original_5seed"],
            scores["iforest_original_5seed"],
            data["active_mask"],
            k,
            k,
        )
        methods["iforest_original_5seed"] = alerts_if
        alerts_fixed = build_unique_quota_alerts(
            scores["ts_only"],
            scores["iforest_view_max_5seed"],
            data["active_mask"],
            k,
            q_from_alpha(k, 0.4),
        )
        methods["fixed_q_0.4K"] = alerts_fixed
        alerts_adaptive, _ = build_adaptive_alerts(
            scores["ts_only"],
            scores["hif_view_max"],
            data["active_mask"],
            k,
            threshold,
            start_index=data["test_start"] if external else 0,
        )
        methods["adaptive_count_5seed"] = alerts_adaptive

        for method, alerts in methods.items():
            result = evaluate_alert_system(
                data,
                alerts,
                method,
                k,
                external=external,
            )
            split_names = (
                ["external_test"] if external else ["validation", "test"]
            )
            for split in split_names:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "split": split,
                        "k": k,
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
    personal42 = (
        calibrated42["zscore_quantile"].astype(np.float32)
    )
    from conditional_combination_r42 import daily_percentile

    personal42 = daily_percentile(personal42, r42["active_mask"])
    original42, augmented42 = fit_multiseed_views(
        r42,
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        calibrated42["ts2vec_quantile_dynamic"].astype(np.float32),
        personal42,
    )
    scores42 = build_scores(
        r42["ts"], original42, augmented42, personal42
    )
    original_metrics = evaluate_dataset(
        r42, scores42, "r4.2_original", external=False
    )

    r52 = load_r52()
    scores52_npz = np.load(
        ROOT / "processed_r52" / "r5.2_conditional_combination_scores.npz",
        allow_pickle=False,
    )
    frozen52 = np.load(
        ROOT / "processed_r52" / "r5.2_frozen_multiscale_risk.npz",
        allow_pickle=False,
    )
    personal52 = frozen52["point"].astype(np.float32)
    original52, augmented52 = fit_multiseed_views(
        r52,
        ROOT / "processed_r52" / "r5.2_user_day_features.csv",
        scores52_npz["ts2vec_dynamic"].astype(np.float32),
        personal52,
    )
    scores52 = build_scores(
        r52["ts"], original52, augmented52, personal52
    )
    external_metrics = evaluate_dataset(
        r52, scores52, "r5.2_original_external", external=True
    )

    retrained = np.load(
        RETRAIN / "r4.2_quota_retrained_scores.npz",
        allow_pickle=False,
    )
    original_retrain, augmented_retrain = fit_multiseed_views(
        r42,
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        retrained["ts2vec_dynamic"].astype(np.float32),
        personal42,
    )
    scores_retrain = build_scores(
        retrained["ts2vec_signal"].astype(np.float32),
        original_retrain,
        augmented_retrain,
        personal42,
    )
    retrain_metrics = evaluate_dataset(
        r42, scores_retrain, "r4.2_retrained", external=False
    )

    final = pd.concat(
        [original_metrics, external_metrics, retrain_metrics],
        ignore_index=True,
    )
    final.to_csv(
        OUT / "multiseed_iforest_adaptive_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "multiseed_iforest_adaptive_scores.npz",
        **{
            f"r42_{name}": value
            for name, value in scores42.items()
        },
        **{
            f"r52_{name}": value
            for name, value in scores52.items()
        },
        **{
            f"retrain_{name}": value
            for name, value in scores_retrain.items()
        },
    )
    summary = {
        "design": (
            "Five IForest seeds per view and fixed adaptive-count q model; "
            "no alpha or threshold tuning on test/external."
        ),
        "seeds": SEEDS,
        "k_grid": K_GRID,
        "adaptive_rule": (
            "q_t(K)=min(floor(K/2), number of IF-only candidates with "
            "H_IF>=past-30-day 95th percentile)"
        ),
        "hif": (
            "0.5*max(mean original IForest seeds, mean augmented IForest "
            "seeds)+0.5*personal-base percentile"
        ),
        "metrics": str(
            OUT / "multiseed_iforest_adaptive_metrics.csv"
        ),
        "scores": str(
            OUT / "multiseed_iforest_adaptive_scores.npz"
        ),
    }
    (OUT / "multiseed_iforest_adaptive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
