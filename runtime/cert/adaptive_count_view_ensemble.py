from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
RETRAIN = ROOT / "processed_r42" / "innovation_sim"

K_GRID = [3, 5, 10, 20, 30, 50]
IFOREST_WEIGHT = 0.5
WINDOW_DAYS = 30
THRESHOLD_QUANTILE = 0.95

sys.path.insert(0, str(ROOT))
from conditional_combination_r42 import daily_percentile  # noqa: E402
from adaptive_count_quota import build_adaptive_alerts  # noqa: E402
from hybrid_dynamic_quota_sim import rolling_threshold  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
)


def evaluate_dataset(
    data: dict,
    iforest_view: np.ndarray,
    personal_base: np.ndarray,
    dataset_name: str,
    external: bool,
) -> pd.DataFrame:
    h_score = (
        IFOREST_WEIGHT * iforest_view
        + (1.0 - IFOREST_WEIGHT) * personal_base
    ).astype(np.float32)
    threshold = rolling_threshold(
        h_score,
        data["active_mask"],
        WINDOW_DAYS,
        THRESHOLD_QUANTILE,
    )
    rows = []
    for k in K_GRID:
        alerts, rescue_flags = build_adaptive_alerts(
            data["ts"],
            h_score,
            data["active_mask"],
            k,
            threshold,
            start_index=data["test_start"] if external else 0,
        )
        result = evaluate_alert_system(
            data,
            alerts,
            "adaptive_count_view_ensemble",
            k,
            external=external,
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            metric = compact_metric(
                result, None if external else split
            )
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
                    "method": "adaptive_count_view_ensemble",
                    "split": split,
                    "k": k,
                    "mean_used_q": float(used.sum(axis=0).mean()),
                    "q_cap": k // 2,
                    **metric,
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
    r52 = load_r52()
    frozen52 = np.load(
        ROOT / "processed_r52" / "r5.2_frozen_multiscale_risk.npz",
        allow_pickle=False,
    )
    personal52 = frozen52["point"].astype(np.float32)

    view_scores = np.load(
        OUT / "iforest_view_ensemble_scores.npz",
        allow_pickle=False,
    )
    r42_view_max = np.maximum(
        view_scores["r42_original"],
        view_scores["r42_augmented"],
    )
    r52_view_max = np.maximum(
        view_scores["r52_original"],
        view_scores["r52_augmented"],
    )
    retrain_view_max = np.maximum(
        view_scores["retrained_original"],
        view_scores["retrained_augmented"],
    )

    retrained = dict(r42)
    scores = np.load(
        RETRAIN / "r4.2_quota_retrained_scores.npz",
        allow_pickle=False,
    )
    retrained["ts"] = scores["ts2vec_signal"].astype(np.float32)

    frames = [
        evaluate_dataset(
            r42,
            r42_view_max,
            personal42,
            "r4.2_original",
            False,
        ),
        evaluate_dataset(
            retrained,
            retrain_view_max,
            personal42,
            "r4.2_retrained",
            False,
        ),
        evaluate_dataset(
            r52,
            r52_view_max,
            personal52,
            "r5.2_original_external",
            True,
        ),
    ]
    final = pd.concat(frames, ignore_index=True)
    final.to_csv(
        OUT / "adaptive_count_view_ensemble_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "model": (
            "q_t(K)=min(floor(K/2), number of IF-only candidates with "
            "H_IF>=rolling threshold); H_IF mixes max(original IForest, "
            "augmented IForest) with personal base distance"
        ),
        "fixed_parameters": {
            "iforest_weight": IFOREST_WEIGHT,
            "window_days": WINDOW_DAYS,
            "threshold_quantile": THRESHOLD_QUANTILE,
        },
        "selection_from_event_labels": False,
        "metrics": str(
            OUT / "adaptive_count_view_ensemble_metrics.csv"
        ),
    }
    (OUT / "adaptive_count_view_ensemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
