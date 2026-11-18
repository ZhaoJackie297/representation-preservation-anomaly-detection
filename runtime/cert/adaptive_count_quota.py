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
WEIGHT_IFOREST = 0.5
WINDOW_DAYS = 30
THRESHOLD_QUANTILE = 0.95

sys.path.insert(0, str(ROOT))
from conditional_combination_r42 import daily_percentile  # noqa: E402
from hybrid_dynamic_quota_sim import rolling_threshold  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
)


def load_retrained() -> dict:
    data = load_r42()
    scores = np.load(
        RETRAIN / "r4.2_quota_retrained_scores.npz",
        allow_pickle=False,
    )
    data["ts"] = scores["ts2vec_signal"].astype(np.float32)
    data["iforest"] = scores["iforest_signal"].astype(np.float32)
    return data


def build_adaptive_alerts(
    ts_score: np.ndarray,
    h_iforest: np.ndarray,
    active_mask: np.ndarray,
    k: int,
    threshold: np.ndarray,
    start_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    alerts = np.zeros(ts_score.shape, dtype=np.uint8)
    rescue_flags = np.zeros(ts_score.shape, dtype=np.uint8)
    last_alert = np.full(ts_score.shape[0], -10_000, dtype=np.int32)
    q_cap = k // 2

    for time_index in range(start_index, ts_score.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= 7
        )
        candidates = np.flatnonzero(available)
        if len(candidates) == 0:
            continue
        ts_order = candidates[
            np.lexsort(
                (candidates, -ts_score[candidates, time_index])
            )
        ]
        ts_top_k = set(ts_order[:k].tolist())

        eligible = []
        if np.isfinite(threshold[time_index]):
            for user_index in candidates:
                user_index = int(user_index)
                if user_index in ts_top_k:
                    continue
                if (
                    h_iforest[user_index, time_index]
                    >= threshold[time_index]
                ):
                    eligible.append(user_index)
        eligible.sort(
            key=lambda user_index: (
                -float(h_iforest[user_index, time_index]),
                user_index,
            )
        )
        q_t = min(len(eligible), q_cap)
        selected = ts_order[: k - q_t].astype(int).tolist()
        for user_index in eligible[:q_t]:
            selected.append(user_index)
            rescue_flags[user_index, time_index] = 1
        if len(selected) < k:
            for user_index in ts_order:
                if int(user_index) not in selected:
                    selected.append(int(user_index))
                if len(selected) >= k:
                    break
        selected_array = np.asarray(selected[:k], dtype=np.int64)
        alerts[selected_array, time_index] = 1
        last_alert[selected_array] = time_index
    return alerts, rescue_flags


def hybrid_score(
    iforest_signal: np.ndarray,
    personal_base: np.ndarray,
) -> np.ndarray:
    return (
        WEIGHT_IFOREST * iforest_signal
        + (1.0 - WEIGHT_IFOREST) * personal_base
    ).astype(np.float32)


def evaluate_data(
    data: dict,
    external: bool,
    window_days: int = WINDOW_DAYS,
    threshold_quantile: float = THRESHOLD_QUANTILE,
) -> pd.DataFrame:
    h_score = hybrid_score(
        data["iforest"], data["personal_base"]
    )
    threshold = rolling_threshold(
        h_score,
        data["active_mask"],
        window_days,
        threshold_quantile,
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
            "adaptive_count_quota",
            k,
            external=external,
        )
        split_names = (
            ["external_test"] if external else ["validation", "test"]
        )
        for split in split_names:
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
                    "method": "adaptive_count_quota",
                    "split": split,
                    "k": k,
                    "mean_used_q": float(used.sum(axis=0).mean()),
                    "max_used_q": int(used.sum(axis=0).max()),
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
    r42["personal_base"] = daily_percentile(
        calibrated42["zscore_quantile"].astype(np.float32),
        r42["active_mask"],
    )

    retrained = load_retrained()
    retrained["personal_base"] = r42["personal_base"]

    r52 = load_r52()
    frozen52 = np.load(
        ROOT
        / "processed_r52"
        / "r5.2_frozen_multiscale_risk.npz",
        allow_pickle=False,
    )
    r52["personal_base"] = frozen52["point"].astype(np.float32)

    datasets = [
        ("r4.2_original", r42, False),
        ("r4.2_retrained", retrained, False),
        ("r5.2_original_external", r52, True),
    ]
    final_rows = []
    for dataset_name, data, external in datasets:
        metrics = evaluate_data(data, external)
        metrics["dataset"] = dataset_name
        final_rows.append(metrics)
    final = pd.concat(final_rows, ignore_index=True)
    final.to_csv(
        OUT / "adaptive_count_quota_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    sensitivity_rows = []
    for window_days in (30, 60):
        for threshold_quantile in (0.95, 0.98):
            for dataset_name, data, external in datasets:
                metrics = evaluate_data(
                    data,
                    external,
                    window_days=window_days,
                    threshold_quantile=threshold_quantile,
                )
                for row in metrics.to_dict(orient="records"):
                    sensitivity_rows.append(
                        {
                            "dataset": dataset_name,
                            "window_days": window_days,
                            "threshold_quantile": threshold_quantile,
                            **row,
                        }
                    )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        OUT / "adaptive_count_quota_sensitivity.csv",
        index=False,
        encoding="utf-8",
    )

    summary = {
        "model": (
            "q_t(K)=min(floor(K/2), count of IF-only candidates "
            "with H_IF>=rolling threshold); TS fills remaining slots"
        ),
        "fixed_parameters": {
            "iforest_weight": WEIGHT_IFOREST,
            "window_days": WINDOW_DAYS,
            "threshold_quantile": THRESHOLD_QUANTILE,
        },
        "sensitivity": {
            "window_days": [30, 60],
            "threshold_quantile": [0.95, 0.98],
        },
        "labels_used_for_threshold": False,
        "labels_used_for_model_fit": False,
        "metrics": str(OUT / "adaptive_count_quota_metrics.csv"),
        "sensitivity_file": str(
            OUT / "adaptive_count_quota_sensitivity.csv"
        ),
    }
    (OUT / "adaptive_count_quota_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
