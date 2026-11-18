from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
RETRAIN = ROOT / "processed_r42" / "innovation_sim"

K_GRID = [3, 5, 10, 20, 30, 50]
ALPHA = 0.4
W_GRID = [0.25, 0.50, 0.75, 1.00]
WINDOW_GRID = [30, 60]
THRESHOLD_QUANTILE_GRID = [0.95, 0.98]

sys.path.insert(0, str(ROOT))
from quota_relationship_sim import (  # noqa: E402
    build_quota_alerts,
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)


def hybrid_score(
    iforest_signal: np.ndarray,
    personal_signal: np.ndarray,
    weight_iforest: float,
) -> np.ndarray:
    return (
        weight_iforest * iforest_signal
        + (1.0 - weight_iforest) * personal_signal
    ).astype(np.float32)


def rolling_threshold(
    score: np.ndarray,
    active_mask: np.ndarray,
    window_days: int,
    quantile: float,
) -> np.ndarray:
    thresholds = np.full(score.shape[1], np.nan, dtype=np.float32)
    for time_index in range(score.shape[1]):
        start = max(0, time_index - window_days)
        if start >= time_index:
            continue
        window_values = score[:, start:time_index][
            active_mask[:, start:time_index]
        ]
        window_values = window_values[np.isfinite(window_values)]
        if len(window_values):
            thresholds[time_index] = float(
                np.quantile(window_values, quantile)
            )
    return thresholds


def build_hybrid_alerts(
    ts_score: np.ndarray,
    hybrid_aux_score: np.ndarray,
    active_mask: np.ndarray,
    k: int,
    q_limit: int,
    threshold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alerts = np.zeros(ts_score.shape, dtype=np.uint8)
    rescue_flags = np.zeros(ts_score.shape, dtype=np.uint8)
    used_per_day = np.zeros(ts_score.shape[1], dtype=np.int16)
    last_alert = np.full(ts_score.shape[0], -10_000, dtype=np.int32)

    for time_index in range(ts_score.shape[1]):
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
        selected = ts_order[: k - q_limit].astype(int).tolist()
        ts_top_k = set(ts_order[:k].tolist())

        if (
            q_limit > 0
            and np.isfinite(threshold[time_index])
        ):
            aux_candidates = [
                int(user_index)
                for user_index in candidates
                if int(user_index) not in selected
                and int(user_index) not in ts_top_k
                and hybrid_aux_score[user_index, time_index]
                >= threshold[time_index]
            ]
            aux_candidates.sort(
                key=lambda user_index: (
                    -float(hybrid_aux_score[user_index, time_index]),
                    user_index,
                )
            )
            for user_index in aux_candidates[:q_limit]:
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
        used_per_day[time_index] = int(rescue_flags[:, time_index].sum())

    return alerts, rescue_flags, used_per_day


def load_retrained_r42() -> dict:
    data = load_r42()
    scores = np.load(
        RETRAIN / "r4.2_quota_retrained_scores.npz",
        allow_pickle=False,
    )
    if not np.array_equal(scores["users"].astype(str), data["users"]):
        raise RuntimeError("Retrained users do not match base data.")
    if not np.array_equal(
        pd.to_datetime(scores["dates"].astype(str)), data["dates"]
    ):
        raise RuntimeError("Retrained dates do not match base data.")
    data["ts"] = scores["ts2vec_signal"].astype(np.float32)
    data["iforest"] = scores["iforest_signal"].astype(np.float32)
    return data


def evaluate_configuration(
    data: dict,
    external: bool,
    iforest_weight: float,
    window_days: int,
    threshold_quantile: float,
) -> tuple[pd.DataFrame, dict]:
    aux_score = hybrid_score(
        data["iforest"], data["personal"], iforest_weight
    )
    threshold = rolling_threshold(
        aux_score, data["active_mask"], window_days, threshold_quantile
    )
    rows = []
    for k in K_GRID:
        q_limit = q_from_alpha(k, ALPHA)
        alerts, _, used_per_day = build_hybrid_alerts(
            data["ts"],
            aux_score,
            data["active_mask"],
            k,
            q_limit,
            threshold,
        )
        result = evaluate_alert_system(
            data,
            alerts,
            "hybrid_dynamic_quota",
            k,
            external=external,
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            metric = compact_metric(
                result, None if external else split
            )
            rows.append(
                {
                    "method": "hybrid_dynamic_quota",
                    "split": split,
                    "k": k,
                    "q_limit": q_limit,
                    "mean_used_rescue_slots": (
                        float(np.mean(used_per_day))
                        if external
                        else float(
                            np.mean(
                                used_per_day[
                                    (
                                        data["train_end"] + 1
                                        if split == "validation"
                                        else data["val_end"] + 1
                                    ) : (
                                        data["val_end"] + 1
                                        if split == "validation"
                                        else len(data["dates"])
                                    )
                                ]
                            )
                        )
                    ),
                    **metric,
                }
            )
    return pd.DataFrame(rows), {
        "iforest_weight": iforest_weight,
        "window_days": window_days,
        "threshold_quantile": threshold_quantile,
        "threshold_by_day": threshold,
    }


def main() -> None:
    r42 = load_r42()
    retrained = load_retrained_r42()
    r52 = load_r52()

    validation_rows = []
    config_details = {}
    for iforest_weight in W_GRID:
        for window_days in WINDOW_GRID:
            for threshold_quantile in THRESHOLD_QUANTILE_GRID:
                metrics, details = evaluate_configuration(
                    r42,
                    external=False,
                    iforest_weight=iforest_weight,
                    window_days=window_days,
                    threshold_quantile=threshold_quantile,
                )
                validation_metrics = metrics[
                    metrics["split"] == "validation"
                ]
                config_name = (
                    f"w{iforest_weight:.2f}_w{window_days}"
                    f"_q{threshold_quantile:.2f}"
                )
                config_details[config_name] = details
                for row in validation_metrics.to_dict(orient="records"):
                    validation_rows.append(
                        {
                            **row,
                            "config": config_name,
                            "iforest_weight": iforest_weight,
                            "window_days": window_days,
                            "threshold_quantile": threshold_quantile,
                        }
                    )

    validation = pd.DataFrame(validation_rows)
    config_summary = (
        validation.groupby(
            [
                "config",
                "iforest_weight",
                "window_days",
                "threshold_quantile",
            ],
            as_index=False,
        )
        .agg(
            mean_validation_recall=("event_recall_3d", "mean"),
            mean_validation_precision=("precision_at_k", "mean"),
            mean_validation_macro=(
                "scenario_macro_recall_3d",
                "mean",
            ),
        )
        .sort_values(
            [
                "mean_validation_recall",
                "mean_validation_precision",
                "mean_validation_macro",
            ],
            ascending=False,
        )
    )
    selected = config_summary.iloc[0]
    selected_config = {
        "iforest_weight": float(selected["iforest_weight"]),
        "window_days": int(selected["window_days"]),
        "threshold_quantile": float(
            selected["threshold_quantile"]
        ),
    }

    validation.to_csv(
        OUT / "hybrid_dynamic_validation_grid.csv",
        index=False,
        encoding="utf-8",
    )
    config_summary.to_csv(
        OUT / "hybrid_dynamic_config_summary.csv",
        index=False,
        encoding="utf-8",
    )

    final_rows = []
    for dataset_name, data, external in (
        ("r4.2_original", r42, False),
        ("r4.2_retrained", retrained, False),
        ("r5.2_original_external", r52, True),
    ):
        hybrid_metrics, _ = evaluate_configuration(
            data,
            external=external,
            **selected_config,
        )
        for row in hybrid_metrics.to_dict(orient="records"):
            final_rows.append(
                {
                    "dataset": dataset_name,
                    **row,
                }
            )

        fixed = (
            pd.read_csv(RETRAIN / "r4.2_quota_retrain_metrics.csv")
            if dataset_name == "r4.2_retrained"
            else pd.read_csv(
                OUT / "quota_relationship_final_metrics.csv"
            )
        )
        if dataset_name == "r5.2_original_external":
            fixed = fixed[
                (fixed["dataset"] == "r5.2")
                & (fixed["split"] == "external_test")
            ]
            fixed["dataset"] = dataset_name
        else:
            if "dataset" not in fixed.columns:
                fixed["dataset"] = dataset_name
            else:
                fixed = fixed[fixed["dataset"] == "r4.2"]
                fixed["dataset"] = dataset_name
            fixed["split"] = fixed["split"].replace(
                {"test": "test", "validation": "validation"}
            )
        for row in fixed.to_dict(orient="records"):
            final_rows.append(row)

    final = pd.DataFrame(final_rows)
    final.to_csv(
        OUT / "hybrid_dynamic_final_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "selected_config": selected_config,
        "q_relation": "q_limit=min(max(1,round(0.4K)),floor(K/2))",
        "selection_split": "r4.2 original validation",
        "selection_objective": (
            "maximize mean event Recall@K over K grid; "
            "tie-break mean Precision@K and scenario macro recall"
        ),
        "test_and_external_used_for_selection": False,
        "validation_grid": str(
            OUT / "hybrid_dynamic_validation_grid.csv"
        ),
        "config_summary": str(
            OUT / "hybrid_dynamic_config_summary.csv"
        ),
        "final_metrics": str(
            OUT / "hybrid_dynamic_final_metrics.csv"
        ),
    }
    (OUT / "hybrid_dynamic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nSelected validation metrics")
    print(
        validation[
            validation["config"] == str(selected["config"])
        ].to_string(index=False),
        flush=True,
    )
    print("\nFinal comparison")
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
