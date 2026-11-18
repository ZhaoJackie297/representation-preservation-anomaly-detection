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
ALPHA = 0.4

sys.path.insert(0, str(ROOT))
from conditional_combination_r42 import daily_percentile  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)


def build_rule_alerts(
    ts_score: np.ndarray,
    iforest_score: np.ndarray,
    personal_base: np.ndarray,
    active_mask: np.ndarray,
    k: int,
    q_limit: int,
    rule: str,
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
        iforest_order = candidates[
            np.lexsort(
                (candidates, -iforest_score[candidates, time_index])
            )
        ]
        selected = ts_order[: k - q_limit].astype(int).tolist()
        ts_top_k = set(ts_order[:k].tolist())

        for user_index in iforest_order:
            user_index = int(user_index)
            if user_index in selected or user_index in ts_top_k:
                continue
            p_value = float(personal_base[user_index, time_index])
            t_value = float(ts_score[user_index, time_index])
            passes = (
                rule == "none"
                or (rule == "personal80" and p_value >= 0.80)
                or (rule == "personal90" and p_value >= 0.90)
                or (rule == "ts90" and t_value >= 0.90)
                or (rule == "ts95" and t_value >= 0.95)
                or (
                    rule == "personal80_ts90"
                    and p_value >= 0.80
                    and t_value >= 0.90
                )
                or (
                    rule == "personal90_ts90"
                    and p_value >= 0.90
                    and t_value >= 0.90
                )
            )
            if not passes:
                continue
            selected.append(user_index)
            rescue_flags[user_index, time_index] = 1
            if len(selected) >= k:
                break

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


def load_retrained() -> dict:
    data = load_r42()
    scores = np.load(
        RETRAIN / "r4.2_quota_retrained_scores.npz",
        allow_pickle=False,
    )
    data["ts"] = scores["ts2vec_signal"].astype(np.float32)
    data["iforest"] = scores["iforest_signal"].astype(np.float32)
    return data


def evaluate_rule(
    data: dict,
    rule: str,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        q_limit = q_from_alpha(k, ALPHA)
        alerts, _, used_per_day = build_rule_alerts(
            data["ts"],
            data["iforest"],
            data["personal_base"],
            data["active_mask"],
            k,
            q_limit,
            rule,
        )
        result = evaluate_alert_system(
            data,
            alerts,
            f"rule_{rule}",
            k,
            external=external,
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            metric = compact_metric(
                result, None if external else split
            )
            if external:
                used = float(np.mean(used_per_day))
            else:
                start = (
                    data["train_end"] + 1
                    if split == "validation"
                    else data["val_end"] + 1
                )
                end = (
                    data["val_end"] + 1
                    if split == "validation"
                    else len(data["dates"])
                )
                used = float(np.mean(used_per_day[start:end]))
            rows.append(
                {
                    "method": f"rule_{rule}",
                    "rule": rule,
                    "split": split,
                    "k": k,
                    "q_limit": q_limit,
                    "mean_used_rescue_slots": used,
                    **metric,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    r42 = load_r42()
    r42["personal_base"] = daily_percentile(
        np.load(
            ROOT / "processed_r42" / "r4.2_calibrated_method_scores.npz",
            allow_pickle=False,
        )["zscore_quantile"].astype(np.float32),
        r42["active_mask"],
    )
    retrained = load_retrained()
    retrained["personal_base"] = r42["personal_base"]
    r52 = load_r52()
    r52["personal_base"] = daily_percentile(
        np.load(
            ROOT / "processed_r52" / "r5.2_conditional_combination_scores.npz",
            allow_pickle=False,
        )["zscore_dynamic"].astype(np.float32),
        r52["active_mask"],
    )

    rules = [
        "none",
        "personal80",
        "personal90",
        "ts90",
        "ts95",
        "personal80_ts90",
        "personal90_ts90",
    ]
    validation_rows = []
    for rule in rules:
        metrics = evaluate_rule(r42, rule, external=False)
        validation = metrics[metrics["split"] == "validation"].copy()
        validation["mean_validation_recall"] = validation[
            "event_recall_3d"
        ].mean()
        validation["mean_validation_precision"] = validation[
            "precision_at_k"
        ].mean()
        validation["mean_validation_macro"] = validation[
            "scenario_macro_recall_3d"
        ].mean()
        validation_rows.append(validation)
    validation = pd.concat(validation_rows, ignore_index=True)
    summary = (
        validation.groupby("rule", as_index=False)
        .agg(
            mean_validation_recall=(
                "mean_validation_recall",
                "first",
            ),
            mean_validation_precision=(
                "mean_validation_precision",
                "first",
            ),
            mean_validation_macro=(
                "mean_validation_macro",
                "first",
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
    selected_rule = str(summary.iloc[0]["rule"])
    validation.to_csv(
        OUT / "quota_rule_validation_grid.csv",
        index=False,
        encoding="utf-8",
    )
    summary.to_csv(
        OUT / "quota_rule_config_summary.csv",
        index=False,
        encoding="utf-8",
    )

    final_rows = []
    for dataset_name, data, external in (
        ("r4.2_original", r42, False),
        ("r4.2_retrained", retrained, False),
        ("r5.2_original_external", r52, True),
    ):
        metrics = evaluate_rule(data, selected_rule, external)
        metrics["dataset"] = dataset_name
        final_rows.append(metrics)

        fixed_path = (
            RETRAIN / "r4.2_quota_retrain_metrics.csv"
            if dataset_name == "r4.2_retrained"
            else OUT / "quota_relationship_final_metrics.csv"
        )
        fixed = pd.read_csv(fixed_path)
        if dataset_name == "r5.2_original_external":
            fixed = fixed[
                (fixed["dataset"] == "r5.2")
                & (fixed["split"] == "external_test")
            ].copy()
            fixed["dataset"] = dataset_name
        else:
            if "dataset" in fixed.columns:
                fixed = fixed[fixed["dataset"] == "r4.2"].copy()
            fixed["dataset"] = dataset_name
        final_rows.append(fixed)

    final = pd.concat(final_rows, ignore_index=True)
    final.to_csv(
        OUT / "quota_rule_final_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    output_summary = {
        "selected_rule": selected_rule,
        "selection_split": "r4.2 original validation",
        "selection_objective": (
            "mean event Recall@K over K grid, tie-break Precision and macro"
        ),
        "test_and_external_used_for_selection": False,
        "rules": rules,
        "outputs": {
            "validation_grid": str(
                OUT / "quota_rule_validation_grid.csv"
            ),
            "config_summary": str(
                OUT / "quota_rule_config_summary.csv"
            ),
            "final_metrics": str(
                OUT / "quota_rule_final_metrics.csv"
            ),
        },
    }
    (OUT / "quota_rule_summary.json").write_text(
        json.dumps(output_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output_summary, ensure_ascii=False, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
