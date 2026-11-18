from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_HORIZON = 3
COOLDOWN_DAYS = 7
K_GRID = [3, 5, 10, 20, 30, 50]
ALPHA_GRID = [round(0.05 * index, 2) for index in range(1, 11)]

sys.path.insert(0, str(ROOT))
from conditional_combination_r42 import (  # noqa: E402
    daily_percentile,
    incident_hit_ids,
    scenario_macro,
)
from external_eval_r52 import evaluate_external  # noqa: E402
from production_protocol_r42 import evaluate, label_matrix  # noqa: E402


def load_r42() -> dict:
    dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    calibrated = np.load(
        ROOT / "processed_r42" / "r4.2_calibrated_method_scores.npz",
        allow_pickle=False,
    )
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    incidents = pd.read_csv(
        ROOT / "processed_r42" / "r4.2_incidents.csv"
    )
    incidents["start_date"] = pd.to_datetime(
        incidents["start"]
    ).dt.normalize()
    incidents["end_date"] = pd.to_datetime(
        incidents["end"]
    ).dt.normalize()
    return {
        "users": users,
        "dates": dates,
        "active_mask": active_mask,
        "train_end": int(dataset["train_end_index"]),
        "val_end": int(dataset["val_end_index"]),
        "ts": daily_percentile(
            calibrated["ts2vec_quantile_dynamic"].astype(np.float32),
            active_mask,
        ),
        "iforest": daily_percentile(
            calibrated["iforest_quantile_dynamic"].astype(np.float32),
            active_mask,
        ),
        "personal": daily_percentile(
            calibrated["zscore_quantile_dynamic"].astype(np.float32),
            active_mask,
        ),
        "labels": label_matrix(users, dates),
        "incidents": incidents,
    }


def load_r52() -> dict:
    dataset = np.load(
        ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    scores = np.load(
        ROOT
        / "processed_r52"
        / "r5.2_conditional_combination_scores.npz",
        allow_pickle=False,
    )
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    test_start = int(dataset["train_end_index"]) + 1

    labels_frame = pd.read_csv(
        ROOT
        / "processed_r52"
        / "r5.2_malicious_account_days.csv",
        parse_dates=["date"],
    )
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
        ROOT / "processed_r52" / "r5.2_incidents.csv",
        parse_dates=["first_observed_date", "last_observed_date"],
    )
    return {
        "users": users,
        "dates": dates,
        "active_mask": active_mask,
        "test_start": test_start,
        "ts": daily_percentile(
            scores["ts2vec_dynamic"].astype(np.float32),
            active_mask,
        ),
        "iforest": daily_percentile(
            scores["iforest_dynamic"].astype(np.float32),
            active_mask,
        ),
        "personal": daily_percentile(
            scores["zscore_dynamic"].astype(np.float32),
            active_mask,
        ),
        "labels": labels,
        "incidents": incidents,
    }


@lru_cache(maxsize=1)
def load_r42_cached() -> dict:
    """Return the immutable-in-practice r4.2 bundle once per process."""

    return load_r42()


@lru_cache(maxsize=1)
def load_r52_cached() -> dict:
    """Return the immutable-in-practice r5.2 bundle once per process."""

    return load_r52()


def q_from_alpha(k: int, alpha: float) -> int:
    if alpha <= 0:
        return 0
    return min(max(1, int(round(alpha * k))), k // 2)


def external_scenario_macro(
    hit_ids: set[str],
    incidents: pd.DataFrame,
) -> tuple[float, dict[str, dict[str, float | int]]]:
    rows = {}
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


def build_quota_alerts(
    ts_score: np.ndarray,
    iforest_score: np.ndarray,
    active_mask: np.ndarray,
    k: int,
    q: int,
    start_index: int = 0,
) -> np.ndarray:
    if q < 0 or q > k:
        raise ValueError("q must satisfy 0 <= q <= k")
    alerts = np.zeros(ts_score.shape, dtype=np.uint8)
    last_alert = np.full(ts_score.shape[0], -10_000, dtype=np.int32)

    for time_index in range(start_index, ts_score.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= COOLDOWN_DAYS
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
        selected = ts_order[: k - q].astype(int).tolist()
        for user_index in iforest_order:
            if int(user_index) in selected:
                continue
            selected.append(int(user_index))
            if len(selected) >= k:
                break
        if len(selected) < k:
            for user_index in ts_order:
                if int(user_index) in selected:
                    continue
                selected.append(int(user_index))
                if len(selected) >= k:
                    break
        if len(selected) < k:
            for user_index in iforest_order:
                if int(user_index) in selected:
                    continue
                selected.append(int(user_index))
                if len(selected) >= k:
                    break
        selected_array = np.asarray(selected[:k], dtype=np.int64)
        alerts[selected_array, time_index] = 1
        last_alert[selected_array] = time_index
    return alerts


def evaluate_alert_system(
    data: dict,
    alerts: np.ndarray,
    method: str,
    k: int,
    external: bool,
) -> dict:
    if external:
        result = evaluate_external(
            data["users"],
            data["dates"],
            data["active_mask"],
            alerts,
            data["labels"],
            data["incidents"],
            data["test_start"],
            method,
            k,
        )["external_test"]
        macro, scenarios = external_scenario_macro(
            set(result["horizons"]["3"]["incident_ids"]),
            data["incidents"],
        )
        result["scenario_macro_recall_3d"] = macro
        result["scenario_recalls"] = scenarios
        return {"external_test": result}

    result = evaluate(
        data["users"],
        data["dates"],
        data["active_mask"],
        alerts,
        data["labels"],
        data["incidents"],
        data["train_end"],
        data["val_end"],
        method,
        k,
    )
    bounds = {
        "validation": (data["train_end"] + 1, data["val_end"]),
        "test": (data["val_end"] + 1, len(data["dates"]) - 1),
    }
    for split, (start_index, end_index) in bounds.items():
        hit_ids = incident_hit_ids(
            data["users"],
            data["dates"],
            alerts,
            data["incidents"],
            start_index,
            end_index,
        )
        macro, scenarios = scenario_macro(
            hit_ids,
            data["incidents"],
            data["dates"],
            start_index,
            end_index,
        )
        result["splits"][split]["scenario_macro_recall_3d"] = macro
        result["splits"][split]["scenario_recalls"] = scenarios
        result["splits"][split]["incident_ids"] = sorted(hit_ids)
    return result


def compact_metric(
    result: dict,
    split: str | None,
) -> dict:
    section = (
        result["external_test"]
        if split is None
        else result["splits"][split]
    )
    return {
        "incident_hits": section["horizons"]["3"]["incidents_hit"],
        "incident_total": section["horizons"]["3"]["incidents_total"],
        "event_recall_3d": section["horizons"]["3"]["event_recall"],
        "event_recall_1d": section["horizons"]["1"]["event_recall"],
        "precision_at_k": section["precision_at_k"],
        "false_alerts_per_day": section["false_alerts_per_day"],
        "scenario_macro_recall_3d": section[
            "scenario_macro_recall_3d"
        ],
    }


def main() -> None:
    r42 = load_r42()
    r52 = load_r52()

    validation_rows = []
    for alpha in ALPHA_GRID:
        per_k = []
        for k in K_GRID:
            q = q_from_alpha(k, alpha)
            alerts = build_quota_alerts(
                r42["ts"],
                r42["iforest"],
                r42["active_mask"],
                k,
                q,
            )
            result = evaluate_alert_system(
                r42, alerts, f"alpha_{alpha:.2f}", k, external=False
            )
            metric = compact_metric(result, "validation")
            validation_rows.append(
                {
                    "alpha": alpha,
                    "k": k,
                    "q": q,
                    **metric,
                }
            )
            per_k.append(metric["event_recall_3d"])
        for row in validation_rows[-len(K_GRID):]:
            row["mean_validation_recall_3d"] = float(np.mean(per_k))

    validation = pd.DataFrame(validation_rows)
    alpha_summary = (
        validation.groupby("alpha", as_index=False)
        .agg(
            mean_validation_recall_3d=(
                "mean_validation_recall_3d",
                "first",
            ),
            mean_validation_precision=(
                "precision_at_k",
                "mean",
            ),
            min_q=("q", "min"),
        )
        .sort_values(
            [
                "mean_validation_recall_3d",
                "mean_validation_precision",
                "alpha",
            ],
            ascending=[False, False, True],
        )
    )
    selected_alpha = float(alpha_summary.iloc[0]["alpha"])
    selected_validation = validation[
        validation["alpha"] == selected_alpha
    ].copy()

    validation.to_csv(
        OUT / "quota_relationship_alpha_grid.csv",
        index=False,
        encoding="utf-8",
    )
    alpha_summary.to_csv(
        OUT / "quota_relationship_alpha_summary.csv",
        index=False,
        encoding="utf-8",
    )

    final_rows = []
    final_results = []
    for dataset_name, data, external in (
        ("r4.2", r42, False),
        ("r5.2", r52, True),
    ):
        for k in K_GRID:
            configurations = {
                "ts_only": 0,
                "iforest_only": k,
                "quota_model": q_from_alpha(k, selected_alpha),
            }
            for method, q in configurations.items():
                alerts = build_quota_alerts(
                    data["ts"],
                    data["iforest"],
                    data["active_mask"],
                    k,
                    q,
                    start_index=(
                        data["test_start"] if external else 0
                    ),
                )
                result = evaluate_alert_system(
                    data,
                    alerts,
                    f"{dataset_name}_{method}",
                    k,
                    external=external,
                )
                final_results.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "k": k,
                        "q": q,
                        "result": result,
                    }
                )
                if external:
                    metric = compact_metric(result, None)
                    final_rows.append(
                        {
                            "dataset": dataset_name,
                            "split": "external_test",
                            "method": method,
                            "k": k,
                            "q": q,
                            **metric,
                        }
                    )
                else:
                    for split in ("validation", "test"):
                        metric = compact_metric(result, split)
                        final_rows.append(
                            {
                                "dataset": dataset_name,
                                "split": split,
                                "method": method,
                                "k": k,
                                "q": q,
                                **metric,
                            }
                        )

    final = pd.DataFrame(final_rows)
    final.to_csv(
        OUT / "quota_relationship_final_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    (
        OUT / "quota_relationship_results.json"
    ).write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    relation = pd.DataFrame(
        {
            "k": K_GRID,
            "alpha": selected_alpha,
            "q": [q_from_alpha(k, selected_alpha) for k in K_GRID],
        }
    )
    relation["q_over_k"] = relation["q"] / relation["k"]
    relation.to_csv(
        OUT / "quota_relationship_formula.csv",
        index=False,
        encoding="utf-8",
    )

    summary = {
        "research_question": (
            "For TS2Vec-primary and IForest-auxiliary rank allocation, "
            "model q as a function of total daily review budget K."
        ),
        "k_grid": K_GRID,
        "alpha_grid": ALPHA_GRID,
        "selection_split": "r4.2 validation",
        "selection_objective": (
            "Maximize mean 3-day event recall over the K grid; "
            "break ties by mean account-day Precision@K."
        ),
        "selected_alpha": selected_alpha,
        "selected_relation": (
            "q(K)=min(max(1,round(alpha*K)),floor(K/2)); "
            f"alpha={selected_alpha:.2f}"
        ),
        "q_by_k": dict(zip(K_GRID, relation["q"].tolist())),
        "test_used_for_selection": False,
        "r52_used_for_selection": False,
        "labels_used_for_model_fit": False,
        "labels_used_for_formula_selection": "r4.2 validation only",
        "outputs": {
            "alpha_grid": str(
                OUT / "quota_relationship_alpha_grid.csv"
            ),
            "alpha_summary": str(
                OUT / "quota_relationship_alpha_summary.csv"
            ),
            "final_metrics": str(
                OUT / "quota_relationship_final_metrics.csv"
            ),
            "relation": str(
                OUT / "quota_relationship_formula.csv"
            ),
            "detailed_results": str(
                OUT / "quota_relationship_results.json"
            ),
        },
    }
    (OUT / "quota_relationship_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nAlpha summary")
    print(alpha_summary.to_string(index=False), flush=True)
    print("\nSelected relation")
    print(relation.to_string(index=False), flush=True)
    print("\nFinal metrics")
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
