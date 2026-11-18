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
ALPHA_GRID = [round(0.05 * index, 2) for index in range(1, 11)]

sys.path.insert(0, str(ROOT))
from quota_relationship_sim import (  # noqa: E402
    build_quota_alerts,
    incident_hit_ids,
    load_r42,
    q_from_alpha,
)


def monthly_bounds(
    dates: pd.DatetimeIndex,
    start_index: int,
    end_index: int,
) -> list[tuple[str, int, int]]:
    bounds = []
    current = dates[start_index].to_period("M")
    while current <= dates[end_index].to_period("M"):
        month_start = max(
            start_index,
            int(np.flatnonzero(dates >= current.start_time)[0]),
        )
        month_end = min(
            end_index,
            int(np.flatnonzero(dates <= current.end_time)[-1]),
        )
        bounds.append(
            (
                str(current),
                month_start,
                month_end,
            )
        )
        current += 1
    return bounds


def month_metrics(
    data: dict,
    alerts: np.ndarray,
    start_index: int,
    end_index: int,
) -> tuple[float, float]:
    hit_ids = incident_hit_ids(
        data["users"],
        data["dates"],
        alerts,
        data["incidents"],
        start_index,
        end_index,
    )
    start_date = data["dates"][start_index].normalize()
    end_date = data["dates"][end_index].normalize()
    month_incidents = data["incidents"][
        (data["incidents"]["start_date"] >= start_date)
        & (data["incidents"]["start_date"] <= end_date)
    ]
    recall = (
        sum(
            str(incident_id) in hit_ids
            for incident_id in month_incidents["incident_id"]
        )
        / len(month_incidents)
        if len(month_incidents)
        else np.nan
    )
    month_alerts = alerts[:, start_index : end_index + 1]
    month_labels = data["labels"][:, start_index : end_index + 1]
    slots = int(month_alerts.sum())
    true_slots = int(
        (
            month_alerts.astype(bool)
            & month_labels.astype(bool)
        ).sum()
    )
    precision = true_slots / slots if slots else np.nan
    return float(recall), float(precision)


def load_retrained() -> dict:
    data = load_r42()
    scores = np.load(
        RETRAIN / "r4.2_quota_retrained_scores.npz",
        allow_pickle=False,
    )
    data["ts"] = scores["ts2vec_signal"].astype(np.float32)
    data["iforest"] = scores["iforest_signal"].astype(np.float32)
    return data


def build_month_grid(
    data: dict,
    dataset_name: str,
) -> pd.DataFrame:
    bounds = monthly_bounds(
        data["dates"], data["train_end"] + 1, data["val_end"]
    )
    rows = []
    for alpha in ALPHA_GRID:
        for k in K_GRID:
            q = q_from_alpha(k, alpha)
            alerts = build_quota_alerts(
                data["ts"],
                data["iforest"],
                data["active_mask"],
                k,
                q,
            )
            for month, start_index, end_index in bounds:
                recall, precision = month_metrics(
                    data, alerts, start_index, end_index
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "alpha": alpha,
                        "k": k,
                        "q": q,
                        "month": month,
                        "event_recall_3d": recall,
                        "precision_at_k": precision,
                    }
                )
    return pd.DataFrame(rows)


def rolling_selection(frame: pd.DataFrame) -> pd.DataFrame:
    months = sorted(frame["month"].unique())
    history = months[:2]
    rows = []
    for holdout in months[2:]:
        train_rows = frame[frame["month"].isin(history)]
        summary = (
            train_rows.groupby("alpha", as_index=False)
            .agg(
                train_recall=("event_recall_3d", "mean"),
                train_precision=("precision_at_k", "mean"),
            )
            .sort_values(
                ["train_recall", "train_precision", "alpha"],
                ascending=[False, False, True],
            )
        )
        selected_alpha = float(summary.iloc[0]["alpha"])
        for alpha in sorted(frame["alpha"].unique()):
            holdout_rows = frame[
                (frame["month"] == holdout)
                & (frame["alpha"] == alpha)
            ]
            rows.append(
                {
                    "holdout_month": holdout,
                    "selected_alpha": selected_alpha,
                    "alpha": alpha,
                    "mean_event_recall_3d": holdout_rows[
                        "event_recall_3d"
                    ].mean(),
                    "mean_precision_at_k": holdout_rows[
                        "precision_at_k"
                    ].mean(),
                }
            )
        history.append(holdout)
    return pd.DataFrame(rows)


def main() -> None:
    original = build_month_grid(load_r42(), "original")
    retrained = build_month_grid(load_retrained(), "retrained")
    monthly = pd.concat([original, retrained], ignore_index=True)
    monthly.to_csv(
        OUT / "quota_rolling_monthly_grid.csv",
        index=False,
        encoding="utf-8",
    )

    selections = []
    for dataset_name, frame in (
        ("original", original),
        ("retrained", retrained),
    ):
        rolling = rolling_selection(frame)
        rolling["dataset"] = dataset_name
        selections.append(rolling)
    rolling = pd.concat(selections, ignore_index=True)
    rolling.to_csv(
        OUT / "quota_rolling_selection.csv",
        index=False,
        encoding="utf-8",
    )

    summary_rows = []
    for dataset_name in ("original", "retrained"):
        subset = rolling[rolling["dataset"] == dataset_name]
        for alpha in sorted(subset["alpha"].unique()):
            values = subset[subset["alpha"] == alpha]
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "alpha": alpha,
                    "mean_holdout_recall": values[
                        "mean_event_recall_3d"
                    ].mean(),
                    "mean_holdout_precision": values[
                        "mean_precision_at_k"
                    ].mean(),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUT / "quota_rolling_holdout_summary.csv",
        index=False,
        encoding="utf-8",
    )

    summary_json = {
        "design": (
            "Rolling-origin selection within r4.2 validation. "
            "First two months choose alpha; each next month is holdout."
        ),
        "k_grid": K_GRID,
        "alpha_grid": ALPHA_GRID,
        "selection_objective": (
            "mean monthly event Recall@K over prior months; "
            "tie-break mean Precision@K"
        ),
        "selected_alpha_by_fold": {
            dataset_name: rolling[
                rolling["dataset"] == dataset_name
            ][["holdout_month", "selected_alpha"]]
            .drop_duplicates()
            .to_dict(orient="records")
            for dataset_name in ("original", "retrained")
        },
        "holdout_summary": summary.to_dict(orient="records"),
        "outputs": {
            "monthly_grid": str(
                OUT / "quota_rolling_monthly_grid.csv"
            ),
            "selection": str(
                OUT / "quota_rolling_selection.csv"
            ),
            "holdout_summary": str(
                OUT / "quota_rolling_holdout_summary.csv"
            ),
        },
    }
    (OUT / "quota_rolling_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_json, ensure_ascii=False, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
