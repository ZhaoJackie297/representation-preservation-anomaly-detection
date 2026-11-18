from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def monthly_hits(
    hits_path: Path,
    incidents_path: Path,
    dataset_path: Path,
    method_column: str,
    split: str | None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    horizon_column: str | None = None,
) -> pd.DataFrame:
    hits = pd.read_csv(hits_path)
    hits = hits[hits["k"] == 10]
    if split is not None:
        hits = hits[hits["split"] == split]
    if horizon_column is not None:
        hits = hits[hits[horizon_column] == 3]

    incidents = pd.read_csv(incidents_path)
    date_column = (
        "first_observed_date"
        if "first_observed_date" in incidents.columns
        else "start"
    )
    incidents[date_column] = pd.to_datetime(incidents[date_column])
    incidents["incident_id"] = incidents["incident_id"].astype(str)
    if start_date is not None:
        incidents = incidents[
            incidents[date_column] >= start_date
        ]
    if end_date is not None:
        incidents = incidents[
            incidents[date_column] <= end_date
        ]
    incidents["month"] = (
        incidents[date_column].dt.to_period("M").astype(str)
    )

    rows = []
    for method, method_hits in hits.groupby(method_column):
        hit_ids = set(method_hits["incident_id"].astype(str))
        for month, group in incidents.groupby("month", sort=True):
            month_ids = set(group["incident_id"])
            total = len(month_ids)
            hits_count = len(month_ids & hit_ids)
            rows.append(
                {
                    "split": split or "external_test",
                    "month": month,
                    "method": method,
                    "incident_hits": hits_count,
                    "incident_total": total,
                    "event_recall_3d": (
                        hits_count / total if total else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    r42_dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    r42_dates = pd.to_datetime(r42_dataset["dates"].astype(str))
    train_end = int(r42_dataset["train_end_index"])
    val_end = int(r42_dataset["val_end_index"])

    r42 = monthly_hits(
        ROOT
        / "processed_r42"
        / "r4.2_conditional_combination_hits.csv",
        ROOT / "processed_r42" / "r4.2_incidents.csv",
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        "variant",
        "validation",
        r42_dates[train_end + 1].normalize(),
        r42_dates[val_end].normalize(),
    )
    r42_test = monthly_hits(
        ROOT
        / "processed_r42"
        / "r4.2_conditional_combination_hits.csv",
        ROOT / "processed_r42" / "r4.2_incidents.csv",
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        "variant",
        "test",
        r42_dates[val_end + 1].normalize(),
        r42_dates[-1].normalize(),
    )
    r42 = pd.concat([r42, r42_test], ignore_index=True)
    r42.to_csv(
        ROOT
        / "processed_r42"
        / "r4.2_conditional_combination_monthly.csv",
        index=False,
        encoding="utf-8",
    )

    r52_dataset = np.load(
        ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    r52_dates = pd.to_datetime(r52_dataset["dates"].astype(str))
    r52_train_end = int(r52_dataset["train_end_index"])
    r52 = monthly_hits(
        ROOT
        / "processed_r52"
        / "r5.2_conditional_combination_hits.csv",
        ROOT / "processed_r52" / "r5.2_incidents.csv",
        ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
        "method",
        None,
        r52_dates[r52_train_end + 1].normalize(),
        r52_dates[-1].normalize(),
        "horizon_days",
    )
    r52.to_csv(
        ROOT
        / "processed_r52"
        / "r5.2_conditional_combination_monthly.csv",
        index=False,
        encoding="utf-8",
    )

    methods = [
        "ts2vec_single",
        "iforest_single",
        "iforest_primary_ts2vec_rescue_q1_nogate",
        "ts2vec_primary_iforest_rescue_q1_nogate",
    ]
    for label, frame in (("r4.2", r42), ("r5.2", r52)):
        selected = frame[frame["method"].isin(methods)]
        print(f"{label} monthly event recall")
        print(
            selected.pivot(
                index=["split", "month"],
                columns="method",
                values="event_recall_3d",
            )
            .fillna(0.0)
            .to_string(float_format=lambda value: f"{value:.3f}")
        )


if __name__ == "__main__":
    main()
