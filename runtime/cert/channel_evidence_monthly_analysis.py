from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
PRIMARY_K = 10
Q_LIMIT = 4
COOLDOWN_DAYS = 7
CHANNELS = ["logon", "device", "file", "email", "http"]

sys.path.insert(0, str(ROOT))
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from production_protocol_r42 import (  # noqa: E402
    build_alerts,
    label_matrix,
)
from quota_relationship_sim import (  # noqa: E402
    evaluate_alert_system,
    load_r42,
    load_r52,
)


def event_start_series(incidents: pd.DataFrame) -> pd.Series:
    if "start_date" in incidents.columns:
        return pd.to_datetime(incidents["start_date"]).dt.normalize()
    return pd.to_datetime(
        incidents["first_observed_date"]
    ).dt.normalize()


def extract_hit_ids(
    result: dict,
    split: str,
    external: bool,
) -> set[str]:
    if external:
        return set(
            result["external_test"]["horizons"]["3"]["incident_ids"]
        )
    return set(result["splits"][split]["incident_ids"])


def monthly_metrics(
    data_block: str,
    method: str,
    data: dict,
    alerts: np.ndarray,
    labels: np.ndarray,
    *,
    external: bool,
    splits: list[str],
) -> pd.DataFrame:
    result = evaluate_alert_system(
        data, alerts, method, PRIMARY_K, external=external
    )
    incidents = data["incidents"].copy()
    incidents["event_start"] = event_start_series(incidents)
    months = pd.period_range(
        data["dates"][0], data["dates"][-1], freq="M"
    )
    rows = []
    for split in splits:
        hit_ids = extract_hit_ids(result, split, external)
        if external:
            split_start = data["dates"][data["test_start"]].normalize()
            split_end = data["dates"][-1].normalize()
        elif split == "validation":
            split_start = data["dates"][
                data["train_end"] + 1
            ].normalize()
            split_end = data["dates"][data["val_end"]].normalize()
        else:
            split_start = data["dates"][
                data["val_end"] + 1
            ].normalize()
            split_end = data["dates"][-1].normalize()

        for month in months:
            month_start = max(
                month.start_time.normalize(), split_start
            )
            month_end = min(month.end_time.normalize(), split_end)
            if month_start > month_end:
                continue
            month_incidents = incidents[
                (incidents["event_start"] >= month_start)
                & (incidents["event_start"] <= month_end)
            ]
            if month_incidents.empty:
                continue
            ids = set(month_incidents["incident_id"].astype(str))
            hits = ids & hit_ids
            recall = len(hits) / len(ids)
            scenario_recalls = []
            for _, group in month_incidents.groupby("scenario"):
                group_ids = set(group["incident_id"].astype(str))
                scenario_recalls.append(
                    len(group_ids & hits) / len(group_ids)
                )
            scenario_macro = float(np.mean(scenario_recalls))
            date_mask = (
                (data["dates"] >= month_start)
                & (data["dates"] <= month_end)
            )
            month_alerts = alerts[:, date_mask]
            month_labels = labels[:, date_mask]
            alert_slots = int(month_alerts.sum())
            true_slots = int((month_alerts & month_labels).sum())
            precision = (
                true_slots / alert_slots if alert_slots else 0.0
            )
            utility = (
                0.5 * recall
                + 0.25 * scenario_macro
                + 0.25 * precision
            )
            rows.append(
                {
                    "data_block": data_block,
                    "method": method,
                    "split": split,
                    "month": str(month),
                    "incident_hits": len(hits),
                    "incident_total": len(ids),
                    "event_recall_3d": recall,
                    "scenario_macro_recall_3d": scenario_macro,
                    "precision_at_k": precision,
                    "balanced_utility": utility,
                }
            )
    return pd.DataFrame(rows)


def dominant_channel_rows(
    arrays,
    data_block: str,
    alerts: np.ndarray,
    active_mask: np.ndarray,
    dates: pd.DatetimeIndex,
    labels: np.ndarray,
    evaluation_mask: np.ndarray,
) -> list[dict[str, object]]:
    quantiles = np.stack(
        [
            arrays[f"{data_block}_channel_{channel}_quantile"]
            for channel in CHANNELS
        ],
        axis=0,
    )
    rows = []
    for time_index in np.flatnonzero(evaluation_mask):
        for user_index in np.flatnonzero(alerts[:, time_index]):
            values = quantiles[:, user_index, time_index]
            order = np.argsort(values)[::-1]
            rows.append(
                {
                    "data_block": data_block,
                    "date": dates[time_index].strftime("%Y-%m-%d"),
                    "user_id": user_index,
                    "dominant_channel": CHANNELS[int(order[0])],
                    "second_channel": CHANNELS[int(order[1])],
                    "top1_quantile": float(values[order[0]]),
                    "top2_quantile": float(values[order[1]]),
                    "audit_malicious_account_day": int(
                        labels[user_index, time_index]
                    ),
                }
            )
    return rows


def main() -> None:
    arrays = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    dual_arrays = np.load(
        OUT / "ts2vec_dual_score_arrays.npz",
        allow_pickle=False,
    )
    r42 = load_r42()
    r52 = load_r52()
    r42_labels = label_matrix(r42["users"], r42["dates"])
    blocks = [
        {
            "name": "r42_original",
            "data": r42,
            "labels": r42_labels,
            "external": False,
            "splits": ["validation", "test"],
            "evaluation_mask": np.zeros(len(r42["dates"]), dtype=bool),
        },
        {
            "name": "r42_retrained",
            "data": r42,
            "labels": r42_labels,
            "external": False,
            "splits": ["validation", "test"],
            "evaluation_mask": np.zeros(len(r42["dates"]), dtype=bool),
        },
        {
            "name": "r52_external",
            "data": r52,
            "labels": r52["labels"],
            "external": True,
            "splits": ["external_test"],
            "evaluation_mask": np.zeros(len(r52["dates"]), dtype=bool),
        },
    ]
    blocks[0]["evaluation_mask"][r42["train_end"] + 1 :] = True
    blocks[1]["evaluation_mask"][r42["train_end"] + 1 :] = True
    blocks[2]["evaluation_mask"][r52["test_start"] :] = True

    monthly_frames = []
    channel_rows = []
    for block in blocks:
        name = str(block["name"])
        data = block["data"]
        auxiliary = dual_arrays[f"{name}_dual_view_iforest"]
        fixed_signal = arrays[f"{name}_fixed_dual_iforest_signal"]
        triple_signal = arrays[f"{name}_triple_mean_iforest_signal"]
        fixed_alerts = build_unique_quota_alerts(
            fixed_signal,
            auxiliary,
            data["active_mask"],
            PRIMARY_K,
            Q_LIMIT,
        )
        triple_alerts = build_unique_quota_alerts(
            triple_signal,
            auxiliary,
            data["active_mask"],
            PRIMARY_K,
            Q_LIMIT,
        )
        monthly_frames.append(
            monthly_metrics(
                name,
                "fixed_dual_iforest",
                data,
                fixed_alerts,
                block["labels"],
                external=bool(block["external"]),
                splits=list(block["splits"]),
            )
        )
        monthly_frames.append(
            monthly_metrics(
                name,
                "triple_mean_iforest",
                data,
                triple_alerts,
                block["labels"],
                external=bool(block["external"]),
                splits=list(block["splits"]),
            )
        )
        channel_rows.extend(
            dominant_channel_rows(
                arrays,
                name,
                triple_alerts,
                data["active_mask"],
                data["dates"],
                block["labels"],
                block["evaluation_mask"],
            )
        )

    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly.to_csv(
        OUT / "channel_evidence_monthly_stability.csv",
        index=False,
        encoding="utf-8",
    )
    pivot = monthly.pivot_table(
        index=["data_block", "split", "month"],
        columns="method",
        values="balanced_utility",
    ).reset_index()
    pivot["triple_minus_fixed"] = (
        pivot["triple_mean_iforest"]
        - pivot["fixed_dual_iforest"]
    )
    pivot.to_csv(
        OUT / "channel_evidence_monthly_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    channel_frame = pd.DataFrame(channel_rows)
    channel_summary = (
        channel_frame.groupby(
            ["data_block", "dominant_channel"],
            as_index=False,
        )
        .agg(
            selected_alerts=("user_id", "size"),
            malicious_account_days=(
                "audit_malicious_account_day",
                "sum",
            ),
            mean_top1_quantile=("top1_quantile", "mean"),
            mean_top2_quantile=("top2_quantile", "mean"),
        )
    )
    channel_summary["audit_density"] = (
        channel_summary["malicious_account_days"]
        / channel_summary["selected_alerts"]
    )
    channel_summary.to_csv(
        OUT / "channel_evidence_dominant_channel_summary.csv",
        index=False,
        encoding="utf-8",
    )

    summary = {
        "months": int(len(pivot)),
        "triple_better_months": int(
            (pivot["triple_minus_fixed"] > 1e-12).sum()
        ),
        "triple_worse_months": int(
            (pivot["triple_minus_fixed"] < -1e-12).sum()
        ),
        "triple_tied_months": int(
            (pivot["triple_minus_fixed"].abs() <= 1e-12).sum()
        ),
        "mean_monthly_utility_difference": float(
            pivot["triple_minus_fixed"].mean()
        ),
        "min_monthly_utility_difference": float(
            pivot["triple_minus_fixed"].min()
        ),
        "max_monthly_utility_difference": float(
            pivot["triple_minus_fixed"].max()
        ),
        "outputs": {
            "monthly_stability": str(
                OUT / "channel_evidence_monthly_stability.csv"
            ),
            "monthly_comparison": str(
                OUT / "channel_evidence_monthly_comparison.csv"
            ),
            "dominant_channels": str(
                OUT / "channel_evidence_dominant_channel_summary.csv"
            ),
        },
    }
    (OUT / "channel_evidence_monthly_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nDominant channels", flush=True)
    print(channel_summary.to_string(index=False), flush=True)
    print("\nMonthly comparison", flush=True)
    print(pivot.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
