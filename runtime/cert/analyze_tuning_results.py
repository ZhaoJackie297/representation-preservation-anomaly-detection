from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED42 = ROOT / "processed_r42"
TUNING = PROCESSED42 / "tuning_v1"
PROCESSED52 = ROOT / "processed_r52"
OUTPUT = TUNING / "r4.2_tuned_model_comparison.json"

PRIMARY_K = 10
COOLDOWN_DAYS = 7

sys.path.insert(0, str(ROOT))
import tune_compare_r42 as tuning  # noqa: E402
from analyze_r52_external import (  # noqa: E402
    mcnemar_exact,
    wilson_interval,
)
from production_protocol_r42 import (  # noqa: E402
    build_alerts,
    label_matrix,
)


def event_hit_ids(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    alerts: np.ndarray,
    incidents: pd.DataFrame,
    start_index: int,
    end_index: int,
    horizon: int = 3,
) -> set[str]:
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    start_date = dates[start_index].normalize()
    end_date = dates[end_index].normalize()
    selected = incidents[
        (incidents["start_date"] >= start_date)
        & (incidents["start_date"] <= end_date)
    ]
    hits = set()
    for incident in selected.itertuples(index=False):
        accounts = [
            user_index[account]
            for account in incident.observed_accounts.split("|")
            if account in user_index
        ]
        candidate_dates = pd.date_range(
            incident.start_date - pd.Timedelta(days=COOLDOWN_DAYS - 1),
            min(
                incident.end_date,
                incident.start_date + pd.Timedelta(days=horizon - 1),
            ),
            freq="D",
        )
        for date in candidate_dates:
            date_text = date.strftime("%Y-%m-%d")
            if date_text not in date_index:
                continue
            time_index = date_index[date_text]
            if alerts[np.asarray(accounts), time_index].any():
                hits.add(str(incident.incident_id))
                break
    return hits


def summarize_hit_sets(
    incidents: pd.DataFrame,
    hit_sets: dict[str, set[str]],
) -> dict:
    ts = hit_sets["ts2vec"]
    iforest = hit_sets["isolation_forest"]
    scenario_macro = {}
    for model_name, selected in hit_sets.items():
        recalls = []
        for _, frame in incidents.groupby("scenario"):
            recalls.append(
                frame["incident_id"].isin(selected).mean()
            )
        scenario_macro[model_name] = float(np.mean(recalls))
    event_count = int(incidents["incident_id"].nunique())
    ts_ci = wilson_interval(len(ts), event_count)
    iforest_ci = wilson_interval(len(iforest), event_count)
    return {
        "events": event_count,
        "ts2vec_hits": len(ts),
        "isolation_forest_hits": len(iforest),
        "ts2vec_recall": len(ts) / event_count,
        "isolation_forest_recall": len(iforest) / event_count,
        "ts2vec_wilson_ci95": ts_ci,
        "isolation_forest_wilson_ci95": iforest_ci,
        "shared_hits": len(ts & iforest),
        "ts2vec_only": len(ts - iforest),
        "isolation_forest_only": len(iforest - ts),
        "scenario_macro_recall": scenario_macro,
        "mcnemar": mcnemar_exact(
            len(ts - iforest),
            len(iforest - ts),
        ),
    }


def month_cluster_bootstrap(
    incidents: pd.DataFrame,
    hit_sets: dict[str, set[str]],
    date_column: str,
    iterations: int = 10_000,
    seed: int = 20261110,
) -> dict:
    clusters = {}
    for row in incidents.itertuples(index=False):
        month = str(
            getattr(row, date_column).to_period("M")
        )
        clusters.setdefault(month, []).append(
            row.incident_id
        )
    names = sorted(clusters)
    rng = np.random.default_rng(seed)
    recalls = {"ts2vec": [], "isolation_forest": []}
    for _ in range(iterations):
        sampled_months = rng.choice(
            names, size=len(names), replace=True
        )
        sampled_ids = [
            incident_id
            for month in sampled_months
            for incident_id in clusters[str(month)]
        ]
        for model_name in recalls:
            recalls[model_name].append(
                np.mean(
                    [
                        incident_id in hit_sets[model_name]
                        for incident_id in sampled_ids
                    ]
                )
            )
    ts_values = np.asarray(recalls["ts2vec"])
    iforest_values = np.asarray(
        recalls["isolation_forest"]
    )
    difference = iforest_values - ts_values
    return {
        "iterations": iterations,
        "seed": seed,
        "month_clusters": len(names),
        "isolation_forest_minus_ts2vec_mean": float(
            difference.mean()
        ),
        "isolation_forest_minus_ts2vec_ci95": [
            float(np.quantile(difference, 0.025)),
            float(np.quantile(difference, 0.975)),
        ],
    }


def main() -> None:
    with np.load(tuning.DATASET, allow_pickle=False) as dataset:
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        values = dataset["values"].astype(np.float32)
        active_mask = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
        val_end = int(dataset["val_end_index"])
    iforest_values, _ = tuning.load_iforest_values(users, dates)
    labels = label_matrix(users, dates)
    incidents = pd.read_csv(tuning.INCIDENTS)
    incidents["start_date"] = pd.to_datetime(
        incidents["start"]
    ).dt.normalize()
    incidents["end_date"] = pd.to_datetime(
        incidents["end"]
    ).dt.normalize()
    incidents["first_observed_date"] = pd.to_datetime(
        incidents["first_observed_date"]
    ).dt.normalize()

    selected_specs = {
        "ts2vec": {
            "model": "ts2vec",
            "config": "ts_depth3",
            "variant": "personal_quantile_dynamic",
        },
        "isolation_forest": {
            "model": "isolation_forest",
            "config": "if_100_auto_f100",
            "variant": "personal_quantile_dynamic",
        },
    }
    r42_scores = {}
    for model_name, spec in selected_specs.items():
        r42_scores[model_name] = tuning.ensemble_variants_for_selection(
            spec,
            users,
            dates,
            active_mask,
            train_end,
            values,
            iforest_values,
        )

    r42_hits = {}
    for split_name, start_index, end_index in (
        ("validation", train_end + 1, val_end),
        ("test", val_end + 1, len(dates) - 1),
    ):
        hit_sets = {}
        for model_name, score in r42_scores.items():
            alerts = build_alerts(
                score,
                active_mask,
                PRIMARY_K,
                COOLDOWN_DAYS,
            )
            hit_sets[model_name] = event_hit_ids(
                users,
                dates,
                alerts,
                incidents,
                start_index,
                end_index,
            )
        split_incidents = incidents[
            (incidents["start_date"] >= dates[start_index].normalize())
            & (incidents["start_date"] <= dates[end_index].normalize())
        ]
        r42_hits[split_name] = summarize_hit_sets(
            split_incidents, hit_sets
        )
        r42_hits[split_name][
            "month_cluster_bootstrap"
        ] = month_cluster_bootstrap(
            split_incidents,
            hit_sets,
            date_column="first_observed_date",
        )

    r52 = json.loads(
        (
            PROCESSED52
            / "tuning_v1"
            / "r5.2_tuned_sensitivity_results.json"
        ).read_text(encoding="utf-8")
    )
    r52_hit_sets = {}
    for model_name, source_name in (
        ("ts2vec", "ts2vec_depth3"),
        ("isolation_forest", "iforest_100_auto"),
    ):
        row = next(
            item
            for item in r52
            if item["k"] == PRIMARY_K
            and item["model"] == source_name
            and item["variant"] == "personal_quantile_dynamic"
        )
        r52_hit_sets[model_name] = set(
            row["external_test"]["horizons"]["3"]["incident_ids"]
        )
    r52_incidents = pd.read_csv(
        PROCESSED52 / "r5.2_incidents.csv",
        parse_dates=["first_observed_date"],
    )
    r52_summary = summarize_hit_sets(
        r52_incidents, r52_hit_sets
    )
    r52_summary["month_cluster_bootstrap"] = (
        month_cluster_bootstrap(
            r52_incidents,
            r52_hit_sets,
            date_column="first_observed_date",
        )
    )

    seed_metrics = pd.read_csv(
        TUNING / "r4.2_tuning_seed_metrics.csv"
    )
    calibration_summary = {}
    for model_name, frame in seed_metrics.groupby("model"):
        calibration_summary[model_name] = {
            variant: {
                "recall_3d_mean": float(
                    group["validation_recall_3d"].mean()
                ),
                "recall_3d_std": float(
                    group["validation_recall_3d"].std(ddof=0)
                ),
            }
            for variant, group in frame.groupby("variant")
        }

    output = {
        "selected_configurations": selected_specs,
        "r4.2_validation": r42_hits["validation"],
        "r4.2_development_test": r42_hits["test"],
        "r5.2_post_hoc_external": r52_summary,
        "calibration_and_risk_summary": calibration_summary,
        "note": (
            "r5.2 is post-hoc sensitivity because it has already been "
            "used to inspect model performance. McNemar p-values are "
            "descriptive and do not account for overlapping incidents."
        ),
    }
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
