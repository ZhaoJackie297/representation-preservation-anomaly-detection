from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r52"
DATASET = PROCESSED / "r5.2_sequence_dataset.npz"
LABELS = PROCESSED / "r5.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r5.2_incidents.csv"
SEEDS_FILE = PROCESSED / "r5.2_random_baseline_seeds.csv"
SUMMARY_FILE = PROCESSED / "r5.2_random_baseline_summary.json"

PRIMARY_K = 10
COOLDOWN_DAYS = 7
RANDOM_SEEDS = 200
RANDOM_BASE_SEED = 20261110

sys.path.insert(0, str(ROOT))
from external_eval_r52 import (  # noqa: E402
    build_external_alerts,
    evaluate_external,
)


def main() -> None:
    with np.load(DATASET, allow_pickle=False) as dataset:
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active_mask = dataset["active_mask"][:, :, 0].astype(bool)
        start_index = int(dataset["train_end_index"]) + 1

    labels_frame = pd.read_csv(LABELS, parse_dates=["date"])
    user_index = {
        user: index for index, user in enumerate(users)
    }
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    labels = np.zeros((len(users), len(dates)), dtype=np.uint8)
    for row in labels_frame.itertuples(index=False):
        labels[
            user_index[row.user_id],
            date_index[row.date.strftime("%Y-%m-%d")],
        ] = 1

    incidents = pd.read_csv(
        INCIDENTS,
        parse_dates=[
            "first_observed_date",
            "last_observed_date",
        ],
    )
    rng = np.random.default_rng(RANDOM_BASE_SEED)
    rows = []
    for seed in range(RANDOM_SEEDS):
        scores = rng.random((len(users), len(dates)), dtype=np.float32)
        scores[~active_mask] = -np.inf
        alerts = build_external_alerts(
            scores,
            active_mask,
            start_index,
            PRIMARY_K,
            COOLDOWN_DAYS,
        )
        result = evaluate_external(
            users,
            dates,
            active_mask,
            alerts,
            labels,
            incidents,
            start_index,
            "random",
            PRIMARY_K,
        )["external_test"]
        rows.append(
            {
                "seed": seed,
                "incidents_hit_1d": result["horizons"]["1"][
                    "incidents_hit"
                ],
                "recall_1d": result["horizons"]["1"]["event_recall"],
                "incidents_hit_3d": result["horizons"]["3"][
                    "incidents_hit"
                ],
                "recall_3d": result["horizons"]["3"]["event_recall"],
                "true_alert_slots": result["true_alert_slots"],
                "precision_at_k": result["precision_at_k"],
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(SEEDS_FILE, index=False, encoding="utf-8")

    def distribution(column: str) -> dict:
        values = frame[column].to_numpy(dtype=float)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "quantile_2_5": float(np.quantile(values, 0.025)),
            "median": float(np.median(values)),
            "quantile_97_5": float(np.quantile(values, 0.975)),
            "max": float(np.max(values)),
        }

    summary = {
        "protocol": {
            "k": PRIMARY_K,
            "cooldown_days": COOLDOWN_DAYS,
            "random_seeds": RANDOM_SEEDS,
            "base_seed": RANDOM_BASE_SEED,
            "start_date": str(dates[start_index].date()),
            "end_date": str(dates[-1].date()),
            "event_count": int(len(incidents)),
        },
        "incidents_hit_1d": distribution("incidents_hit_1d"),
        "incidents_hit_3d": distribution("incidents_hit_3d"),
        "recall_1d": distribution("recall_1d"),
        "recall_3d": distribution("recall_3d"),
        "true_alert_slots": distribution("true_alert_slots"),
        "precision_at_k": distribution("precision_at_k"),
        "seed_file": str(SEEDS_FILE),
        "purpose": (
            "No-skill reference under the same finite review budget; "
            "not used for model selection."
        ),
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
