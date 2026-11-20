from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


METHODS = ("A0-official", "A0-matched", "A1-Cov")


def label_matrix(
    path: Path,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    frame = pd.read_csv(path, parse_dates=["date"])
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    matrix = np.zeros((len(users), len(dates)), dtype=np.uint8)
    for row in frame.itertuples(index=False):
        user = user_index.get(str(row.user_id))
        day = date_index.get(pd.Timestamp(row.date).strftime("%Y-%m-%d"))
        if user is not None and day is not None:
            matrix[user, day] = 1
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(20261116, 20261126)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.runtime_dir.resolve()))
    from e6_r62_audit import point_metrics
    from production_protocol_r42 import build_alerts, evaluate

    dataset_path = args.processed_dir / "r4.2_sequence_dataset.npz"
    labels_path = args.processed_dir / "r4.2_malicious_account_days.csv"
    incidents_path = args.processed_dir / "r4.2_incidents.csv"
    dataset = np.load(dataset_path, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    test_start = int(dataset["test_start_index"])
    labels = label_matrix(labels_path, users, dates)
    incidents = pd.read_csv(incidents_path)

    rows = []
    for seed in args.seeds:
        for method in METHODS:
            score_path = args.score_root / f"seed_{seed}" / f"{method}_scores.npz"
            with np.load(score_path, allow_pickle=False) as scores:
                euc = scores["euc_z"].astype(np.float32)
                mag = scores["mag_g_z"].astype(np.float32)
                t2 = scores["t2_z"].astype(np.float32)
                project = scores["pai_a1"].astype(np.float32)
            official = 0.6 * euc + 0.4 * mag + 0.2 * t2
            for variant, signal in (
                ("project_pai_a1", project),
                ("official_scalar_weights", official),
            ):
                point = point_metrics(signal, labels, active, test_start)
                output = {
                    "seed": seed,
                    "method": method,
                    "pai_variant": variant,
                    "test_pr_auc": point["pr_auc"],
                    "test_roc_auc": point["roc_auc"],
                }
                data = {
                    "users": users,
                    "dates": dates,
                    "active_mask": active,
                    "labels": labels,
                    "incidents": incidents,
                    "test_start": test_start,
                }
                for k in (5, 10, 20):
                    alerts = build_alerts(signal, active, k, 7)
                    evaluated = evaluate(
                        users,
                        dates,
                        active,
                        alerts,
                        labels,
                        incidents,
                        train_end,
                        val_end,
                        variant,
                        k,
                    )
                    test = evaluated["splits"]["test"]
                    output[f"event_recall_k{k}"] = test["horizons"]["3"][
                        "event_recall"
                    ]
                    output[f"event_hits_k{k}"] = test["horizons"]["3"][
                        "incidents_hit"
                    ]
                rows.append(output)

    by_seed = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(
        args.output_dir / "official_pai_10seed_by_seed.csv",
        index=False,
    )
    metric_columns = (
        "test_pr_auc",
        "test_roc_auc",
        "event_recall_k5",
        "event_recall_k10",
        "event_recall_k20",
    )
    aggregate_rows = []
    paired_rows = []
    for method in METHODS:
        method_frame = by_seed.loc[by_seed["method"] == method]
        for variant in ("project_pai_a1", "official_scalar_weights"):
            selected = method_frame.loc[method_frame["pai_variant"] == variant]
            row = {"method": method, "pai_variant": variant}
            for metric in metric_columns:
                row[f"{metric}_mean"] = float(selected[metric].mean())
                row[f"{metric}_std"] = float(selected[metric].std(ddof=1))
            aggregate_rows.append(row)
        pivot = method_frame.pivot(
            index="seed",
            columns="pai_variant",
            values=list(metric_columns),
        )
        for metric in metric_columns:
            delta = (
                pivot[(metric, "project_pai_a1")]
                - pivot[(metric, "official_scalar_weights")]
            )
            paired_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean_project_minus_official": float(delta.mean()),
                    "positive_seeds": int((delta > 0).sum()),
                    "negative_seeds": int((delta < 0).sum()),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    paired = pd.DataFrame(paired_rows)
    aggregate.to_csv(
        args.output_dir / "official_pai_10seed_summary.csv",
        index=False,
    )
    paired.to_csv(
        args.output_dir / "official_pai_10seed_paired.csv",
        index=False,
    )
    payload = {
        "seeds": args.seeds,
        "project_weights": [1.0, 0.6, 0.4],
        "official_weights": [0.6, 0.4, 0.2],
        "summary": aggregate.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
    }
    (args.output_dir / "official_pai_10seed_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
