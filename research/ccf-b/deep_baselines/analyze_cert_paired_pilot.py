from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


METRICS = (
    "pr_auc",
    "roc_auc",
    "event_recall_k5",
    "event_recall_k10",
    "event_recall_k20",
)


def load_pipeline_metrics(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"seed", "method", "split", "k", "pr_auc", "roc_auc"}
    if not required.issubset(frame.columns):
        raise ValueError(f"invalid pipeline metric table: {path}")
    selected = frame.loc[
        (frame["method"] == method) & (frame["split"] == "test")
    ].copy()
    if selected.empty:
        raise ValueError(f"no test rows for {method} in {path}")

    rows = []
    for seed, seed_frame in selected.groupby("seed"):
        point = seed_frame[["pr_auc", "roc_auc"]].drop_duplicates()
        if len(point) != 1:
            raise ValueError(f"point metrics vary by K for seed {seed}")
        row: dict[str, object] = {
            "seed": int(seed),
            "pr_auc": float(point.iloc[0]["pr_auc"]),
            "roc_auc": float(point.iloc[0]["roc_auc"]),
        }
        for k in (5, 10, 20):
            matches = seed_frame.loc[seed_frame["k"] == k]
            if len(matches) != 1:
                raise ValueError(f"expected one K={k} row for seed {seed}")
            row[f"event_recall_k{k}"] = float(matches.iloc[0]["event_recall_3d"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def load_baseline_metrics(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "method",
        "seed",
        "pr_auc",
        "roc_auc",
        "event_recall_k5",
        "event_recall_k10",
        "event_recall_k20",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"invalid baseline metric table: {path}")
    selected = frame.loc[frame["method"] == method].copy()
    if selected.empty:
        raise ValueError(f"no rows for {method} in {path}")
    if selected["seed"].duplicated().any():
        raise ValueError(f"duplicate seeds for {method} in {path}")
    columns = ["seed", *METRICS]
    return selected[columns].sort_values("seed").reset_index(drop=True)


def paired_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    comparison: str,
) -> pd.DataFrame:
    merged = left.merge(right, on="seed", suffixes=("_left", "_right"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f"seed mismatch for {comparison}")
    rows = []
    for metric in METRICS:
        for row in merged.itertuples(index=False):
            left_value = float(getattr(row, f"{metric}_left"))
            right_value = float(getattr(row, f"{metric}_right"))
            rows.append(
                {
                    "comparison": comparison,
                    "seed": int(row.seed),
                    "metric": metric,
                    "left": left_value,
                    "right": right_value,
                    "delta": left_value - right_value,
                }
            )
    return pd.DataFrame(rows)


def summarize(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comparison, metric), selected in by_seed.groupby(
        ["comparison", "metric"]
    ):
        values = selected.sort_values("seed")["delta"].to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        half = float(
            student_t.ppf(0.975, len(values) - 1)
            * std
            / np.sqrt(len(values))
        )
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "seeds": len(values),
                "left_mean": float(selected["left"].mean()),
                "right_mean": float(selected["right"].mean()),
                "mean_delta": mean,
                "std_delta": std,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "positive_seeds": int((values > 0).sum()),
                "negative_seeds": int((values < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair CERT T0/T4 pilot metrics with Anomaly Transformer and "
            "t-NNCLR on the same seeds."
        )
    )
    parser.add_argument("--t0-metrics", type=Path, required=True)
    parser.add_argument("--t4-metrics", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--t0-method",
        default="T0_global_z",
    )
    parser.add_argument(
        "--t4-method",
        default="T4_full_soft_global_z",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = load_pipeline_metrics(args.t0_metrics, args.t0_method)
    t4 = load_pipeline_metrics(args.t4_metrics, args.t4_method)
    anomaly = load_baseline_metrics(
        args.baseline_metrics,
        "Anomaly Transformer",
    )
    tnnclr = load_baseline_metrics(
        args.baseline_metrics,
        "t-NNCLR",
    )
    seeds = sorted(set(t0["seed"]) & set(t4["seed"]))
    expected = sorted(range(20261201, 20261204))
    if seeds != expected:
        raise ValueError(f"expected pilot seeds {expected}, found {seeds}")

    comparisons = (
        ("T4_full_soft_minus_T0", t4, t0),
        ("T4_full_soft_minus_Anomaly_Transformer", t4, anomaly),
        ("T4_full_soft_minus_t_NNCLR", t4, tnnclr),
        ("T0_matched_minus_Anomaly_Transformer", t0, anomaly),
        ("T0_matched_minus_t_NNCLR", t0, tnnclr),
    )
    by_seed = pd.concat(
        [
            paired_rows(left, right, comparison=name)
            for name, left, right in comparisons
        ],
        ignore_index=True,
    )
    summary = summarize(by_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(
        args.output_dir / "cert_paired_pilot_by_seed.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "cert_paired_pilot_summary.csv",
        index=False,
    )
    payload = {
        "status": "exploratory_adaptation_pilot",
        "seeds": seeds,
        "t0_method": args.t0_method,
        "t4_method": args.t4_method,
        "comparisons": summary.to_dict(orient="records"),
    }
    (args.output_dir / "cert_paired_pilot_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
