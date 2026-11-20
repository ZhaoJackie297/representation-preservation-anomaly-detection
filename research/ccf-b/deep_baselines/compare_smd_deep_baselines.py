from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("pr_auc", "roc_auc", "pa_f1_top5")


def load(path: Path, seed: int) -> pd.DataFrame:
    file = path / f"seed_{seed}" / "results_by_machine.csv"
    if not file.exists():
        raise FileNotFoundError(file)
    return pd.read_csv(file).sort_values("machine").reset_index(drop=True)


def cluster_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    *,
    iterations: int,
) -> tuple[float, float]:
    machines = sorted(frame["machine"].unique())
    grouped = {
        machine: frame.loc[frame["machine"] == machine, metric].to_numpy()
        for machine in machines
    }
    metric_offset = sum(ord(character) for character in metric)
    rng = np.random.default_rng(20261201 + metric_offset)
    estimates = []
    for _ in range(iterations):
        sampled = rng.choice(machines, size=len(machines), replace=True)
        estimates.append(
            float(np.concatenate([grouped[machine] for machine in sampled]).mean())
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--core-prefix", default="pai_")
    parser.add_argument("--core-method", required=True)
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(20261201, 20261206)),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deltas = []
    seed_rows = []
    for seed in args.seeds:
        core = load(args.core_root, seed)
        baseline = load(args.baseline_root, seed)
        merged = core[["machine"]].merge(
            baseline[["machine"]],
            on="machine",
            how="inner",
        )
        if len(merged) != len(core) or len(merged) != len(baseline):
            raise ValueError(f"machine mismatch for seed {seed}")
        for metric in METRICS:
            delta = (
                core[f"{args.core_prefix}{metric}"].to_numpy()
                - baseline[metric].to_numpy()
            )
            frame = pd.DataFrame(
                {
                    "seed": seed,
                    "machine": core["machine"],
                    "metric": metric,
                    "delta": delta,
                }
            )
            deltas.append(frame)
            seed_rows.append(
                {
                    "comparison": f"{args.core_method} - {args.baseline_method}",
                    "seed": seed,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "positive_machines": int((delta > 0).sum()),
                    "negative_machines": int((delta < 0).sum()),
                }
            )
    delta_frame = pd.concat(deltas, ignore_index=True)
    summaries = []
    for metric in METRICS:
        selected = delta_frame.loc[delta_frame["metric"] == metric]
        seed_means = (
            selected.groupby("seed")["delta"].mean().sort_index().to_numpy()
        )
        low, high = cluster_bootstrap(
            selected,
            "delta",
            iterations=args.bootstrap_iterations,
        )
        summaries.append(
            {
                "comparison": f"{args.core_method} - {args.baseline_method}",
                "metric": metric,
                "seed_mean_delta": float(seed_means.mean()),
                "positive_seeds": int((seed_means > 0).sum()),
                "negative_seeds": int((seed_means < 0).sum()),
                "machine_cluster_ci95_low": low,
                "machine_cluster_ci95_high": high,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seed_rows).to_csv(
        args.output.with_name(args.output.stem + "_by_seed.csv"),
        index=False,
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
