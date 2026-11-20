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
    "pa_f1_top5",
    "f1_q99",
    "segment_recall_q99",
)
PAIRS = (
    ("pai_t4_vs_t0", "pai", "pai"),
    ("pai_official_t4_vs_t0", "pai_official", "pai_official"),
    ("euc_t4_vs_t0", "euc", "euc"),
    ("mask_t4_vs_t0", "original", "original"),
)


def load_variant(root: Path, seed: int) -> pd.DataFrame:
    path = root / f"seed_{seed}" / "results_by_machine.csv"
    frame = pd.read_csv(path)
    if "machine" not in frame.columns or "seed" not in frame.columns:
        raise ValueError(f"invalid machine result table: {path}")
    if frame["seed"].nunique() != 1:
        raise ValueError(f"multiple seeds in {path}")
    return frame.sort_values("machine").reset_index(drop=True)


def cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    seed_column: int,
    iterations: int,
) -> tuple[float, float]:
    machine_values = sorted(frame["machine"].unique())
    arrays = {
        machine: frame.loc[frame["machine"] == machine, "delta"].to_numpy()
        for machine in machine_values
    }
    rng = np.random.default_rng(seed_column)
    estimates = []
    for _ in range(iterations):
        sampled = rng.choice(machine_values, size=len(machine_values), replace=True)
        values = np.concatenate([arrays[machine] for machine in sampled])
        estimates.append(float(values.mean()))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_pair(
    t0_root: Path,
    t4_root: Path,
    seeds: list[int],
    *,
    name: str,
    variant: str,
    bootstrap_iterations: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    seed_rows: list[dict[str, object]] = []
    delta_frames: list[pd.DataFrame] = []

    for seed in seeds:
        t0 = load_variant(t0_root, seed)
        t4 = load_variant(t4_root, seed)
        merged = t0[["machine", "seed"]].copy()
        for metric in METRICS:
            merged[f"{metric}_t0"] = t0[f"{variant}_{metric}"].to_numpy()
            merged[f"{metric}_t4"] = t4[f"{variant}_{metric}"].to_numpy()
            merged[f"{metric}_delta"] = (
                merged[f"{metric}_t4"] - merged[f"{metric}_t0"]
            )
            seed_rows.append(
                {
                    "comparison": name,
                    "seed": seed,
                    "metric": metric,
                    "t0_mean": float(merged[f"{metric}_t0"].mean()),
                    "t4_mean": float(merged[f"{metric}_t4"].mean()),
                    "delta": float(merged[f"{metric}_delta"].mean()),
                    "positive_machines": int(
                        (merged[f"{metric}_delta"] > 0).sum()
                    ),
                    "negative_machines": int(
                        (merged[f"{metric}_delta"] < 0).sum()
                    ),
                }
            )
        for metric in METRICS:
            delta_frames.append(
                pd.DataFrame(
                    {
                        "comparison": name,
                        "seed": seed,
                        "machine": merged["machine"],
                        "metric": metric,
                        "delta": merged[f"{metric}_delta"],
                    }
                )
            )

    all_deltas = pd.concat(delta_frames, ignore_index=True)
    bootstrap_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    seed_frame = pd.DataFrame(seed_rows)

    for metric in METRICS:
        metric_seed = seed_frame.loc[seed_frame["metric"] == metric].sort_values(
            "seed"
        )
        values = metric_seed["delta"].to_numpy(dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if len(values) > 1:
            half = float(
                student_t.ppf(0.975, len(values) - 1)
                * std
                / np.sqrt(len(values))
            )
        else:
            half = 0.0
        metric_deltas = all_deltas.loc[
            all_deltas["metric"] == metric,
            ["machine", "delta"],
        ]
        low, high = cluster_bootstrap(
            metric_deltas,
            seed_column=20261201 + METRICS.index(metric),
            iterations=bootstrap_iterations,
        )
        bootstrap_rows.append(
            {
                "comparison": name,
                "metric": metric,
                "seed_mean_delta": mean,
                "seed_std_delta": std,
                "seed_t_ci95_low": mean - half,
                "seed_t_ci95_high": mean + half,
                "positive_seeds": int((values > 0).sum()),
                "zero_seeds": int((values == 0).sum()),
                "negative_seeds": int((values < 0).sum()),
                "machine_cluster_ci95_low": low,
                "machine_cluster_ci95_high": high,
            }
        )
        summary[metric] = {
            "seed_mean_delta": mean,
            "seed_std_delta": std,
            "seed_t_ci95": [mean - half, mean + half],
            "machine_cluster_ci95": [low, high],
            "positive_seeds": int((values > 0).sum()),
            "negative_seeds": int((values < 0).sum()),
        }

    return seed_rows, bootstrap_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-root", type=Path, required=True)
    parser.add_argument("--t4-root", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(20261201, 20261206)),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "seeds": args.seeds,
        "bootstrap_iterations": args.bootstrap_iterations,
        "comparisons": {},
    }
    for name, variant, _ in PAIRS:
        pair_seed, pair_bootstrap, pair_summary = summarize_pair(
            args.t0_root,
            args.t4_root,
            args.seeds,
            name=name,
            variant=variant,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        seed_rows.extend(pair_seed)
        bootstrap_rows.extend(pair_bootstrap)
        summary["comparisons"][name] = pair_summary

    pd.DataFrame(seed_rows).to_csv(
        args.output_dir / "smd_multiseed_by_seed.csv",
        index=False,
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output_dir / "smd_multiseed_bootstrap.csv",
        index=False,
    )
    (args.output_dir / "smd_multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
