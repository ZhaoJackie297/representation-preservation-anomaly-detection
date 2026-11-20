from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("pr_auc", "roc_auc", "pa_f1_top5")


def mean_row(
    frame: pd.DataFrame,
    *,
    method: str,
    seed: int,
    prefix: str = "",
) -> dict[str, object]:
    row = {"method": method, "seed": seed, "machines": len(frame)}
    for metric in METRICS:
        row[metric] = float(frame[f"{prefix}{metric}"].mean())
    return row


def load_seed(path: Path, *, method: str, seed: int, prefix: str = ""):
    file = path / f"seed_{seed}" / "results_by_machine.csv"
    if not file.exists():
        return None
    return mean_row(
        pd.read_csv(file),
        method=method,
        seed=seed,
        prefix=prefix,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-root", type=Path)
    parser.add_argument("--t4-root", type=Path)
    parser.add_argument("--anomaly-transformer-root", type=Path)
    parser.add_argument("--omni-root", type=Path)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(20261201, 20261206)),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for seed in args.seeds:
        if args.t0_root:
            for method, prefix in (
                ("T0-matched + project PAI", "pai_"),
                ("T0-matched + official PAI", "pai_official_"),
            ):
                row = load_seed(
                    args.t0_root,
                    method=method,
                    seed=seed,
                    prefix=prefix,
                )
                if row:
                    rows.append(row)
        if args.t4_root:
            for method, prefix in (
                ("T4-full-soft + project PAI", "pai_"),
                ("T4-full-soft + official PAI", "pai_official_"),
            ):
                row = load_seed(
                    args.t4_root,
                    method=method,
                    seed=seed,
                    prefix=prefix,
                )
                if row:
                    rows.append(row)
        if args.anomaly_transformer_root:
            row = load_seed(
                args.anomaly_transformer_root,
                method="Anomaly Transformer-adapted",
                seed=seed,
            )
            if row:
                rows.append(row)
        if args.omni_root:
            row = load_seed(
                args.omni_root,
                method="OmniAnomaly-adapted",
                seed=seed,
            )
            if row:
                rows.append(row)
    if not rows:
        raise FileNotFoundError("no SMD baseline rows found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_seed = pd.DataFrame(rows).sort_values(["method", "seed"])
    by_seed.to_csv(args.output_dir / "smd_baselines_by_seed.csv", index=False)
    summary = (
        by_seed.groupby("method")
        .agg(
            seeds=("seed", "count"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pa_f1_top5_mean=("pa_f1_top5", "mean"),
            pa_f1_top5_std=("pa_f1_top5", "std"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "smd_baselines_summary.csv", index=False)
    (args.output_dir / "smd_baselines_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
