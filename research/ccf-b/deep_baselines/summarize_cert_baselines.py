from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_summary(path: Path, method: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "method": method,
        "seed": int(payload["seed"]),
        "pr_auc": float(payload["point"]["pr_auc"]),
        "roc_auc": float(payload["point"]["roc_auc"]),
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
    }
    for k in (5, 10, 20):
        evaluated = payload.get("event", {}).get(str(k))
        row[f"event_recall_k{k}"] = (
            float(
                evaluated["splits"]["test"]["horizons"]["3"]["event_recall"]
            )
            if evaluated
            else None
        )
        row[f"event_hits_k{k}"] = (
            int(evaluated["splits"]["test"]["horizons"]["3"]["incidents_hit"])
            if evaluated
            else None
        )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tnnclr-root", type=Path)
    parser.add_argument("--anomaly-transformer-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    roots = (
        (args.tnnclr_root, "t-NNCLR"),
        (args.anomaly_transformer_root, "Anomaly Transformer"),
    )
    for root, method in roots:
        if root is None:
            continue
        for path in sorted(root.glob("seed_*/summary.json")):
            row = load_summary(path, method)
            if row is not None:
                rows.append(row)
    if not rows:
        raise FileNotFoundError("no CERT baseline seed summaries found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["method", "seed"])
    frame.to_csv(args.output_dir / "cert_deep_baselines_by_seed.csv", index=False)
    summary = (
        frame.groupby("method", dropna=False)
        .agg(
            seeds=("seed", "count"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_k10_mean=("event_recall_k10", "mean"),
            event_recall_k10_std=("event_recall_k10", "std"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "cert_deep_baselines_summary.csv", index=False)
    (args.output_dir / "cert_deep_baselines_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
