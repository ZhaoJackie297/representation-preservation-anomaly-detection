from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "experiments" / "representation_preservation"
OUT = BASE / "e10"
OUT.mkdir(parents=True, exist_ok=True)

R62_T4 = BASE / "e4_joint" / "r62" / "metrics_by_seed.csv"
R62_T0 = BASE / "e4_joint" / "r62_matched_t0" / "metrics_by_seed.csv"
SMD_T4 = BASE / "e8" / "smd" / "seed_20261116" / "results_by_machine.csv"
SMD_T0 = (
    BASE
    / "e8"
    / "smd_matched_t0"
    / "seed_20261116"
    / "results_by_machine.csv"
)


def paired_test(delta: pd.Series) -> dict:
    values = delta.to_numpy(dtype=float)
    return {
        "mean_difference": float(np.mean(values)),
        "median_difference": float(np.median(values)),
        "positive": int((values > 0).sum()),
        "negative": int((values < 0).sum()),
        "zero": int((values == 0).sum()),
        "wilcoxon_p": (
            float(wilcoxon(values).pvalue)
            if len(values) >= 2 and np.any(values != 0)
            else None
        ),
    }


def compare_r62() -> dict:
    if not R62_T4.exists() or not R62_T0.exists():
        return {"status": "pending"}
    t4 = pd.read_csv(R62_T4)
    t0 = pd.read_csv(R62_T0)
    pairs = {
        "pai": ("T4_pai_a1", "T0m_pai_a1"),
        "euclidean": ("T4_bank_euclidean", "T0m_bank_euclidean"),
        "original": ("T4_original", "T0m_original"),
    }
    result = {"status": "complete", "comparisons": {}}
    for label, (left, right) in pairs.items():
        left_frame = t4[t4["method"] == left].copy()
        right_frame = t0[t0["method"] == right].copy()
        merged = left_frame.merge(
            right_frame,
            on=["seed", "k"],
            suffixes=("_t4", "_t0"),
            validate="one_to_one",
        )
        comparison = {}
        for metric in ("pr_auc", "roc_auc"):
            point = merged[merged["k"] == merged["k"].min()]
            comparison[metric] = paired_test(
                point[f"{metric}_t4"] - point[f"{metric}_t0"]
            )
        for k in sorted(merged["k"].unique()):
            rows = merged[merged["k"] == k]
            comparison[f"incident_hits_k{k}"] = paired_test(
                rows["incident_hits_t4"] - rows["incident_hits_t0"]
            )
        result["comparisons"][label] = comparison
    return result


def compare_smd() -> dict:
    if not SMD_T4.exists() or not SMD_T0.exists():
        return {"status": "pending"}
    t4 = pd.read_csv(SMD_T4)
    t0 = pd.read_csv(SMD_T0)
    merged = t4.merge(
        t0,
        on="machine",
        suffixes=("_t4", "_t0"),
        validate="one_to_one",
    )
    result = {"status": "complete", "comparisons": {}}
    for variant in ("original", "cos", "euc", "pai"):
        comparison = {}
        for metric in (
            "roc_auc",
            "pr_auc",
            "f1_q99",
            "pa_f1_top5",
        ):
            comparison[metric] = paired_test(
                merged[f"{variant}_{metric}_t4"]
                - merged[f"{variant}_{metric}_t0"]
            )
        result["comparisons"][variant] = comparison
    return result


def main() -> None:
    payload = {
        "r62_joint_vs_matched_t0": compare_r62(),
        "smd_joint_vs_matched_t0": compare_smd(),
        "protocol_note": (
            "r6.2 uses the same fixed 1,000-user subset and three seeds; "
            "SMD uses the same 512-point windows and seed."
        ),
    }
    (OUT / "joint_control_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
