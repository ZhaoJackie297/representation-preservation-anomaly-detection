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


def compare_frames(
    t0: pd.DataFrame,
    t4: pd.DataFrame,
    pairs: dict[str, tuple[str, str]],
    *,
    split: str,
) -> dict:
    result = {}
    for label, (t0_method, t4_method) in pairs.items():
        left = t0[(t0["method"] == t0_method) & (t0["split"] == split)]
        right = t4[(t4["method"] == t4_method) & (t4["split"] == split)]
        merged = left.merge(
            right,
            on=["seed", "k"],
            suffixes=("_t0", "_t4"),
            validate="one_to_one",
        )
        comparison = {}
        for metric in ("pr_auc", "roc_auc"):
            point = merged[merged["k"] == merged["k"].min()]
            comparison[metric] = paired_test(
                point[f"{metric}_t4"] - point[f"{metric}_t0"]
            )
        for k in sorted(merged["k"].unique()):
            row = merged[merged["k"] == k]
            comparison[f"incident_hits_k{k}"] = paired_test(
                row["incident_hits_t4"] - row["incident_hits_t0"]
            )
        result[label] = comparison
    return result


def main() -> None:
    r42_t0 = pd.read_csv(BASE / "e3" / "metrics_by_seed.csv")
    r42_t4 = pd.read_csv(BASE / "e4_joint" / "metrics_by_seed.csv")
    r42_pairs = {
        "pai": ("A1_user_channel_mean_pai", "T4_full_soft_pai_a1"),
        "euclidean": ("S2_bank_euc_global", "T4_full_soft_euclidean"),
        "cosine": ("S2_bank_euc_global", "T4_full_soft_cosine"),
        "original": ("S0_mask_global_z", "T4_full_soft_original"),
    }

    r52_t0 = pd.read_csv(BASE / "e5" / "r52" / "metrics_by_seed.csv")
    r52_t4 = pd.read_csv(BASE / "e4_joint" / "r52" / "metrics_by_seed.csv")
    if "split" not in r52_t4.columns:
        r52_t4["split"] = "external_test"
    r52_pairs = {
        "pai": (
            "pai_a1_channel_mean",
            "T4_pai_a1",
        ),
        "euclidean": (
            "bank_euclidean_global",
            "T4_bank_euclidean",
        ),
        "cosine": (
            "bank_cosine_global",
            "T4_bank_cosine",
        ),
        "original": (
            "original_ts2vec_global",
            "T4_original",
        ),
    }
    payload = {
        "r42_validation": compare_frames(
            r42_t0,
            r42_t4,
            r42_pairs,
            split="validation",
        ),
        "r42_test": compare_frames(
            r42_t0,
            r42_t4,
            r42_pairs,
            split="test",
        ),
        "r52_external_test": compare_frames(
            r52_t0,
            r52_t4,
            r52_pairs,
            split="external_test",
        ),
        "note": (
            "All comparisons are paired by random seed. PR-AUC and "
            "ROC-AUC are point-level metrics; incident hits are evaluated "
            "under the frozen fixed-K protocol."
        ),
    }
    (OUT / "cert_joint_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
