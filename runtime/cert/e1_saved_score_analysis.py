from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e0_e1"
)
SEEDS = [20261116, 20261117, 20261118]

sys.path.insert(0, str(ROOT))
from e0_e1_representation_preservation import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    global_robust,
    global_train_z,
    user_tail_evidence,
    zscore_aligned,
)
from quota_relationship_sim import load_r42  # noqa: E402


def variants_for_seed(
    arrays: np.lib.npyio.NpzFile,
    active_mask: np.ndarray,
    train_end: int,
) -> dict[str, np.ndarray]:
    masked_raw = arrays["masked_raw"]
    cos_score = arrays["cos_score"]
    euc_score = arrays["euc_score"]
    G = arrays["G"]
    Q = arrays["Q"]

    robust = global_robust(masked_raw, active_mask, train_end)
    masked_user = user_tail_evidence(
        masked_raw,
        active_mask,
        train_end,
    )
    masked_global_z = global_train_z(
        masked_raw,
        active_mask,
        train_end,
    )
    cos_global_z = global_train_z(cos_score, active_mask, train_end)
    euc_global_z = global_train_z(euc_score, active_mask, train_end)
    g_global_z = global_train_z(G, active_mask, train_end)
    q_global_z = global_train_z(Q, active_mask, train_end)
    cos_user = user_tail_evidence(cos_score, active_mask, train_end)
    euc_user = user_tail_evidence(euc_score, active_mask, train_end)
    g_user = user_tail_evidence(G, active_mask, train_end)
    q_user = user_tail_evidence(Q, active_mask, train_end)
    cos_user_z = global_train_z(cos_user, active_mask, train_end)
    euc_user_z = global_train_z(euc_user, active_mask, train_end)
    g_user_z = global_train_z(g_user, active_mask, train_end)
    q_user_z = global_train_z(q_user, active_mask, train_end)
    return {
        "S0_mask_raw": masked_raw,
        "S0_mask_global_robust": robust,
        "S0_mask_global_z": masked_global_z,
        "S0_mask_user": masked_user,
        "S1_bank_cos_global": cos_global_z,
        "S1_bank_cos_user": cos_user_z,
        "S2_bank_euc_global": euc_global_z,
        "S2_bank_euc_user": euc_user_z,
        "S3_pai_mag_global": (
            euc_global_z + PAI_W_G * g_global_z
        ),
        "S3_pai_mag_user": euc_user_z + PAI_W_G * g_user_z,
        "S4_pai_t2_global": (
            euc_global_z + PAI_W_Q * q_global_z
        ),
        "S4_pai_t2_user": euc_user_z + PAI_W_Q * q_user_z,
        "S5_pai_full_global": (
            euc_global_z
            + PAI_W_G * g_global_z
            + PAI_W_Q * q_global_z
        ),
        "S5_pai_full_user": (
            euc_user_z
            + PAI_W_G * g_user_z
            + PAI_W_Q * q_user_z
        ),
        "S6_amp_only_global": (
            PAI_W_G * g_global_z + PAI_W_Q * q_global_z
        ),
        "S6_amp_only_user": (
            PAI_W_G * g_user_z + PAI_W_Q * q_user_z
        ),
    }


def point_metric(
    score: np.ndarray,
    labels: np.ndarray,
    active_mask: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    split_score = score[:, start : end + 1]
    split_labels = labels[:, start : end + 1]
    split_active = active_mask[:, start : end + 1]
    valid = split_active & np.isfinite(split_score)
    y_true = split_labels[valid]
    y_score = split_score[valid]
    positives = int(y_true.sum())
    if positives == 0 or positives == len(y_true):
        return {
            "positive_days": positives,
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
        }
    return {
        "positive_days": positives,
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def main() -> None:
    data = load_r42()
    active_mask = data["active_mask"]
    labels = data["labels"]
    train_end = int(data["train_end"])
    val_end = int(data["val_end"])
    test_start = val_end + 1
    test_end = len(data["dates"]) - 1
    rows = []
    for seed in SEEDS:
        arrays = np.load(
            OUT / "r42" / f"seed_{seed}" / "scores.npz",
            allow_pickle=False,
        )
        variants = variants_for_seed(
            arrays,
            active_mask,
            train_end,
        )
        for method, score in variants.items():
            for split, start, end in [
                ("validation", train_end + 1, val_end),
                ("test", test_start, test_end),
            ]:
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "split": split,
                        **point_metric(
                            score,
                            labels,
                            active_mask,
                            start,
                            end,
                        ),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        OUT / "r42" / "point_metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        metrics.groupby(["method", "split"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            positive_days=("positive_days", "first"),
        )
        .sort_values(["split", "pr_auc_mean"], ascending=[True, False])
    )
    summary.to_csv(
        OUT / "r42" / "point_metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    comparison = []
    pivot = metrics.pivot_table(
        index=["seed", "split"],
        columns="method",
        values="pr_auc",
    )
    for split in ("validation", "test"):
        for left, right in [
            ("S2_bank_euc_user", "S1_bank_cos_user"),
            ("S5_pai_full_user", "S2_bank_euc_user"),
            ("S5_pai_full_global", "S2_bank_euc_global"),
        ]:
            values = pivot.loc[(slice(None), split), left] - pivot.loc[
                (slice(None), split), right
            ]
            comparison.append(
                {
                    "split": split,
                    "comparison": f"{left} - {right}",
                    "mean_delta": float(values.mean()),
                    "std_delta": float(values.std()),
                    "positive_seeds": int((values > 0).sum()),
                    "seed_deltas": values.tolist(),
                }
            )
    (OUT / "r42" / "point_comparisons.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
