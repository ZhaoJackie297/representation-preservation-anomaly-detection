from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e5"
    / "r52"
)
OUT.mkdir(parents=True, exist_ok=True)
E1_DIR = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e0_e1"
    / "r52"
)
SEEDS = [20261116, 20261117, 20261118]
K_GRID = (5, 10, 20)

sys.path.insert(0, str(ROOT))
from e0_e1_representation_preservation import (  # noqa: E402
    align_raw_features,
    global_train_z,
)
from e3_multichannel_amplitude import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    aggregate_components,
    build_alerts,
    point_metrics,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r52,
)


def evaluate_external(
    name: str,
    signal: np.ndarray,
    data: dict,
    active_mask: np.ndarray,
    test_start: int,
    seed: int,
) -> list[dict]:
    rank_signal = signal.astype(np.float32, copy=True)
    rank_signal[~np.isfinite(rank_signal)] = -np.inf
    end = len(data["dates"]) - 1
    point = point_metrics(
        signal,
        data,
        active_mask,
        test_start,
        end,
    )
    rows = []
    for k in K_GRID:
        alerts = build_alerts(rank_signal, active_mask, k)
        result = evaluate_alert_system(
            data,
            alerts,
            name,
            k,
            external=True,
        )
        rows.append(
            {
                "seed": seed,
                "method": name,
                "split": "external_test",
                "k": k,
                **compact_metric(result, None),
                **point,
            }
        )
    return rows


def main() -> None:
    dataset = np.load(
        ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    test_start = int(dataset["test_start_index"])
    data = load_r52()
    raw = align_raw_features(
        ROOT / "processed_r52" / "r5.2_user_day_features.csv",
        users,
        dates,
    )
    g_a1, q_a1 = aggregate_components(
        raw,
        active_mask,
        train_end,
        user_specific=True,
        aggregation="channel_mean",
    )
    g_a1_z = global_train_z(g_a1, active_mask, train_end)
    q_a1_z = global_train_z(q_a1, active_mask, train_end)
    rows = []
    for seed in SEEDS:
        store = np.load(
            E1_DIR / f"seed_{seed}" / "scores.npz",
            allow_pickle=False,
        )
        masked = store["masked_raw"]
        cos_score = store["cos_score"]
        euc_score = store["euc_score"]
        g_a3 = store["G"]
        q_a3 = store["Q"]
        masked_z = global_train_z(masked, active_mask, train_end)
        cos_z = global_train_z(cos_score, active_mask, train_end)
        euc_z = global_train_z(euc_score, active_mask, train_end)
        g_a3_z = global_train_z(g_a3, active_mask, train_end)
        q_a3_z = global_train_z(q_a3, active_mask, train_end)
        pai_a1 = euc_z + PAI_W_G * g_a1_z + PAI_W_Q * q_a1_z
        pai_a3 = euc_z + PAI_W_G * g_a3_z + PAI_W_Q * q_a3_z
        variants = {
            "original_ts2vec_global": masked_z,
            "bank_cosine_global": cos_z,
            "bank_euclidean_global": euc_z,
            "pai_a1_channel_mean": pai_a1,
            "pai_a3_channel_top2": pai_a3,
        }
        for name, signal in variants.items():
            rows.extend(
                evaluate_external(
                    name,
                    signal,
                    data,
                    active_mask,
                    test_start,
                    seed,
                )
            )
        np.savez_compressed(
            OUT / f"seed_{seed}_signals.npz",
            **{name: signal for name, signal in variants.items()},
        )
        print(f"E5 r5.2 seed={seed} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        frame.groupby(["method", "k"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            event_recall_3d_std=("event_recall_3d", "std"),
            precision_at_k_mean=("precision_at_k", "mean"),
            precision_at_k_std=("precision_at_k", "std"),
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_std=("incident_hits", "std"),
        )
        .sort_values(["k", "method"])
    )
    summary.to_csv(
        OUT / "metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    pivot = frame.pivot_table(
        index=["seed", "k"],
        columns="method",
        values="incident_hits",
    )
    comparisons = []
    for candidate in (
        "pai_a1_channel_mean",
        "pai_a3_channel_top2",
        "bank_euclidean_global",
    ):
        delta = (
            pivot[candidate]
            - pivot["original_ts2vec_global"]
        )
        comparisons.append(
            {
                "comparison": (
                    f"{candidate} - original_ts2vec_global"
                ),
                "by_k": {
                    str(k): {
                        "mean_delta": float(
                            delta.xs(k, level="k").mean()
                        ),
                        "seed_deltas": delta.xs(k, level="k")
                        .tolist(),
                    }
                    for k in K_GRID
                },
            }
        )
    (OUT / "comparisons.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
