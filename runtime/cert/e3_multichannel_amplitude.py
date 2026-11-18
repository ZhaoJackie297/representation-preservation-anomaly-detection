from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e3"
)
OUT.mkdir(parents=True, exist_ok=True)
E1_DIR = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e0_e1"
    / "r42"
)
SEEDS = [20261116, 20261117, 20261118]
K_GRID = (5, 10, 20)
COOLDOWN_DAYS = 7
T2_WINDOW = 32
PAI_W_G = 0.6
PAI_W_Q = 0.4

sys.path.insert(0, str(ROOT))
from e0_e1_representation_preservation import (  # noqa: E402
    align_raw_features,
    causal_moving_mean,
    global_train_z,
    group_indices,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
)


def robust_statistics(
    raw: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    *,
    user_specific: bool,
) -> tuple[np.ndarray, np.ndarray]:
    users = raw.shape[0]
    train_values = raw[:, : train_end + 1]
    train_active = active_mask[:, : train_end + 1]
    all_rows = train_values[train_active]
    global_median = np.median(all_rows, axis=0)
    global_mad = np.median(
        np.abs(all_rows - global_median),
        axis=0,
    )
    global_scale = np.maximum(1.4826 * global_mad, 1e-6)
    if not user_specific:
        return (
            np.tile(global_median, (users, 1)),
            np.tile(global_scale, (users, 1)),
        )
    medians = np.tile(global_median, (users, 1))
    scales = np.tile(global_scale, (users, 1))
    for user_index in range(users):
        rows = train_values[user_index][train_active[user_index]]
        if len(rows) < 2:
            continue
        med = np.median(rows, axis=0)
        mad = np.median(np.abs(rows - med), axis=0)
        medians[user_index] = med
        scales[user_index] = np.maximum(1.4826 * mad, 1e-6)
    return medians, scales


def aggregate_components(
    raw: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    *,
    user_specific: bool,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray]:
    medians, scales = robust_statistics(
        raw,
        active_mask,
        train_end,
        user_specific=user_specific,
    )
    point = np.abs(raw - medians[:, None, :])
    point /= scales[:, None, :]
    local_mean = causal_moving_mean(raw, T2_WINDOW)
    shift = np.abs(local_mean - medians[:, None, :])
    shift /= scales[:, None, :]

    if aggregation == "feature_top2":
        point_order = np.sort(point, axis=2)
        shift_order = np.sort(shift, axis=2)
        G = point_order[:, :, -2:].mean(axis=2)
        Q = shift_order[:, :, -2:].mean(axis=2)
    else:
        groups = group_indices()
        point_channels = np.stack(
            [
                point[:, :, indices].max(axis=2)
                for indices in groups
            ],
            axis=2,
        )
        shift_channels = np.stack(
            [
                shift[:, :, indices].max(axis=2)
                for indices in groups
            ],
            axis=2,
        )
        if aggregation == "channel_mean":
            G = point_channels.mean(axis=2)
            Q = shift_channels.mean(axis=2)
        elif aggregation == "channel_top1":
            G = point_channels.max(axis=2)
            Q = shift_channels.max(axis=2)
        elif aggregation == "channel_top2":
            point_order = np.sort(point_channels, axis=2)
            shift_order = np.sort(shift_channels, axis=2)
            G = point_order[:, :, -2:].mean(axis=2)
            Q = shift_order[:, :, -2:].mean(axis=2)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    G[~active_mask] = np.nan
    Q[~active_mask] = np.nan
    return G.astype(np.float32), Q.astype(np.float32)


def build_alerts(
    signal: np.ndarray,
    active_mask: np.ndarray,
    k: int,
) -> np.ndarray:
    alerts = np.zeros(signal.shape, dtype=np.uint8)
    last_alert = np.full(signal.shape[0], -10000, dtype=np.int32)
    for time_index in range(signal.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= COOLDOWN_DAYS
        )
        candidates = np.flatnonzero(available)
        if len(candidates) == 0:
            continue
        order = candidates[
            np.lexsort(
                (candidates, -signal[candidates, time_index])
            )
        ][:k]
        alerts[order, time_index] = 1
        last_alert[order] = time_index
    return alerts


def point_metrics(
    signal: np.ndarray,
    data: dict,
    active_mask: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    score = signal[:, start : end + 1]
    labels = data["labels"][:, start : end + 1]
    active = active_mask[:, start : end + 1]
    valid = active & np.isfinite(score)
    return {
        "pr_auc": float(
            average_precision_score(labels[valid], score[valid])
        ),
        "roc_auc": float(
            roc_auc_score(labels[valid], score[valid])
        ),
        "positive_days": int(labels[valid].sum()),
    }


def evaluate_variant(
    name: str,
    signal: np.ndarray,
    data: dict,
    active_mask: np.ndarray,
    train_end: int,
    val_end: int,
    seed: int,
) -> list[dict]:
    rank_signal = signal.astype(np.float32, copy=True)
    rank_signal[~np.isfinite(rank_signal)] = -np.inf
    test_start = val_end + 1
    test_end = len(data["dates"]) - 1
    rows = []
    for split, start, end in [
        ("validation", train_end + 1, val_end),
        ("test", test_start, test_end),
    ]:
        point = point_metrics(
            signal,
            data,
            active_mask,
            start,
            end,
        )
        for k in K_GRID:
            alerts = build_alerts(rank_signal, active_mask, k)
            result = evaluate_alert_system(
                data,
                alerts,
                name,
                k,
                external=False,
            )
            rows.append(
                {
                    "seed": seed,
                    "method": name,
                    "split": split,
                    "k": k,
                    **compact_metric(result, split),
                    **point,
                }
            )
    return rows


def main() -> None:
    dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    data = load_r42()
    raw = align_raw_features(
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        users,
        dates,
    )
    configurations = [
        ("A0_global_channel_mean", False, "channel_mean"),
        ("A1_user_channel_mean", True, "channel_mean"),
        ("A2_user_channel_top1", True, "channel_top1"),
        ("A3_user_channel_top2", True, "channel_top2"),
        ("A4_user_feature_top2", True, "feature_top2"),
    ]
    rows = []
    saved_components = {}
    for seed in SEEDS:
        score_store = np.load(
            E1_DIR / f"seed_{seed}" / "scores.npz",
            allow_pickle=False,
        )
        euc = score_store["euc_score"]
        masked = score_store["masked_raw"]
        euc_z = global_train_z(euc, active_mask, train_end)
        masked_z = global_train_z(masked, active_mask, train_end)
        rows.extend(
            evaluate_variant(
                "S0_mask_global_z",
                masked_z,
                data,
                active_mask,
                train_end,
                val_end,
                seed,
            )
        )
        rows.extend(
            evaluate_variant(
                "S2_bank_euc_global",
                euc_z,
                data,
                active_mask,
                train_end,
                val_end,
                seed,
            )
        )
        for name, user_specific, aggregation in configurations:
            G, Q = aggregate_components(
                raw,
                active_mask,
                train_end,
                user_specific=user_specific,
                aggregation=aggregation,
            )
            G_z = global_train_z(G, active_mask, train_end)
            Q_z = global_train_z(Q, active_mask, train_end)
            pai = euc_z + PAI_W_G * G_z + PAI_W_Q * Q_z
            amplitude = PAI_W_G * G_z + PAI_W_Q * Q_z
            rows.extend(
                evaluate_variant(
                    f"{name}_pai",
                    pai,
                    data,
                    active_mask,
                    train_end,
                    val_end,
                    seed,
                )
            )
            rows.extend(
                evaluate_variant(
                    f"{name}_amplitude_only",
                    amplitude,
                    data,
                    active_mask,
                    train_end,
                    val_end,
                    seed,
                )
            )
            if name == "A3_user_channel_top2":
                saved_components[f"seed_{seed}_G"] = G
                saved_components[f"seed_{seed}_Q"] = Q
                saved_components[f"seed_{seed}_pai"] = pai
            print(f"E3 seed={seed} {name} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        frame.groupby(["method", "split", "k"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            event_recall_3d_std=("event_recall_3d", "std"),
            precision_at_k_mean=("precision_at_k", "mean"),
            precision_at_k_std=("precision_at_k", "std"),
            incident_hits_mean=("incident_hits", "mean"),
        )
        .sort_values(["split", "k", "method"])
    )
    summary.to_csv(
        OUT / "metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    if saved_components:
        np.savez_compressed(
            OUT / "a3_components.npz",
            **saved_components,
        )
    print(
        summary[
            (summary["k"] == 10)
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
