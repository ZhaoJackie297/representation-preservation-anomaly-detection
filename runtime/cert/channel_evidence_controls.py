from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
K_GRID = [3, 5, 10, 20, 30, 50]
PRIMARY_K = 10
ALPHA = 0.4

CHANNELS = {
    "logon": list(range(1, 8)),
    "device": [8, 9],
    "file": list(range(10, 14)),
    "email": list(range(14, 20)),
    "http": list(range(20, 24)),
}

sys.path.insert(0, str(ROOT))
from application_utility_analysis import PROFILES  # noqa: E402
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    daily_percentile,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)
from risk_topk_r42 import compute_risk  # noqa: E402


def robust_personal_channel_scores(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> dict[str, np.ndarray]:
    scores = {}
    for channel, indices in CHANNELS.items():
        channel_values = values[:, :, indices]
        deviations = np.full(
            values.shape[:2], np.nan, dtype=np.float64
        )
        for user_index in range(values.shape[0]):
            training_indices = np.flatnonzero(
                active_mask[user_index, : train_end + 1]
                & np.isfinite(
                    channel_values[user_index, : train_end + 1]
                ).all(axis=1)
            )
            if len(training_indices) == 0:
                median = np.zeros(len(indices), dtype=np.float64)
                scale = np.ones(len(indices), dtype=np.float64)
            else:
                train_rows = channel_values[
                    user_index, training_indices
                ]
                median = np.median(train_rows, axis=0)
                mad = np.median(
                    np.abs(train_rows - median), axis=0
                )
                scale = 1.4826 * mad
                global_scale = np.std(train_rows, axis=0)
                scale = np.where(
                    scale >= 1e-8,
                    scale,
                    np.where(global_scale >= 1e-8, global_scale, 1.0),
                )
            active_indices = np.flatnonzero(active_mask[user_index])
            z = np.abs(
                (
                    channel_values[user_index, active_indices]
                    - median
                )
                / scale
            )
            deviations[user_index, active_indices] = np.mean(z, axis=1)
        scores[channel] = deviations.astype(np.float32)
    return scores


def top_two_mean(stacked: np.ndarray) -> np.ndarray:
    ordered = np.sort(stacked, axis=0)
    return ordered[-2:].mean(axis=0)


def dynamic_signal(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    quantile: np.ndarray,
) -> np.ndarray:
    dynamic = compute_risk(
        users, dates, quantile, active_mask
    )
    return daily_percentile(dynamic, active_mask).astype(np.float32)


def permute_daily(
    scores: np.ndarray,
    active_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permuted = np.full(scores.shape, np.nan, dtype=np.float32)
    for time_index in range(scores.shape[1]):
        active = np.flatnonzero(active_mask[:, time_index])
        if len(active) == 0:
            continue
        permuted[active, time_index] = rng.permutation(
            scores[active, time_index]
        )
    return permuted


def add_utilities(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for profile, weights in PROFILES.items():
        frame[profile] = sum(
            frame[column] * weight
            for column, weight in weights.items()
        )
    return frame


def evaluate(
    data: dict,
    signal: np.ndarray,
    auxiliary: np.ndarray,
    method: str,
    *,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        q = q_from_alpha(k, ALPHA)
        alerts = build_unique_quota_alerts(
            signal,
            auxiliary,
            data["active_mask"],
            k,
            q,
        )
        result = evaluate_alert_system(
            data,
            alerts,
            method,
            k,
            external=external,
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "k": k,
                    "q": q,
                    **compact_metric(
                        result, None if external else split
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    channel_arrays = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    dual_arrays = np.load(
        OUT / "ts2vec_dual_score_arrays.npz",
        allow_pickle=False,
    )
    r42 = load_r42()
    r52 = load_r52()
    r42_dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    r52_dataset = np.load(
        ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    blocks = [
        {
            "name": "r42_original",
            "data": r42,
            "values": r42_dataset["values"].astype(np.float32),
            "train_end": int(r42_dataset["train_end_index"]),
            "external": False,
        },
        {
            "name": "r42_retrained",
            "data": r42,
            "values": r42_dataset["values"].astype(np.float32),
            "train_end": int(r42_dataset["train_end_index"]),
            "external": False,
        },
        {
            "name": "r52_external",
            "data": r52,
            "values": r52_dataset["values"].astype(np.float32),
            "train_end": int(r52_dataset["train_end_index"]),
            "external": True,
        },
    ]

    frames = []
    diagnostics = []
    saved = {}
    for seed_offset, block in enumerate(blocks):
        name = str(block["name"])
        data = block["data"]
        active = data["active_mask"]
        users = data["users"]
        dates = data["dates"]
        train_end = int(block["train_end"])
        auxiliary = dual_arrays[f"{name}_dual_view_iforest"]
        mask_q = channel_arrays[
            f"{name}_triple_mean_quantile"
        ].copy()
        fixed_dual_signal = channel_arrays[
            f"{name}_fixed_dual_iforest_signal"
        ]
        channel_signal = channel_arrays[
            f"{name}_triple_mean_iforest_signal"
        ]
        channel_top2_q = channel_arrays[
            f"{name}_channel_top2_quantile"
        ]

        feature_raw = robust_personal_channel_scores(
            block["values"], active, train_end
        )
        feature_q = {
            channel: per_user_quantile_scores(
                raw, active, train_end
            )
            for channel, raw in feature_raw.items()
        }
        feature_top2_q = top_two_mean(
            np.stack(
                [
                    feature_q[channel]
                    for channel in CHANNELS
                ],
                axis=0,
            )
        )
        fixed_mask_q = dual_arrays[
            f"{name}_ts_mask_quantile"
        ]
        fixed_latent_q = dual_arrays[
            f"{name}_ts_latent_quantile"
        ]
        feature_triple_q = (
            fixed_mask_q + fixed_latent_q + feature_top2_q
        ) / 3.0
        feature_signal = dynamic_signal(
            users, dates, active, feature_triple_q
        )

        permuted_channel_q = permute_daily(
            channel_top2_q,
            active,
            seed=20261109 + seed_offset,
        )
        permuted_triple_q = (
            fixed_mask_q + fixed_latent_q + permuted_channel_q
        ) / 3.0
        permuted_signal = dynamic_signal(
            users, dates, active, permuted_triple_q
        )

        variants = {
            "fixed_dual_iforest": fixed_dual_signal,
            "ts_channel_triple_iforest": channel_signal,
            "feature_channel_triple_iforest": feature_signal,
            "permuted_channel_triple_iforest": permuted_signal,
        }
        for method, signal in variants.items():
            frame = evaluate(
                data,
                signal,
                auxiliary,
                method,
                external=bool(block["external"]),
            )
            frame.insert(0, "data_block", name)
            frame["external"] = int(bool(block["external"]))
            frames.append(frame)
            saved[f"{name}_{method}_signal"] = signal.astype(
                np.float32
            )
        saved[f"{name}_feature_channel_top2_quantile"] = (
            feature_top2_q.astype(np.float32)
        )
        saved[f"{name}_permuted_channel_top2_quantile"] = (
            permuted_channel_q.astype(np.float32)
        )
        diagnostics.append(
            {
                "data_block": name,
                "feature_triple_active_median": float(
                    np.nanmedian(feature_triple_q)
                ),
                "permuted_triple_active_median": float(
                    np.nanmedian(permuted_triple_q)
                ),
            }
        )

    metrics = add_utilities(
        pd.concat(frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "channel_evidence_control_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    k10 = metrics[metrics["k"] == PRIMARY_K]
    summary_table = (
        k10.groupby("method", as_index=False)
        .agg(
            mean_utility=("balanced", "mean"),
            mean_recall=("event_recall_3d", "mean"),
            mean_scenario=("scenario_macro_recall_3d", "mean"),
            mean_precision=("precision_at_k", "mean"),
        )
        .sort_values("mean_utility", ascending=False)
    )
    summary_table.to_csv(
        OUT / "channel_evidence_control_k10_summary.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "channel_evidence_control_arrays.npz",
        **saved,
    )
    summary = {
        "primary_k": PRIMARY_K,
        "labels_used": False,
        "controls": [
            "feature-channel triple",
            "permuted-channel triple",
        ],
        "diagnostics": diagnostics,
        "outputs": {
            "metrics": str(
                OUT / "channel_evidence_control_metrics.csv"
            ),
            "k10_summary": str(
                OUT / "channel_evidence_control_k10_summary.csv"
            ),
            "arrays": str(
                OUT / "channel_evidence_control_arrays.npz"
            ),
        },
    }
    (OUT / "channel_evidence_control_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_table.to_string(index=False), flush=True)
    print("\nBy K", flush=True)
    print(
        (
            100
            * metrics.groupby(["k", "method"])["balanced"]
            .mean()
            .unstack()
        )
        .round(2)
        .to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
