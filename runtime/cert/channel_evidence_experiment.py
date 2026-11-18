from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
ARRAYS_FILE = OUT / "ts2vec_dual_score_arrays.npz"

K_GRID = [3, 5, 10, 20, 30, 50]
PRIMARY_K = 10
ALPHA = 0.4
COOLDOWN_DAYS = 7

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
from production_protocol_r42 import build_alerts  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    daily_percentile,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from ts2vec_dual_score_experiment import (  # noqa: E402
    ENCODE_ARGS,
    load_model,
)


def encode_plain(
    model,
    values: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        model.encode(values, **ENCODE_ARGS),
        dtype=np.float32,
    )


def channel_scores(
    model,
    values: np.ndarray,
    plain: np.ndarray,
) -> dict[str, np.ndarray]:
    raw_scores = {}
    for channel, indices in CHANNELS.items():
        print(f"  encoding channel ablation: {channel}", flush=True)
        modified = values.copy()
        modified[:, :, indices] = 0.0
        representation = np.asarray(
            model.encode(modified, **ENCODE_ARGS),
            dtype=np.float32,
        )
        raw_scores[channel] = np.abs(
            representation - plain
        ).sum(axis=-1)
        del modified, representation
    return raw_scores


def calibrate_channels(
    raw_scores: dict[str, np.ndarray],
    active_mask: np.ndarray,
    train_end: int,
) -> dict[str, np.ndarray]:
    return {
        channel: per_user_quantile_scores(
            scores, active_mask, train_end
        )
        for channel, scores in raw_scores.items()
    }


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
        users,
        dates,
        quantile.astype(np.float32),
        active_mask,
    )
    return daily_percentile(dynamic, active_mask).astype(np.float32)


def add_application_utilities(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    for profile_name, weights in PROFILES.items():
        metrics[profile_name] = sum(
            metrics[column] * weight
            for column, weight in weights.items()
        )
    return metrics


def evaluate_signal(
    data: dict,
    signal: np.ndarray,
    auxiliary: np.ndarray,
    method: str,
    *,
    external: bool,
    use_iforest: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        q = q_from_alpha(k, ALPHA) if use_iforest else 0
        if use_iforest:
            alerts = build_unique_quota_alerts(
                signal,
                auxiliary,
                data["active_mask"],
                k,
                q,
            )
        else:
            alerts = build_alerts(
                signal,
                data["active_mask"],
                k,
                COOLDOWN_DAYS,
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


def build_rankings(k10: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (block, split), group in k10.groupby(["data_block", "split"]):
        for profile in PROFILES:
            ordered = group.sort_values(
                [profile, "precision_at_k"],
                ascending=False,
            ).copy()
            ordered["rank"] = np.arange(1, len(ordered) + 1)
            for row in ordered.itertuples(index=False):
                rows.append(
                    {
                        "data_block": block,
                        "split": split,
                        "profile": profile,
                        "method": row.method,
                        "utility": float(getattr(row, profile)),
                        "rank": int(row.rank),
                    }
                )
    return pd.DataFrame(rows)


def summarize_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    maxima = rankings.groupby(
        ["data_block", "split", "profile"]
    )["utility"].transform("max")
    rankings = rankings.copy()
    rankings["relative_utility"] = rankings["utility"] / maxima
    rankings["regret"] = 1.0 - rankings["relative_utility"]
    return (
        rankings.groupby("method", as_index=False)
        .agg(
            appearances=("rank", "size"),
            first_place=(
                "rank",
                lambda values: int((values == 1).sum()),
            ),
            top_three=(
                "rank",
                lambda values: int((values <= 3).sum()),
            ),
            mean_rank=("rank", "mean"),
            worst_rank=("rank", "max"),
            mean_utility=("utility", "mean"),
            mean_relative_utility=("relative_utility", "mean"),
            worst_relative_utility=("relative_utility", "min"),
            mean_regret=("regret", "mean"),
            worst_regret=("regret", "max"),
        )
        .sort_values(
            ["mean_rank", "mean_regret"],
            ascending=[True, True],
        )
    )


def main() -> None:
    np.random.seed(20261109)
    torch.manual_seed(20261109)

    arrays = np.load(ARRAYS_FILE, allow_pickle=False)
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
    tasks = [
        {
            "name": "r42_original",
            "data": r42,
            "values": r42_dataset["values"].astype(np.float32),
            "model": ROOT
            / "processed_r42"
            / "r4.2_ts2vec_model.pth",
            "train_end": int(r42_dataset["train_end_index"]),
            "external": False,
        },
        {
            "name": "r42_retrained",
            "data": r42,
            "values": r42_dataset["values"].astype(np.float32),
            "model": ROOT
            / "processed_r42"
            / "innovation_sim"
            / "r4.2_ts2vec_quota_retrain.pth",
            "train_end": int(r42_dataset["train_end_index"]),
            "external": False,
        },
        {
            "name": "r52_external",
            "data": r52,
            "values": r52_dataset["values"].astype(np.float32),
            "model": ROOT
            / "processed_r52"
            / "r5.2_ts2vec_model.pth",
            "train_end": int(r52_dataset["train_end_index"]),
            "external": True,
        },
    ]

    metric_frames = []
    saved_arrays = {}
    diagnostics = []
    for task in tasks:
        name = str(task["name"])
        print(f"\nEvaluating {name}", flush=True)
        data = task["data"]
        values = task["values"]
        active = data["active_mask"]
        users = data["users"]
        dates = data["dates"]
        train_end = int(task["train_end"])
        auxiliary = arrays[f"{name}_dual_view_iforest"].astype(np.float32)
        fixed_mask = arrays[f"{name}_ts_mask_quantile"].astype(np.float32)
        fixed_latent = arrays[f"{name}_ts_latent_quantile"].astype(
            np.float32
        )
        fixed_signal = arrays[
            f"{name}_ts_dual_mean_signal"
        ].astype(np.float32)

        model = load_model(task["model"], values.shape[-1])
        print("  encoding plain representation", flush=True)
        plain = encode_plain(model, values)
        raw_channels = channel_scores(model, values, plain)
        channel_quantiles = calibrate_channels(
            raw_channels, active, train_end
        )
        stacked = np.stack(
            [
                channel_quantiles[channel]
                for channel in CHANNELS
            ],
            axis=0,
        )
        top2_quantile = top_two_mean(stacked)
        channel_signal = dynamic_signal(
            users, dates, active, top2_quantile
        )
        triple_quantile = (
            fixed_mask + fixed_latent + top2_quantile
        ) / 3.0
        triple_signal = dynamic_signal(
            users, dates, active, triple_quantile
        )

        variants = [
            ("fixed_dual_iforest", fixed_signal, True),
            ("channel_top2_iforest", channel_signal, True),
            ("triple_mean_iforest", triple_signal, True),
            ("triple_mean_complete", triple_signal, False),
        ]
        for method, signal, use_iforest in variants:
            frame = evaluate_signal(
                data,
                signal,
                auxiliary,
                method,
                external=bool(task["external"]),
                use_iforest=use_iforest,
            )
            frame.insert(0, "data_block", name)
            frame["external"] = int(bool(task["external"]))
            metric_frames.append(frame)
            saved_arrays[f"{name}_{method}_signal"] = signal.astype(
                np.float32
            )

        iforest = evaluate_signal(
            data,
            auxiliary,
            auxiliary,
            "iforest_complete",
            external=bool(task["external"]),
            use_iforest=False,
        )
        iforest.insert(0, "data_block", name)
        iforest["external"] = int(bool(task["external"]))
        metric_frames.append(iforest)

        for channel, raw in raw_channels.items():
            saved_arrays[f"{name}_channel_{channel}_raw"] = raw.astype(
                np.float32
            )
            saved_arrays[
                f"{name}_channel_{channel}_quantile"
            ] = channel_quantiles[channel].astype(np.float32)
        saved_arrays[f"{name}_channel_top2_quantile"] = (
            top2_quantile.astype(np.float32)
        )
        saved_arrays[f"{name}_triple_mean_quantile"] = (
            triple_quantile.astype(np.float32)
        )
        saved_arrays[f"{name}_channel_top2_signal"] = channel_signal
        saved_arrays[f"{name}_triple_mean_signal"] = triple_signal

        diagnostics.append(
            {
                "data_block": name,
                "channels": list(CHANNELS),
                "representation_dims": int(plain.shape[-1]),
                "channel_raw_shapes": {
                    channel: list(raw.shape)
                    for channel, raw in raw_channels.items()
                },
            }
        )
        del model, plain
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = add_application_utilities(
        pd.concat(metric_frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "channel_evidence_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    k10 = metrics[metrics["k"] == PRIMARY_K].copy()
    rankings = build_rankings(k10)
    stability = summarize_stability(rankings)
    rankings.to_csv(
        OUT / "channel_evidence_k10_rankings.csv",
        index=False,
        encoding="utf-8",
    )
    stability.to_csv(
        OUT / "channel_evidence_k10_rank_stability.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "channel_evidence_arrays.npz",
        **saved_arrays,
    )

    summary = {
        "experiment": "channel-ablation evidence",
        "channels": CHANNELS,
        "aggregation": "mean of top two channel quantiles",
        "triple_score": "equal mean of mask, latent, channel-top2",
        "primary_k": PRIMARY_K,
        "q": q_from_alpha(PRIMARY_K, ALPHA),
        "labels_used": False,
        "diagnostics": diagnostics,
        "outputs": {
            "metrics": str(OUT / "channel_evidence_metrics.csv"),
            "k10_rankings": str(
                OUT / "channel_evidence_k10_rankings.csv"
            ),
            "k10_rank_stability": str(
                OUT / "channel_evidence_k10_rank_stability.csv"
            ),
            "arrays": str(OUT / "channel_evidence_arrays.npz"),
        },
    }
    (OUT / "channel_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nK=10 rank stability", flush=True)
    print(stability.to_string(index=False), flush=True)
    print("\nK=10 winners", flush=True)
    print(
        rankings[rankings["rank"] == 1].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
