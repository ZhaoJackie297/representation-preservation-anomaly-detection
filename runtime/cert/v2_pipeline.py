from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
ALL_K = [3, 5, 10, 20, 30, 50]
ALPHA = 0.4
COOLDOWN_DAYS = 7
SEED = 20261109

sys.path.insert(0, str(ROOT))
from application_utility_analysis import PROFILES  # noqa: E402
from channel_evidence_experiment import (  # noqa: E402
    CHANNELS,
    calibrate_channels,
    channel_scores,
    encode_plain,
    top_two_mean,
)
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
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
    latent_deviation,
    load_model,
)


BLOCKS = {
    "r42_original": {
        "family": "r42",
        "model": ROOT / "processed_r42" / "r4.2_ts2vec_model.pth",
        "feature_file": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "external": False,
    },
    "r42_retrained": {
        "family": "r42",
        "model": (
            ROOT
            / "processed_r42"
            / "innovation_sim"
            / "r4.2_ts2vec_quota_retrain.pth"
        ),
        "feature_file": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "external": False,
    },
    "r52_external": {
        "family": "r52",
        "model": ROOT / "processed_r52" / "r5.2_ts2vec_model.pth",
        "feature_file": (
            ROOT / "processed_r52" / "r5.2_user_day_features.csv"
        ),
        "external": True,
    },
}


def add_utilities(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for profile, weights in PROFILES.items():
        frame[profile] = sum(
            frame[column] * weight
            for column, weight in weights.items()
        )
    return frame


def reuse_components(
    block_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    iforest = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    return (
        channel[f"{block_name}_triple_mean_quantile"].astype(np.float32),
        channel[f"{block_name}_triple_mean_signal"].astype(np.float32),
        iforest[
            f"{block_name}_v2_aligned_dual_view_iforest"
        ].astype(np.float32),
    )


def recompute_components(
    block_name: str,
    config: dict,
    data: dict,
    sequence_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = np.load(sequence_path, allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    train_end = (
        int(dataset["train_end_index"])
        if "train_end_index" in dataset.files
        else data["test_start"] - 1
    )
    model = load_model(config["model"], values.shape[-1])

    plain = encode_plain(model, values)
    if block_name == "r42_original":
        score_arrays = np.load(
            ROOT / "processed_r42" / "r4.2_ts2vec_scores.npz",
            allow_pickle=False,
        )
        masked_raw = score_arrays["raw_error"].astype(np.float32)
        masked = None
    elif block_name == "r52_external":
        score_arrays = np.load(
            ROOT / "processed_r52" / "r5.2_ts2vec_scores.npz",
            allow_pickle=False,
        )
        masked_raw = score_arrays["anomaly_score"].astype(np.float32)
        masked = None
    else:
        print("  encoding masked representation", flush=True)
        masked = np.asarray(
            model.encode(values, mask="mask_last", **ENCODE_ARGS),
            dtype=np.float32,
        )
        masked_raw = np.abs(masked - plain).sum(axis=-1)
    latent_raw, _ = latent_deviation(
        plain, data["active_mask"], train_end
    )
    print("  encoding five channel ablations", flush=True)
    channel_raw = channel_scores(model, values, plain)

    masked_q = per_user_quantile_scores(
        masked_raw, data["active_mask"], train_end
    )
    latent_q = per_user_quantile_scores(
        latent_raw, data["active_mask"], train_end
    )
    channel_q = calibrate_channels(
        channel_raw, data["active_mask"], train_end
    )
    channel_top2_q = top_two_mean(
        np.stack(
            [channel_q[channel] for channel in CHANNELS],
            axis=0,
        )
    )
    triple_q = (masked_q + latent_q + channel_top2_q) / 3.0
    triple_dynamic = compute_risk(
        data["users"], data["dates"], triple_q, data["active_mask"]
    )
    triple_signal = daily_percentile(
        triple_dynamic, data["active_mask"]
    ).astype(np.float32)
    original_iforest, augmented_iforest = fit_view_scores(
        data,
        config["feature_file"],
        triple_dynamic,
        data["personal"],
        SEED,
    )
    auxiliary = np.maximum(
        original_iforest, augmented_iforest
    ).astype(np.float32)
    del model, masked, plain
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return triple_q, triple_signal, auxiliary


def select_queue_with_sources(
    triple_signal: np.ndarray,
    auxiliary: np.ndarray,
    active_mask: np.ndarray,
    k: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    q_limit = q_from_alpha(k, ALPHA)
    alerts = np.zeros(triple_signal.shape, dtype=np.uint8)
    rows = []
    last_alert = np.full(triple_signal.shape[0], -10_000, dtype=np.int32)
    for time_index in range(triple_signal.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= COOLDOWN_DAYS
        )
        candidates = np.flatnonzero(available)
        if len(candidates) == 0:
            continue
        ts_order = candidates[
            np.lexsort(
                (
                    candidates,
                    -triple_signal[candidates, time_index],
                )
            )
        ]
        aux_order = candidates[
            np.lexsort(
                (
                    candidates,
                    -auxiliary[candidates, time_index],
                )
            )
        ]
        ts_top_k = set(ts_order[:k].tolist())
        selected = [
            (int(user), "ts_main")
            for user in ts_order[: k - q_limit]
        ]
        selected_users = {user for user, _ in selected}
        for user in aux_order:
            user = int(user)
            if user in selected_users or user in ts_top_k:
                continue
            selected.append((user, "iforest_rescue"))
            selected_users.add(user)
            if len(selected) >= k:
                break
        if len(selected) < k:
            for user in ts_order:
                user = int(user)
                if user in selected_users:
                    continue
                selected.append((user, "ts_fallback"))
                selected_users.add(user)
                if len(selected) >= k:
                    break
        selected = selected[:k]
        selected_users_array = np.asarray(
            [user for user, _ in selected], dtype=np.int64
        )
        alerts[selected_users_array, time_index] = 1
        last_alert[selected_users_array] = time_index
        for rank, (user, source) in enumerate(selected, start=1):
            rows.append(
                {
                    "date_index": time_index,
                    "rank": rank,
                    "user_index": user,
                    "source": source,
                    "ts_score": float(
                        triple_signal[user, time_index]
                    ),
                    "iforest_score": float(
                        auxiliary[user, time_index]
                    ),
                }
            )
    return alerts, pd.DataFrame(rows)


def evaluate_block(
    block_name: str,
    data: dict,
    triple_signal: np.ndarray,
    auxiliary: np.ndarray,
    k_values: list[int],
    *,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in k_values:
        q_limit = q_from_alpha(k, ALPHA)
        alerts = build_unique_quota_alerts(
            triple_signal,
            auxiliary,
            data["active_mask"],
            k,
            q_limit,
        )
        result = evaluate_alert_system(
            data,
            alerts,
            "v2_aligned",
            k,
            external=external,
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            rows.append(
                {
                    "data_block": block_name,
                    "split": split,
                    "method": "v2_aligned",
                    "k": k,
                    "q": q_limit,
                    **compact_metric(
                        result, None if external else split
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(BLOCKS),
        help="Dataset block; repeat to select more than one.",
    )
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--reuse-components", action="store_true")
    args = parser.parse_args()
    selected_blocks = args.dataset or sorted(BLOCKS)
    k_values = sorted(set(ALL_K + [args.primary_k]))

    r42 = load_r42()
    r52 = load_r52()
    families = {"r42": r42, "r52": r52}
    sequence_paths = {
        "r42": ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        "r52": ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
    }

    metric_frames = []
    queue_frames = []
    component_arrays = {}
    for block_name in selected_blocks:
        print(f"\nRunning {block_name}", flush=True)
        config = BLOCKS[block_name]
        data = families[config["family"]]
        if args.reuse_components:
            triple_q, triple_signal, auxiliary = reuse_components(
                block_name
            )
        else:
            triple_q, triple_signal, auxiliary = recompute_components(
                block_name,
                config,
                data,
                sequence_paths[config["family"]],
            )
        metric_frames.append(
            evaluate_block(
                block_name,
                data,
                triple_signal,
                auxiliary,
                k_values,
                external=bool(config["external"]),
            )
        )

        alerts, queue = select_queue_with_sources(
            triple_signal,
            auxiliary,
            data["active_mask"],
            args.primary_k,
        )
        queue["data_block"] = block_name
        queue["date"] = queue["date_index"].map(
            lambda index: data["dates"][index].strftime("%Y-%m-%d")
        )
        queue["user_id"] = queue["user_index"].map(
            lambda index: str(data["users"][index])
        )
        if config["external"]:
            queue = queue[
                queue["date_index"] >= data["test_start"]
            ].copy()
            queue["split"] = "external_test"
        else:
            queue = queue[
                queue["date_index"] > data["train_end"]
            ].copy()
            queue["split"] = np.where(
                queue["date_index"] <= data["val_end"],
                "validation",
                "test",
            )
        queue_frames.append(
            queue[
                [
                    "data_block",
                    "split",
                    "date",
                    "rank",
                    "user_id",
                    "source",
                    "ts_score",
                    "iforest_score",
                ]
            ]
        )
        component_arrays[f"{block_name}_triple_quantile"] = triple_q
        component_arrays[f"{block_name}_triple_signal"] = triple_signal
        component_arrays[f"{block_name}_aligned_iforest"] = auxiliary
        component_arrays[f"{block_name}_alerts_k{args.primary_k}"] = alerts

    metrics = add_utilities(
        pd.concat(metric_frames, ignore_index=True)
    )
    queues = pd.concat(queue_frames, ignore_index=True)
    metrics.to_csv(
        OUT / "v2_pipeline_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    queues.to_csv(
        OUT / f"v2_pipeline_daily_queue_k{args.primary_k}.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "v2_pipeline_components.npz",
        **component_arrays,
    )
    primary = metrics[metrics["k"] == args.primary_k]
    summary = {
        "model_version": "V2",
        "datasets": selected_blocks,
        "primary_k": args.primary_k,
        "q": q_from_alpha(args.primary_k, ALPHA),
        "cooldown_days": COOLDOWN_DAYS,
        "reused_components": bool(args.reuse_components),
        "labels_used_for_ranking": False,
        "primary_metrics": primary.to_dict(orient="records"),
        "outputs": {
            "metrics": str(OUT / "v2_pipeline_metrics.csv"),
            "queue": str(
                OUT
                / f"v2_pipeline_daily_queue_k{args.primary_k}.csv"
            ),
            "components": str(
                OUT / "v2_pipeline_components.npz"
            ),
        },
    }
    (OUT / "v2_pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
