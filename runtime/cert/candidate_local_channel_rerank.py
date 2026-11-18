from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
K = 10
Q_LIMIT = 4
COOLDOWN_DAYS = 7
CANDIDATE_POOL = 50
WINDOW_DAYS = 30
SEED = 20261109

CHANNELS = {
    "logon": list(range(1, 8)),
    "device": [8, 9],
    "file": list(range(10, 14)),
    "email": list(range(14, 20)),
    "http": list(range(20, 24)),
}

sys.path.insert(0, str(ROOT))
from application_utility_analysis import PROFILES  # noqa: E402
from production_protocol_r42 import label_matrix  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
)
from ts2vec_dual_score_experiment import (  # noqa: E402
    ENCODE_ARGS,
    load_model,
)


BLOCKS = {
    "r42_original": {
        "family": "r42",
        "model": ROOT / "processed_r42" / "r4.2_ts2vec_model.pth",
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
        "external": False,
    },
    "r52_external": {
        "family": "r52",
        "model": ROOT / "processed_r52" / "r5.2_ts2vec_model.pth",
        "external": True,
    },
}


def candidate_indices_by_day(
    signal: np.ndarray,
    auxiliary: np.ndarray,
    active_mask: np.ndarray,
) -> dict[int, list[int]]:
    candidates = {}
    for time_index in range(signal.shape[1]):
        active = np.flatnonzero(active_mask[:, time_index])
        if len(active) == 0:
            candidates[time_index] = []
            continue
        ts_order = active[
            np.lexsort((active, -signal[active, time_index]))
        ]
        aux_order = active[
            np.lexsort((active, -auxiliary[active, time_index]))
        ]
        selected = set(ts_order[:CANDIDATE_POOL]) | set(
            aux_order[:CANDIDATE_POOL]
        )
        candidates[time_index] = sorted(
            int(user) for user in selected
        )
    return candidates


def local_channel_scores(
    model,
    values: np.ndarray,
    candidates: dict[int, list[int]],
) -> tuple[dict[str, dict[tuple[int, int], float]], np.ndarray]:
    keys = []
    windows = []
    for time_index, users in candidates.items():
        for user in users:
            start = max(0, time_index - WINDOW_DAYS + 1)
            window = values[user, start : time_index + 1]
            if len(window) < WINDOW_DAYS:
                pad = np.full(
                    (
                        WINDOW_DAYS - len(window),
                        values.shape[-1],
                    ),
                    np.nan,
                    dtype=np.float32,
                )
                window = np.concatenate([pad, window], axis=0)
            keys.append((user, time_index))
            windows.append(window)
    window_array = np.stack(windows).astype(np.float32)
    print(
        f"  local candidate windows: {len(window_array)}",
        flush=True,
    )
    plain = np.asarray(
        model.encode(window_array, **ENCODE_ARGS),
        dtype=np.float32,
    )[:, -1]
    scores = {}
    for channel, indices in CHANNELS.items():
        print(f"  local channel ablation: {channel}", flush=True)
        masked = window_array.copy()
        masked[:, -1, indices] = 0.0
        representation = np.asarray(
            model.encode(masked, **ENCODE_ARGS),
            dtype=np.float32,
        )[:, -1]
        raw = np.abs(representation - plain).sum(axis=1)
        scores[channel] = {
            key: float(value) for key, value in zip(keys, raw)
        }
        del masked, representation
    return scores, window_array


def local_personal_quantiles(
    raw_scores: dict[str, dict[tuple[int, int], float]],
    train_end: int,
) -> tuple[
    dict[str, dict[tuple[int, int], float]],
    dict[str, float],
]:
    result = {channel: {} for channel in CHANNELS}
    fallback = {}
    for channel in CHANNELS:
        training_by_user = defaultdict(list)
        training_pool = []
        for (user, time_index), value in raw_scores[channel].items():
            if time_index <= train_end:
                training_by_user[user].append(value)
                training_pool.append(value)
        global_sorted = np.sort(np.asarray(training_pool))
        fallback_value = (
            float(np.median(global_sorted))
            if len(global_sorted)
            else 0.0
        )
        fallback[channel] = fallback_value
        for (user, time_index), value in raw_scores[channel].items():
            reference = np.sort(np.asarray(training_by_user[user]))
            if len(reference) < 5:
                reference = global_sorted
            if len(reference) == 0:
                result[channel][(user, time_index)] = 0.0
                continue
            rank = int(
                np.searchsorted(reference, value, side="right")
            )
            tail = (len(reference) - rank + 1) / (
                len(reference) + 1
            )
            result[channel][(user, time_index)] = max(
                0.0, -math.log(tail)
            )
    return result, fallback


def build_local_rerank_signal(
    base_signal: np.ndarray,
    local_quantiles: dict[str, dict[tuple[int, int], float]],
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], str]]:
    local_top2 = np.full(base_signal.shape, np.nan, dtype=np.float32)
    dominant = {}
    for time_index in range(base_signal.shape[1]):
        active = np.flatnonzero(active_mask[:, time_index])
        keys = [
            (int(user), time_index)
            for user in active
            if (int(user), time_index)
            in local_quantiles["logon"]
        ]
        if not keys:
            continue
        values = np.asarray(
            [
                sorted(
                    local_quantiles[channel][key]
                    for channel in CHANNELS
                )[-2:]
                for key in keys
            ],
            dtype=np.float32,
        )
        top2 = values.mean(axis=1)
        order = np.argsort(top2)
        pool_percentile = np.empty(len(keys), dtype=np.float32)
        pool_percentile[order] = np.linspace(
            0.0, 1.0, len(keys), endpoint=True
        )
        for key, value, percentile_value in zip(
            keys, top2, pool_percentile
        ):
            user, _ = key
            channel_values = {
                channel: local_quantiles[channel][key]
                for channel in CHANNELS
            }
            dominant[key] = max(
                channel_values, key=channel_values.get
            )
            local_top2[user, time_index] = value
        available = np.asarray([key[0] for key in keys])
        rerank = (
            0.5 * base_signal[available, time_index]
            + 0.5 * pool_percentile
        )
        local_top2[available, time_index] = rerank
    return local_top2, base_signal, dominant


def select_alerts(
    base_signal: np.ndarray,
    rerank_signal: np.ndarray,
    auxiliary: np.ndarray,
    active_mask: np.ndarray,
    *,
    rerank_all: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    alerts = np.zeros(base_signal.shape, dtype=np.uint8)
    last_alert = np.full(base_signal.shape[0], -10_000, dtype=np.int32)
    source_counts = defaultdict(int)
    for time_index in range(base_signal.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= COOLDOWN_DAYS
        )
        candidates = np.flatnonzero(available)
        if len(candidates) == 0:
            continue
        base_order = candidates[
            np.lexsort(
                (candidates, -base_signal[candidates, time_index])
            )
        ]
        rerank_pool = candidates[
            np.isfinite(rerank_signal[candidates, time_index])
        ]
        if len(rerank_pool):
            rerank_order = rerank_pool[
                np.lexsort(
                    (
                        rerank_pool,
                        -rerank_signal[rerank_pool, time_index],
                    )
                )
            ]
        else:
            rerank_order = base_order
        selected = [int(user) for user in rerank_order[: K - Q_LIMIT]]
        selected_set = set(selected)
        source_counts["ts_main"] += len(selected)
        aux_order = candidates[
            np.lexsort(
                (candidates, -auxiliary[candidates, time_index])
            )
        ]
        if rerank_all:
            aux_candidates = set(int(user) for user in rerank_pool)
            aux_order = np.asarray(
                [
                    int(user)
                    for user in rerank_order
                    if int(user) in aux_candidates
                ],
                dtype=np.int64,
            )
        ts_top_k = set(int(user) for user in base_order[:K])
        for user in aux_order:
            user = int(user)
            if user in selected_set or user in ts_top_k:
                continue
            selected.append(user)
            selected_set.add(user)
            source_counts["iforest_rescue"] += 1
            if len(selected) >= K:
                break
        if len(selected) < K:
            for user in base_order:
                user = int(user)
                if user in selected_set:
                    continue
                selected.append(user)
                selected_set.add(user)
                source_counts["ts_fallback"] += 1
                if len(selected) >= K:
                    break
        selected_array = np.asarray(selected[:K], dtype=np.int64)
        alerts[selected_array, time_index] = 1
        last_alert[selected_array] = time_index
    return alerts, dict(source_counts)


def evaluate_alerts(
    data: dict,
    alerts: np.ndarray,
    method: str,
    *,
    external: bool,
) -> pd.DataFrame:
    result = evaluate_alert_system(
        data, alerts, method, K, external=external
    )
    splits = ["external_test"] if external else ["validation", "test"]
    rows = []
    for split in splits:
        rows.append(
            {
                "method": method,
                "split": split,
                **compact_metric(
                    result, None if external else split
                ),
            }
        )
    return pd.DataFrame(rows)


def add_utilities(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for profile, weights in PROFILES.items():
        frame[profile] = sum(
            frame[column] * weight
            for column, weight in weights.items()
        )
    return frame


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dual = np.load(
        OUT / "ts2vec_dual_score_arrays.npz",
        allow_pickle=False,
    )
    channel = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    aligned = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    r42 = load_r42()
    r52 = load_r52()
    families = {"r42": r42, "r52": r52}
    datasets = {
        "r42": np.load(
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
            allow_pickle=False,
        ),
        "r52": np.load(
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
            allow_pickle=False,
        ),
    }
    metric_frames = []
    saved = {}
    summaries = []
    for block_name, config in BLOCKS.items():
        print(f"\nRunning {block_name}", flush=True)
        data = families[config["family"]]
        dataset = datasets[config["family"]]
        base_signal = channel[
            f"{block_name}_triple_mean_iforest_signal"
        ].astype(np.float32)
        auxiliary = aligned[
            f"{block_name}_v2_aligned_dual_view_iforest"
        ].astype(np.float32)
        candidates = candidate_indices_by_day(
            base_signal, auxiliary, data["active_mask"]
        )
        model = load_model(config["model"], dataset["values"].shape[-1])
        raw_local, windows = local_channel_scores(
            model, dataset["values"].astype(np.float32), candidates
        )
        train_end = (
            int(dataset["train_end_index"])
            if "train_end_index" in dataset.files
            else data["test_start"] - 1
        )
        local_q, _ = local_personal_quantiles(
            raw_local, train_end
        )
        rerank_signal, _, dominant = build_local_rerank_signal(
            base_signal, local_q, data["active_mask"]
        )
        base_alerts, base_sources = select_alerts(
            base_signal,
            base_signal,
            auxiliary,
            data["active_mask"],
            rerank_all=False,
        )
        ts_rerank_alerts, ts_rerank_sources = select_alerts(
            base_signal,
            rerank_signal,
            auxiliary,
            data["active_mask"],
            rerank_all=False,
        )
        all_rerank_alerts, all_rerank_sources = select_alerts(
            base_signal,
            rerank_signal,
            auxiliary,
            data["active_mask"],
            rerank_all=True,
        )
        for method, alerts, sources in (
            ("v2_base", base_alerts, base_sources),
            ("v2_local_ts_rerank", ts_rerank_alerts, ts_rerank_sources),
            ("v2_local_all_rerank", all_rerank_alerts, all_rerank_sources),
        ):
            frame = evaluate_alerts(
                data,
                alerts,
                method,
                external=bool(config["external"]),
            )
            frame.insert(0, "data_block", block_name)
            metric_frames.append(frame)
            saved[f"{block_name}_{method}_alerts"] = alerts
        saved[f"{block_name}_local_rerank_signal"] = rerank_signal
        for channel_name in CHANNELS:
            score = np.full(base_signal.shape, np.nan, dtype=np.float32)
            for key, value in local_q[channel_name].items():
                score[key] = value
            saved[f"{block_name}_local_{channel_name}_quantile"] = score
        summaries.append(
            {
                "data_block": block_name,
                "candidate_windows": len(windows),
                "base_sources": base_sources,
                "ts_rerank_sources": ts_rerank_sources,
                "all_rerank_sources": all_rerank_sources,
                "dominant_local_channel_counts": pd.Series(
                    list(dominant.values())
                ).value_counts().to_dict(),
            }
        )
        del model, windows
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = add_utilities(
        pd.concat(metric_frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "candidate_local_channel_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "candidate_local_channel_arrays.npz",
        **saved,
    )
    summary = {
        "primary_k": K,
        "q": Q_LIMIT,
        "candidate_pool_size": CANDIDATE_POOL,
        "local_window_days": WINDOW_DAYS,
        "rerank_weight": 0.5,
        "labels_used": False,
        "blocks": summaries,
        "outputs": {
            "metrics": str(
                OUT / "candidate_local_channel_metrics.csv"
            ),
            "arrays": str(
                OUT / "candidate_local_channel_arrays.npz"
            ),
        },
    }
    (OUT / "candidate_local_channel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nK=10 metrics", flush=True)
    print(
        metrics[
            [
                "data_block",
                "split",
                "method",
                "incident_hits",
                "incident_total",
                "event_recall_3d",
                "scenario_macro_recall_3d",
                "precision_at_k",
                "balanced",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
