from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
SEEDS = [20261111, 20261112, 20261113]
PRIMARY_K = 10
WINDOW_DAYS = 7
DECAY = 0.65
HIGH_SCORE_THRESHOLD = 3.0
SEVERITY_WEIGHT = 0.7
MEMORY_WEIGHT = 0.2
FREQUENCY_WEIGHT = 0.1

sys.path.insert(0, str(ROOT))
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
from production_protocol_r42 import build_alerts  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42,
    load_r52,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from v2_m_pipeline import add_utilities, evaluate_block  # noqa: E402


BLOCKS = {
    "r42_original": {
        "family": "r42",
        "masked": OUT / "e17_r42_M0_masked_quantile.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r42_original_channel_top2_quantile",
        ),
        "manifold_prefix": "r42",
        "feature_file": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        ),
        "external": False,
    },
    "r42_retrained": {
        "family": "r42",
        "masked": OUT / "e19_r42_retrained_masked_quantile.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r42_retrained_channel_top2_quantile",
        ),
        "manifold_prefix": "r42_retrained",
        "feature_file": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        ),
        "external": False,
    },
    "r52_external": {
        "family": "r52",
        "masked": OUT / "e17_r52_M0_masked_quantile.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r52_external_channel_top2_quantile",
        ),
        "manifold_prefix": "r52",
        "feature_file": (
            ROOT / "processed_r52" / "r5.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        ),
        "external": True,
    },
}


def compute_calendar_risk(
    scores: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Keep risk state across inactive days and decay by calendar gap."""
    risk = np.full(scores.shape, np.nan, dtype=np.float32)
    for user_index in range(scores.shape[0]):
        memory = 0.0
        last_active = None
        high_days: deque[tuple[int, int]] = deque()

        for time_index in range(scores.shape[1]):
            if not active_mask[user_index, time_index]:
                continue
            score = float(scores[user_index, time_index])
            if last_active is not None:
                gap = time_index - last_active
                memory *= DECAY**gap
            memory = max(score, memory)

            high_days.append(
                (time_index, int(score > HIGH_SCORE_THRESHOLD))
            )
            while (
                high_days
                and time_index - high_days[0][0] >= WINDOW_DAYS
            ):
                high_days.popleft()
            frequency = (
                sum(flag for _, flag in high_days) / WINDOW_DAYS
            )
            risk[user_index, time_index] = (
                SEVERITY_WEIGHT * score
                + MEMORY_WEIGHT * memory
                + FREQUENCY_WEIGHT * frequency
            )
            last_active = time_index
    return risk


def topk_overlap(
    first_alert: np.ndarray,
    second_alert: np.ndarray,
    active_mask: np.ndarray,
    start_index: int,
    end_index: int,
) -> float:
    overlaps = []
    for time_index in range(start_index, end_index + 1):
        first = set(
            np.flatnonzero(first_alert[:, time_index]).tolist()
        )
        second = set(
            np.flatnonzero(second_alert[:, time_index]).tolist()
        )
        denominator = max(
            min(PRIMARY_K, int(active_mask[:, time_index].sum())),
            1,
        )
        overlaps.append(
            len(first & second) / denominator
        )
    return float(np.mean(overlaps)) if overlaps else 0.0


def run_block(block_name: str) -> None:
    config = BLOCKS[block_name]
    data = (
        load_r52()
        if config["family"] == "r52"
        else load_r42()
    )
    dataset = np.load(config["sequence"], allow_pickle=False)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)

    masked_q = np.load(config["masked"]).astype(np.float32)
    channel_store = np.load(
        config["channel"][0], allow_pickle=False
    )
    channel_q = channel_store[config["channel"][1]].astype(
        np.float32
    )
    manifold_q = np.mean(
        np.stack(
            [
                np.load(
                    OUT
                    / (
                        f"e19_{config['manifold_prefix']}_learned_"
                        f"manifold_quantile_{seed}.npy"
                    )
                )
                for seed in SEEDS
            ],
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)
    triple_q = (masked_q + manifold_q + channel_q) / 3.0

    baseline_dynamic = compute_risk(
        data["users"],
        data["dates"],
        triple_q,
        data["active_mask"],
    )
    calendar_dynamic = compute_calendar_risk(
        triple_q, data["active_mask"]
    )
    signal = daily_percentile(
        calendar_dynamic, data["active_mask"]
    ).astype(np.float32)

    original, augmented = fit_view_scores(
        data,
        config["feature_file"],
        calendar_dynamic,
        data["personal"],
        20261109,
    )
    auxiliary = np.maximum(original, augmented)
    metrics = add_utilities(
        evaluate_block(
            block_name,
            data,
            signal,
            auxiliary,
            "S3_calendar_risk",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s3_calendar_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    baseline_alert = build_alerts(
        daily_percentile(
            baseline_dynamic, data["active_mask"]
        ).astype(np.float32),
        data["active_mask"],
        PRIMARY_K,
        7,
    )
    calendar_alert = build_alerts(
        signal, data["active_mask"], PRIMARY_K, 7
    )
    ranges = (
        {"external_test": (data["test_start"], len(data["dates"]) - 1)}
        if config["external"]
        else {
            "validation": (data["train_end"] + 1, data["val_end"]),
            "test": (data["val_end"] + 1, len(data["dates"]) - 1),
        }
    )
    overlap = {
        split: topk_overlap(
            calendar_alert,
            baseline_alert,
            data["active_mask"],
            start,
            end,
        )
        for split, (start, end) in ranges.items()
    }

    active = data["active_mask"] & np.isfinite(
        calendar_dynamic
    ) & np.isfinite(baseline_dynamic)
    risk_delta = np.zeros(calendar_dynamic.shape, dtype=np.float32)
    risk_delta[active] = (
        calendar_dynamic[active] - baseline_dynamic[active]
    )
    gaps = np.full(calendar_dynamic.shape, 0, dtype=np.int16)
    for user_index in range(calendar_dynamic.shape[0]):
        last = None
        for time_index in np.flatnonzero(
            data["active_mask"][user_index]
        ):
            if last is not None:
                gaps[user_index, time_index] = time_index - last
            last = time_index
    returning = active & (gaps >= 2)

    np.savez_compressed(
        OUT / f"s3_calendar_{block_name}_components.npz",
        triple_quantile=triple_q,
        baseline_dynamic=baseline_dynamic,
        calendar_dynamic=calendar_dynamic,
        calendar_signal=signal,
        calendar_aligned_iforest=auxiliary,
        baseline_alert_k10=baseline_alert,
        calendar_alert_k10=calendar_alert,
        active_mask=data["active_mask"],
    )
    summary = {
        "block": block_name,
        "change": "calendar_gap_decay_and_frequency_persistence",
        "window_days": WINDOW_DAYS,
        "decay": DECAY,
        "high_score_threshold": HIGH_SCORE_THRESHOLD,
        "weights": {
            "severity": SEVERITY_WEIGHT,
            "memory": MEMORY_WEIGHT,
            "frequency": FREQUENCY_WEIGHT,
        },
        "primary_k": PRIMARY_K,
        "top10_overlap_with_current_risk": overlap,
        "returning_active_days": int(returning.sum()),
        "mean_risk_change_on_return": float(
            risk_delta[returning].mean()
        )
        if returning.any()
        else 0.0,
        "median_risk_change_on_return": float(
            np.median(risk_delta[returning])
        )
        if returning.any()
        else 0.0,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s3_calendar_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
                "top10_overlap_with_current_risk": overlap,
                "returning_active_days": int(returning.sum()),
                "mean_risk_change_on_return": float(
                    risk_delta[returning].mean()
                )
                if returning.any()
                else 0.0,
                "k10": metrics[
                    metrics["k"] == PRIMARY_K
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices=sorted(BLOCKS),
        default="r42_original",
    )
    args = parser.parse_args()
    run_block(args.block)
