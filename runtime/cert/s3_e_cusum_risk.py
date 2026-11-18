from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
SEEDS = [20261111, 20261112, 20261113]
PRIMARY_K = 10
KAPPA = 0.5

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


def conformal_p_values(
    scores: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    p_values = np.full(scores.shape, np.nan, dtype=np.float64)
    for user_index in range(scores.shape[0]):
        calibration = np.sort(
            scores[
                user_index, : train_end + 1
            ][
                active_mask[user_index, : train_end + 1]
                & np.isfinite(scores[user_index, : train_end + 1])
            ]
        )
        if len(calibration) == 0:
            continue
        for time_index in np.flatnonzero(active_mask[user_index]):
            value = float(scores[user_index, time_index])
            first_at_least = int(
                np.searchsorted(calibration, value, side="left")
            )
            at_least_count = len(calibration) - first_at_least
            p_values[user_index, time_index] = (
                at_least_count + 1
            ) / (len(calibration) + 1)
    return p_values


def e_cusum_evidence(
    p_values: np.ndarray,
    active_mask: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    evidence = np.full(p_values.shape, np.nan, dtype=np.float64)
    e_values = np.full(p_values.shape, np.nan, dtype=np.float64)
    for user_index in range(p_values.shape[0]):
        state = 0.0
        for time_index in range(p_values.shape[1]):
            if not active_mask[user_index, time_index]:
                state = 0.0
                continue
            p_value = min(
                max(float(p_values[user_index, time_index]), 1e-12),
                1.0,
            )
            e_value = kappa * p_value ** (kappa - 1.0)
            state = max(
                0.0,
                state + math.log(max(e_value, 1e-12)),
            )
            e_values[user_index, time_index] = e_value
            evidence[user_index, time_index] = state
    return evidence, e_values


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
        overlaps.append(len(first & second) / denominator)
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
    train_end = int(dataset["train_end_index"])

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

    p_values = conformal_p_values(
        triple_q, data["active_mask"], train_end
    )
    evidence, e_values = e_cusum_evidence(
        p_values, data["active_mask"], KAPPA
    )
    evidence_float = evidence.astype(np.float32)
    signal = daily_percentile(
        evidence_float, data["active_mask"]
    ).astype(np.float32)

    original, augmented = fit_view_scores(
        data,
        config["feature_file"],
        evidence_float,
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
            "S3_e_cusum_risk",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s3_e_cusum_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    current_dynamic = compute_risk(
        data["users"],
        data["dates"],
        triple_q,
        data["active_mask"],
    )
    current_signal = daily_percentile(
        current_dynamic, data["active_mask"]
    ).astype(np.float32)
    current_alert = build_alerts(
        current_signal, data["active_mask"], PRIMARY_K, 7
    )
    evidence_alert = build_alerts(
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
            evidence_alert,
            current_alert,
            data["active_mask"],
            start,
            end,
        )
        for split, (start, end) in ranges.items()
    }
    valid = data["active_mask"] & np.isfinite(evidence_float)
    correlation = float(
        np.corrcoef(
            evidence_float[valid], current_dynamic[valid]
        )[0, 1]
    )

    np.savez_compressed(
        OUT / f"s3_e_cusum_{block_name}_components.npz",
        triple_quantile=triple_q,
        p_value=p_values.astype(np.float32),
        e_value=e_values.astype(np.float32),
        evidence=evidence_float,
        evidence_signal=signal,
        evidence_aligned_iforest=auxiliary,
        current_dynamic=current_dynamic,
        current_alert_k10=current_alert,
        evidence_alert_k10=evidence_alert,
        active_mask=data["active_mask"],
    )
    summary = {
        "block": block_name,
        "method": "conformal p-value -> valid p-to-e calibrator -> log e-CUSUM",
        "kappa": KAPPA,
        "p_value": "(1 + number of calibration scores >= score) / (n + 1)",
        "e_value": "kappa * p^(kappa - 1)",
        "risk_state": "max(0, previous_state + log(e_value))",
        "labels_used_in_risk": False,
        "primary_k": PRIMARY_K,
        "correlation_with_current_risk": correlation,
        "top10_overlap_with_current_risk": overlap,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s3_e_cusum_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
                "correlation_with_current_risk": correlation,
                "top10_overlap_with_current_risk": overlap,
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
