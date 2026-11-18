from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
SEEDS = [20261111, 20261112, 20261113]
PRIMARY_K = 10

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
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
        "plain": OUT / "e17_r42_M0_plain.npy",
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
        "plain": OUT / "e19_r42_retrained_plain.npy",
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
        "plain": OUT / "e17_r52_M0_plain.npy",
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


def representation_curvature(
    representations: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    curvature = np.full(
        active_mask.shape, np.nan, dtype=np.float32
    )
    for user_index in range(active_mask.shape[0]):
        active = np.flatnonzero(active_mask[user_index])
        for position in range(2, len(active)):
            previous_previous = int(active[position - 2])
            previous = int(active[position - 1])
            current = int(active[position])
            first_velocity = (
                representations[user_index, previous]
                - representations[
                    user_index, previous_previous
                ]
            )
            second_velocity = (
                representations[user_index, current]
                - representations[user_index, previous]
            )
            first_norm = np.linalg.norm(first_velocity)
            second_norm = np.linalg.norm(second_velocity)
            denominator = max(
                float(first_norm * second_norm), 1e-8
            )
            cosine = float(
                np.clip(
                    np.dot(first_velocity, second_velocity)
                    / denominator,
                    -1.0,
                    1.0,
                )
            )
            curvature[user_index, current] = (
                0.5 * (1.0 - cosine)
            )
    return curvature


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
    plain = np.load(config["plain"]).astype(np.float32)

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

    curvature_raw = representation_curvature(
        plain, data["active_mask"]
    )
    curvature_q = per_user_quantile_scores(
        curvature_raw, data["active_mask"], train_end
    )
    consensus_q = np.sqrt(
        np.maximum(triple_q, 0.0)
        * np.maximum(curvature_q, 0.0)
    ).astype(np.float32)

    current_dynamic = compute_risk(
        data["users"],
        data["dates"],
        triple_q,
        data["active_mask"],
    )
    consensus_dynamic = compute_risk(
        data["users"],
        data["dates"],
        consensus_q,
        data["active_mask"],
    )
    signal = daily_percentile(
        consensus_dynamic, data["active_mask"]
    ).astype(np.float32)

    original, augmented = fit_view_scores(
        data,
        config["feature_file"],
        consensus_dynamic,
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
            "S3_trajectory_consensus",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s3_trajectory_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    current_signal = daily_percentile(
        current_dynamic, data["active_mask"]
    ).astype(np.float32)
    current_alert = build_alerts(
        current_signal, data["active_mask"], PRIMARY_K, 7
    )
    consensus_alert = build_alerts(
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
            consensus_alert,
            current_alert,
            data["active_mask"],
            start,
            end,
        )
        for split, (start, end) in ranges.items()
    }
    valid = (
        data["active_mask"]
        & np.isfinite(curvature_q)
        & np.isfinite(triple_q)
    )
    correlation = float(
        np.corrcoef(
            curvature_q[valid], triple_q[valid]
        )[0, 1]
    )

    np.savez_compressed(
        OUT / f"s3_trajectory_{block_name}_components.npz",
        triple_quantile=triple_q,
        curvature_raw=curvature_raw,
        curvature_quantile=curvature_q,
        consensus_quantile=consensus_q,
        current_dynamic=current_dynamic,
        consensus_dynamic=consensus_dynamic,
        consensus_signal=signal,
        consensus_aligned_iforest=auxiliary,
        current_alert_k10=current_alert,
        consensus_alert_k10=consensus_alert,
        active_mask=data["active_mask"],
    )
    summary = {
        "block": block_name,
        "method": (
            "TS2Vec velocity-direction curvature "
            "with geometric-mean consensus"
        ),
        "curvature": (
            "0.5 * (1 - cosine(v_t, v_t-1)), "
            "v_t = representation_t - representation_t-1"
        ),
        "consensus": "sqrt(triple_quantile * curvature_quantile)",
        "labels_used_in_risk": False,
        "primary_k": PRIMARY_K,
        "correlation_curvature_with_triple": correlation,
        "top10_overlap_with_current_risk": overlap,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s3_trajectory_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
                "correlation_curvature_with_triple": correlation,
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
