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
    components = np.stack(
        [masked_q, manifold_q, channel_q], axis=0
    )
    mean_q = components.mean(axis=0).astype(np.float32)
    median_q = np.median(components, axis=0).astype(np.float32)

    current_dynamic = compute_risk(
        data["users"],
        data["dates"],
        mean_q,
        data["active_mask"],
    )
    consensus_dynamic = compute_risk(
        data["users"],
        data["dates"],
        median_q,
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
            "S3_consensus_severity",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s3_consensus_{block_name}_metrics.csv",
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
    valid = active_mask & np.isfinite(mean_q) & np.isfinite(
        median_q
    )
    correlation = float(
        np.corrcoef(median_q[valid], mean_q[valid])[0, 1]
    )

    np.savez_compressed(
        OUT / f"s3_consensus_{block_name}_components.npz",
        masked_quantile=masked_q,
        manifold_quantile=manifold_q,
        channel_quantile=channel_q,
        mean_quantile=mean_q,
        median_quantile=median_q,
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
        "method": "three-view median consensus severity",
        "components": [
            "masked representation difference",
            "A2 user manifold",
            "channel top2 evidence",
        ],
        "median_rule": (
            "median(masked_quantile, manifold_quantile, "
            "channel_quantile)"
        ),
        "labels_used_in_risk": False,
        "primary_k": PRIMARY_K,
        "correlation_median_with_mean": correlation,
        "top10_overlap_with_current_risk": overlap,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s3_consensus_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
                "correlation_median_with_mean": correlation,
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
