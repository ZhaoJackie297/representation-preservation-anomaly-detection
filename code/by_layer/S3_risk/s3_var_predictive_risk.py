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


def build_evidence_vector(
    masked_q: np.ndarray,
    manifold_q: np.ndarray,
    channel_q: np.ndarray,
) -> np.ndarray:
    stacked = np.stack(
        [masked_q, manifold_q, channel_q], axis=-1
    ).astype(np.float64)
    stacked = np.maximum(stacked, 0.0)
    return np.log1p(stacked)


def gaussian_var_predictive_residual(
    evidence: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Fit per-user VAR(1) and return Mahalanobis predictive residuals."""
    users, time_steps, dimensions = evidence.shape
    residual_score = np.full(
        (users, time_steps), np.nan, dtype=np.float32
    )
    fitted_users = 0
    usable_transitions = 0

    for user_index in range(users):
        active = np.flatnonzero(active_mask[user_index])
        pair_positions = []
        for position in range(1, len(active)):
            previous = int(active[position - 1])
            current = int(active[position])
            pair_positions.append((previous, current))
        training_pairs = [
            (previous, current)
            for previous, current in pair_positions
            if current <= train_end
        ]
        if len(training_pairs) <= dimensions + 2:
            continue

        train_x = np.asarray(
            [
                np.concatenate(
                    [
                        [1.0],
                        evidence[user_index, previous],
                        [
                            np.log1p(
                                max(current - previous, 0)
                            )
                        ],
                    ]
                )
                for previous, current in training_pairs
            ],
            dtype=np.float64,
        )
        train_y = np.asarray(
            [
                evidence[user_index, current]
                for _, current in training_pairs
            ],
            dtype=np.float64,
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            train_x, train_y, rcond=None
        )
        train_prediction = train_x @ coefficients
        train_residual = train_y - train_prediction
        covariance = (
            train_residual.T @ train_residual
        ) / max(len(train_residual) - 1, 1)
        covariance[np.diag_indices_from(covariance)] += 1e-8
        inverse_covariance = np.linalg.pinv(covariance)

        for previous, current in pair_positions:
            features = np.concatenate(
                [
                    [1.0],
                    evidence[user_index, previous],
                    [
                        np.log1p(
                            max(current - previous, 0)
                        )
                    ],
                ]
            )
            prediction = features @ coefficients
            residual = (
                evidence[user_index, current] - prediction
            )
            score = float(
                residual.T @ inverse_covariance @ residual
            )
            residual_score[user_index, current] = max(
                score, 0.0
            )
        fitted_users += 1
        usable_transitions += len(pair_positions)

    return residual_score, {
        "fitted_users": fitted_users,
        "usable_transitions": usable_transitions,
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
    evidence = build_evidence_vector(
        masked_q, manifold_q, channel_q
    )
    predictive_raw, fit_summary = (
        gaussian_var_predictive_residual(
            evidence, data["active_mask"], train_end
        )
    )
    predictive_q = per_user_quantile_scores(
        predictive_raw, data["active_mask"], train_end
    )
    predictive_q[
        data["active_mask"] & ~np.isfinite(predictive_q)
    ] = 0.0
    signal = daily_percentile(
        predictive_q, data["active_mask"]
    ).astype(np.float32)

    original, augmented = fit_view_scores(
        data,
        config["feature_file"],
        predictive_q,
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
            "S3_var_predictive",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s3_var_{block_name}_metrics.csv",
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
    var_alert = build_alerts(
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
            var_alert,
            current_alert,
            data["active_mask"],
            start,
            end,
        )
        for split, (start, end) in ranges.items()
    }
    valid = active_mask & np.isfinite(predictive_q)
    correlation = float(
        np.corrcoef(
            predictive_q[valid], triple_q[valid]
        )[0, 1]
    )

    np.savez_compressed(
        OUT / f"s3_var_{block_name}_components.npz",
        evidence=evidence.astype(np.float32),
        triple_quantile=triple_q,
        predictive_raw=predictive_raw,
        predictive_quantile=predictive_q,
        predictive_signal=signal,
        predictive_aligned_iforest=auxiliary,
        current_dynamic=current_dynamic,
        current_alert_k10=current_alert,
        predictive_alert_k10=var_alert,
        active_mask=data["active_mask"],
    )
    summary = {
        "block": block_name,
        "method": "per-user Gaussian VAR(1) predictive residual",
        "model": (
            "log1p(evidence)_t = b + A*log1p(evidence)_t-1 "
            "+ calendar-gap term + epsilon_t"
        ),
        "risk": (
            "Mahalanobis residual under epsilon ~ N(0, Sigma)"
        ),
        "labels_used_in_risk": False,
        "fit": fit_summary,
        "primary_k": PRIMARY_K,
        "correlation_with_triple": correlation,
        "top10_overlap_with_current_risk": overlap,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s3_var_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
                "fit": fit_summary,
                "correlation_with_triple": correlation,
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
