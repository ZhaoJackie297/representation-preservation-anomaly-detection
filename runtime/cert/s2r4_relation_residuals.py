from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
SEEDS = [20261111, 20261112, 20261113]
TRANSITION_NEIGHBORS = 5
MAX_ACTIVE_GAP = 7
RIDGE_ALPHA = 1.0

CHANNELS = {
    "logon": list(range(1, 8)),
    "device": [8, 9],
    "file": list(range(10, 14)),
    "email": list(range(14, 20)),
    "http": list(range(20, 24)),
}

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
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


def temporal_transition_residual(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    users, time_steps, dimensions = values.shape
    delta_scale = []
    for user in range(users):
        active = np.flatnonzero(active_mask[user, : train_end + 1])
        for left, right in zip(active[:-1], active[1:]):
            if right - left <= MAX_ACTIVE_GAP:
                delta_scale.append(values[user, right] - values[user, left])
    global_scale = np.sqrt(
        np.mean(np.square(np.stack(delta_scale)), axis=0)
    )
    global_scale = np.where(
        np.isfinite(global_scale) & (global_scale >= 1e-8),
        global_scale,
        1.0,
    )
    output = np.full(
        (users, time_steps), np.nan, dtype=np.float32
    )
    for user in range(users):
        active = np.flatnonzero(active_mask[user, : train_end + 1])
        training_pairs = [
            (left, right)
            for left, right in zip(active[:-1], active[1:])
            if right - left <= MAX_ACTIVE_GAP
        ]
        if len(training_pairs) < TRANSITION_NEIGHBORS + 1:
            continue
        training_deltas = np.stack(
            [
                (
                    values[user, right]
                    - values[user, left]
                )
                / global_scale
                for left, right in training_pairs
            ]
        )
        pairwise = np.linalg.norm(
            training_deltas[:, None, :]
            - training_deltas[None, :, :],
            axis=-1,
        )
        np.fill_diagonal(pairwise, np.inf)
        train_scores = np.sort(pairwise, axis=1)[
            :, :TRANSITION_NEIGHBORS
        ].mean(axis=1)
        train_positions = np.array(
            [right for _, right in training_pairs]
        )
        output[user, train_positions] = train_scores

        all_active = np.flatnonzero(active_mask[user])
        for time_index in all_active:
            if time_index <= 0:
                continue
            previous_candidates = all_active[
                all_active < time_index
            ]
            if len(previous_candidates) == 0:
                continue
            previous = int(previous_candidates[-1])
            if time_index - previous > MAX_ACTIVE_GAP:
                continue
            delta = (
                values[user, time_index]
                - values[user, previous]
            ) / global_scale
            distances = np.linalg.norm(
                training_deltas - delta, axis=1
            )
            output[user, time_index] = float(
                np.sort(distances)[:TRANSITION_NEIGHBORS].mean()
            )
    return output


def channel_summary(values: np.ndarray) -> np.ndarray:
    summaries = []
    for indices in CHANNELS.values():
        selected = np.nan_to_num(values[:, :, indices])
        if len(indices) == 1:
            summaries.append(selected[:, :, 0])
        else:
            summaries.append(selected.mean(axis=-1))
    return np.stack(summaries, axis=-1)


def cross_channel_relation_residual(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    summaries = channel_summary(values)
    calendar = np.nan_to_num(values[:, :, 24:27])
    train_mask = active_mask[:, : train_end + 1]
    train_indices = np.nonzero(train_mask)
    train_rows = summaries[:, : train_end + 1][train_indices]
    train_calendar = calendar[:, : train_end + 1][train_indices]
    residuals = []
    for channel_index in range(summaries.shape[-1]):
        other = np.delete(
            summaries, channel_index, axis=-1
        )
        predictors_train = np.concatenate(
            [
                other[:, : train_end + 1][train_indices],
                train_calendar,
            ],
            axis=1,
        )
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(predictors_train, train_rows[:, channel_index])
        active_indices = np.nonzero(active_mask)
        predictors_all = np.concatenate(
            [
                other[active_indices],
                calendar[active_indices],
            ],
            axis=1,
        )
        residual = (
            summaries[active_indices][:, channel_index]
            - model.predict(predictors_all)
        )
        channel_residual = np.full(
            active_mask.shape, np.nan, dtype=np.float64
        )
        channel_residual[active_indices] = residual
        calibration_mask = train_mask & np.isfinite(
            channel_residual[:, : train_end + 1]
        )
        calibration_values = channel_residual[
            :, : train_end + 1
        ][calibration_mask]
        center = np.median(calibration_values)
        scale = 1.4826 * np.median(
            np.abs(calibration_values - center)
        )
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(calibration_values)
        scale = max(float(scale), 1e-8)
        residuals.append(
            np.abs(channel_residual - center) / scale
        )
    stacked = np.stack(residuals, axis=0)
    ordered = np.sort(stacked, axis=0)
    return ordered[-2:].mean(axis=0).astype(np.float32)


def build_signal(
    data: dict,
    masked_q: np.ndarray,
    manifold_q: np.ndarray,
    channel_q: np.ndarray,
    residual_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = (
        masked_q + manifold_q + channel_q + residual_q
    ) / 4.0
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        scores,
        data["active_mask"],
    )
    return (
        daily_percentile(
            dynamic, data["active_mask"]
        ).astype(np.float32),
        dynamic,
    )


def run_block(block_name: str) -> None:
    config = BLOCKS[block_name]
    data = load_r52() if config["family"] == "r52" else load_r42()
    dataset = np.load(config["sequence"], allow_pickle=False)
    values = np.nan_to_num(
        dataset["values"].astype(np.float32)
    )
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    masked_q = np.load(config["masked"])
    channel_store = np.load(config["channel"][0], allow_pickle=False)
    channel_q = channel_store[config["channel"][1]].astype(np.float32)
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
    print("Building temporal-transition residual", flush=True)
    temporal_raw = temporal_transition_residual(
        values, active_mask, train_end
    )
    temporal_q = per_user_quantile_scores(
        temporal_raw, active_mask, train_end
    )
    print("Building cross-channel relation residual", flush=True)
    relation_raw = cross_channel_relation_residual(
        values, active_mask, train_end
    )
    relation_q = per_user_quantile_scores(
        relation_raw, active_mask, train_end
    )
    combined_q = (
        temporal_q + relation_q
    ) / 2.0
    methods = {
        "S2R4_temporal": temporal_q,
        "S2R4_relation": relation_q,
        "S2R4_combined": combined_q,
    }
    metric_frames = []
    saved = {
        "temporal_raw": temporal_raw,
        "temporal_quantile": temporal_q,
        "relation_raw": relation_raw,
        "relation_quantile": relation_q,
        "combined_quantile": combined_q,
    }
    for method, residual_q in methods.items():
        signal, dynamic = build_signal(
            data,
            masked_q,
            manifold_q,
            channel_q,
            residual_q,
        )
        original, augmented = fit_view_scores(
            data,
            config["feature_file"],
            dynamic,
            data["personal"],
            20261109,
        )
        auxiliary = np.maximum(original, augmented)
        metric_frames.append(
            evaluate_block(
                block_name,
                data,
                signal,
                auxiliary,
                method,
                external=bool(config["external"]),
            )
        )
        saved[f"{method}_signal"] = signal
        saved[f"{method}_aligned_iforest"] = auxiliary
    metrics = add_utilities(
        pd.concat(metric_frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / f"s2r4_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / f"s2r4_{block_name}_components.npz",
        **saved,
    )
    summary = {
        "block": block_name,
        "transition_neighbors": TRANSITION_NEIGHBORS,
        "ridge_alpha": RIDGE_ALPHA,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s2r4_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
                "k10": metrics[
                    metrics["k"] == 10
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
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
