from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from e19_user_manifold import (  # noqa: E402
    dataset_config,
    evaluate_method,
    load_components,
)
from quota_relationship_sim import daily_percentile  # noqa: E402
from risk_topk_r42 import compute_risk  # noqa: E402


CHANNEL_INDICES = {
    "logon": [1, 2, 3, 4, 5, 6, 7],
    "device": [8, 9],
    "file": [10, 11, 12, 13],
    "email": [14, 15, 16, 17, 18, 19],
    "http": [20, 21, 22, 23],
}


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return (numerator + 0.5) / (denominator + 1.0)


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.01, 0.99)
    return np.log(clipped) - np.log1p(-clipped)


def relation_matrix(
    values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    raw = np.maximum(
        values.astype(np.float64)
        * scales.astype(np.float64)
        + means.astype(np.float64),
        0.0,
    )
    channel_volumes = []
    for name, indices in CHANNEL_INDICES.items():
        selected = raw[:, :, indices]
        if name == "logon":
            volume = np.log1p(
                selected[:, :, 4] / 60.0
            )
            volume += np.log1p(selected[:, :, [0, 1, 2, 3]]).sum(
                axis=-1
            )
        elif name == "email":
            volume = np.log1p(
                selected[:, :, 4] * 1024.0
            )
            volume += np.log1p(selected[:, :, [0, 1, 2, 3, 5]]).sum(
                axis=-1
            )
        else:
            volume = np.log1p(selected).sum(axis=-1)
        channel_volumes.append(volume)
    volumes = np.stack(channel_volumes, axis=-1)
    probabilities = (volumes + 0.5) / (
        volumes.sum(axis=-1, keepdims=True) + 2.5
    )
    log_probabilities = np.log(probabilities)
    proportions = log_probabilities - log_probabilities.mean(
        axis=-1, keepdims=True
    )

    logon = raw[:, :, 1]
    email = raw[:, :, 14]
    http = raw[:, :, 20]
    file_copy = raw[:, :, 10]
    fractions = np.stack(
        [
            safe_ratio(raw[:, :, 6], logon),
            safe_ratio(raw[:, :, 7], logon),
            safe_ratio(raw[:, :, 16], email),
            safe_ratio(raw[:, :, 17], email),
            safe_ratio(raw[:, :, 19], email),
            safe_ratio(raw[:, :, 22], http),
            safe_ratio(raw[:, :, 21], http),
            safe_ratio(raw[:, :, 23], http),
            safe_ratio(raw[:, :, 12], file_copy),
            safe_ratio(raw[:, :, 11], file_copy),
        ],
        axis=-1,
    )
    return np.concatenate(
        [proportions, logit(fractions)], axis=-1
    ).astype(np.float32)


def relation_deviation(
    relation: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, dict]:
    users, time_steps, dimensions = relation.shape
    train_mask = (
        active_mask[:, : train_end + 1]
        & np.isfinite(relation[:, : train_end + 1]).all(axis=-1)
    )
    global_rows = relation[:, : train_end + 1][train_mask]
    global_center = np.median(global_rows, axis=0)
    centers = np.empty((users, dimensions), dtype=np.float64)
    residuals = []
    for user in range(users):
        training = np.flatnonzero(train_mask[user])
        if len(training) >= 4:
            centers[user] = np.median(
                relation[user, training], axis=0
            )
            residuals.append(
                relation[user, training] - centers[user]
            )
        else:
            centers[user] = global_center
    residual_rows = np.concatenate(residuals, axis=0)
    scale = np.sqrt(
        np.mean(np.square(residual_rows), axis=0)
    )
    scale = np.where(
        np.isfinite(scale) & (scale >= 1e-8),
        scale,
        1.0,
    )
    distances = np.full(
        active_mask.shape, np.nan, dtype=np.float32
    )
    for user in range(users):
        active = np.flatnonzero(active_mask[user])
        if len(active) == 0:
            continue
        standardized = (
            relation[user, active] - centers[user]
        ) / scale
        valid = np.isfinite(standardized).all(axis=1)
        distances[user, active[valid]] = np.sqrt(
            np.mean(np.square(standardized[valid]), axis=1)
        )
    return distances, {
        "dimensions": dimensions,
        "fallback_users": int(
            sum(
                train_mask[user].sum() < 4
                for user in range(users)
            )
        ),
        "mean_scale": float(np.mean(scale)),
    }


def build_quad_signal(
    data: dict,
    masked_q: np.ndarray,
    manifold_q: np.ndarray,
    channel_q: np.ndarray,
    relation_q: np.ndarray,
) -> np.ndarray:
    scores = (masked_q + manifold_q + channel_q + relation_q) / 4.0
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        scores,
        data["active_mask"],
    )
    return daily_percentile(
        dynamic, data["active_mask"]
    ).astype(np.float32)


def run_dataset(dataset_name: str) -> None:
    config = dataset_config(dataset_name)
    data = config["data"]
    dataset = np.load(
        config["dataset_file"], allow_pickle=False
    )
    values = dataset["values"].astype(np.float32)
    means = dataset["mean"]
    scales = dataset["std"]
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    _, masked_q, _, channel_q = load_components(dataset_name)
    dual_q = np.load(
        OUT / f"e19_{dataset_name}_dual_mean_manifold_quantile.npy"
    )
    auxiliary_all = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    auxiliary = auxiliary_all[config["auxiliary_key"]]

    relation = relation_matrix(values, means, scales)
    relation_raw, diagnostics = relation_deviation(
        relation, active_mask, train_end
    )
    relation_q = per_user_quantile_scores(
        relation_raw, active_mask, train_end
    )
    stage_b_signal = build_quad_signal(
        data, masked_q, dual_q, channel_q, relation_q
    )
    stage_a_signal = np.load(
        OUT / f"e19_{dataset_name}_A3_dual_mean_signal.npy"
    )
    results = {
        "A3_dual_mean": evaluate_method(
            data,
            stage_a_signal,
            auxiliary,
            f"E19_{dataset_name}_A3",
            external=config["external"],
        ),
        "B1_relation_equal": evaluate_method(
            data,
            stage_b_signal,
            auxiliary,
            f"E19_{dataset_name}_B1",
            external=config["external"],
        ),
    }
    np.save(
        OUT / f"e19_{dataset_name}_relation_raw.npy",
        relation_raw,
    )
    np.save(
        OUT / f"e19_{dataset_name}_relation_quantile.npy",
        relation_q,
    )
    np.save(
        OUT / f"e19_{dataset_name}_B1_relation_equal_signal.npy",
        stage_b_signal,
    )
    summary = {
        "dataset": dataset_name,
        "protocol": {
            "fusion": (
                "masked + dual_manifold + channel_top2 + relation"
                " equal-weight daily quantiles"
            ),
            "anomaly_labels_used_for_training": False,
            "weights_searched": False,
        },
        "relation_diagnostics": diagnostics,
        "results": results,
    }
    path = OUT / f"e19_{dataset_name}_relation_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["r42", "r52", "all"], default="all"
    )
    args = parser.parse_args()
    if args.dataset in {"r42", "all"}:
        run_dataset("r42")
    if args.dataset in {"r52", "all"}:
        run_dataset("r52")


if __name__ == "__main__":
    main()
