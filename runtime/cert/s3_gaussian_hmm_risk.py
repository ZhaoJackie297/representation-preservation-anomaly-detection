from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
SEEDS = [20261111, 20261112, 20261113]
HMM_SEED = 20261111
N_STATES = 2
MAX_ITER = 30
TOL = 1e-5
VARIANCE_FLOOR = 1e-3
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


def build_evidence(
    masked_q: np.ndarray,
    manifold_q: np.ndarray,
    channel_q: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.stack(
        [masked_q, manifold_q, channel_q], axis=-1
    ).astype(np.float64)
    raw = np.log1p(np.maximum(raw, 0.0))
    normalized = np.full(raw.shape, np.nan, dtype=np.float64)
    for user_index in range(raw.shape[0]):
        train_mask = active_mask[
            user_index, : train_end + 1
        ]
        train = raw[user_index, : train_end + 1][train_mask]
        center = train.mean(axis=0)
        scale = train.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        normalized[user_index] = (raw[user_index] - center) / scale
    return raw, normalized


def active_segments(
    user_mask: np.ndarray,
) -> list[np.ndarray]:
    active = np.flatnonzero(user_mask)
    if len(active) == 0:
        return []
    split_points = np.flatnonzero(np.diff(active) > 1) + 1
    return [
        segment
        for segment in np.split(active, split_points)
        if len(segment)
    ]


def gaussian_log_prob(
    values: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * variance)
        + (values[:, None, :] - mean[None, :, :]) ** 2
        / variance[None, :, :],
        axis=-1,
    )


def forward_filter_segment(
    values: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    log_emission = gaussian_log_prob(
        values, mean, variance
    )
    emission = np.exp(
        log_emission - log_emission.max(axis=1, keepdims=True)
    )
    alpha = initial * emission[0]
    alpha /= max(alpha.sum(), 1e-300)
    filtered = [alpha]
    for time_index in range(1, len(values)):
        alpha = emission[time_index] * (alpha @ transition)
        alpha /= max(alpha.sum(), 1e-300)
        filtered.append(alpha)
    return np.asarray(filtered)


def expectation_step(
    sequences: list[np.ndarray],
    initial: np.ndarray,
    transition: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    initial_sum = np.zeros(N_STATES, dtype=np.float64)
    transition_sum = np.zeros(
        (N_STATES, N_STATES), dtype=np.float64
    )
    state_count = np.zeros(N_STATES, dtype=np.float64)
    weighted_values = np.zeros_like(mean)
    weighted_squares = np.zeros_like(mean)
    gammas = []
    xis = []
    total_log_likelihood = 0.0

    for values in sequences:
        log_emission = gaussian_log_prob(
            values, mean, variance
        )
        emission_shift = log_emission.max(
            axis=1, keepdims=True
        )
        emission = np.exp(
            log_emission - emission_shift
        )
        length = len(values)
        alpha = np.zeros((length, N_STATES), dtype=np.float64)
        beta = np.zeros((length, N_STATES), dtype=np.float64)
        scales = np.zeros(length, dtype=np.float64)

        alpha[0] = initial * emission[0]
        scales[0] = alpha[0].sum()
        alpha[0] /= max(scales[0], 1e-300)
        for time_index in range(1, length):
            alpha[time_index] = emission[time_index] * (
                alpha[time_index - 1] @ transition
            )
            scales[time_index] = alpha[time_index].sum()
            alpha[time_index] /= max(
                scales[time_index], 1e-300
            )

        beta[-1] = 1.0
        for time_index in range(length - 2, -1, -1):
            beta[time_index] = (
                transition
                @ (
                    emission[time_index + 1]
                    * beta[time_index + 1]
                )
            ) / max(scales[time_index + 1], 1e-300)

        gamma = alpha * beta
        gamma /= np.maximum(
            gamma.sum(axis=1, keepdims=True), 1e-300
        )
        xi = np.zeros(
            (max(length - 1, 0), N_STATES, N_STATES),
            dtype=np.float64,
        )
        for time_index in range(length - 1):
            numerator = (
                alpha[time_index, :, None]
                * transition
                * emission[time_index + 1][None, :]
                * beta[time_index + 1][None, :]
            )
            xi[time_index] = numerator / max(
                numerator.sum(), 1e-300
            )

        initial_sum += gamma[0]
        transition_sum += xi.sum(axis=0)
        state_count += gamma.sum(axis=0)
        weighted_values += gamma.T @ values
        weighted_squares += gamma.T @ (values**2)
        gammas.append(gamma)
        xis.append(xi)
        total_log_likelihood += float(
            np.log(np.maximum(scales, 1e-300)).sum()
            + emission_shift[:, 0].sum()
        )

    return (
        initial_sum,
        transition_sum,
        gammas,
        xis,
        total_log_likelihood,
        state_count,
        weighted_values,
        weighted_squares,
    )


def fit_gaussian_hmm(
    sequences: list[np.ndarray],
) -> dict[str, np.ndarray | float | int]:
    flat = np.concatenate(sequences, axis=0)
    mixture = GaussianMixture(
        n_components=N_STATES,
        covariance_type="diag",
        reg_covar=VARIANCE_FLOOR,
        random_state=HMM_SEED,
        max_iter=200,
    ).fit(flat)
    order = np.argsort(mixture.means_.sum(axis=1))
    mean = mixture.means_[order].copy()
    variance = mixture.covariances_[order].copy()
    variance = np.maximum(variance, VARIANCE_FLOOR)
    initial = np.asarray([0.8, 0.2], dtype=np.float64)
    transition = np.asarray(
        [[0.95, 0.05], [0.15, 0.85]], dtype=np.float64
    )

    previous_likelihood = -np.inf
    history = []
    for iteration in range(MAX_ITER):
        (
            initial_sum,
            transition_sum,
            _,
            _,
            likelihood,
            state_count,
            weighted_values,
            weighted_squares,
        ) = expectation_step(
            sequences,
            initial,
            transition,
            mean,
            variance,
        )

        initial = initial_sum / max(initial_sum.sum(), 1e-300)
        transition = transition_sum / np.maximum(
            transition_sum.sum(axis=1, keepdims=True), 1e-300
        )
        mean = weighted_values / np.maximum(
            state_count[:, None], 1e-300
        )
        variance = (
            weighted_squares / np.maximum(state_count[:, None], 1e-300)
            - mean**2
        )
        variance = np.maximum(variance, VARIANCE_FLOOR)

        order = np.argsort(mean.sum(axis=1))
        mean = mean[order]
        variance = variance[order]
        initial = initial[order]
        transition = transition[np.ix_(order, order)]

        history.append(likelihood)
        if (
            np.isfinite(previous_likelihood)
            and likelihood - previous_likelihood
            < TOL * max(abs(previous_likelihood), 1.0)
        ):
            break
        previous_likelihood = likelihood

    return {
        "initial": initial,
        "transition": transition,
        "means": mean,
        "variances": variance,
        "log_likelihood": float(likelihood),
        "iterations": len(history),
        "likelihood_history": np.asarray(history),
    }


def hmm_filtered_risk(
    normalized: np.ndarray,
    active_mask: np.ndarray,
    model: dict,
) -> np.ndarray:
    risk = np.full(active_mask.shape, np.nan, dtype=np.float32)
    for user_index in range(active_mask.shape[0]):
        for segment in active_segments(active_mask[user_index]):
            values = normalized[user_index, segment]
            filtered = forward_filter_segment(
                values,
                model["initial"],
                model["transition"],
                model["means"],
                model["variances"],
            )
            risk[user_index, segment] = filtered[:, 1]
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
    evidence_raw, normalized = build_evidence(
        masked_q,
        manifold_q,
        channel_q,
        data["active_mask"],
        train_end,
    )

    training_sequences = []
    for user_index in range(normalized.shape[0]):
        for segment in active_segments(
            data["active_mask"][user_index, : train_end + 1]
        ):
            if len(segment) >= 2:
                training_sequences.append(
                    normalized[user_index, segment]
                )
    model = fit_gaussian_hmm(training_sequences)
    hmm_risk = hmm_filtered_risk(
        normalized, data["active_mask"], model
    )
    hmm_risk[
        data["active_mask"] & ~np.isfinite(hmm_risk)
    ] = 0.0
    signal = daily_percentile(
        hmm_risk, data["active_mask"]
    ).astype(np.float32)

    original, augmented = fit_view_scores(
        data,
        config["feature_file"],
        hmm_risk,
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
            "S3_gaussian_hmm",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s3_hmm_{block_name}_metrics.csv",
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
    hmm_alert = build_alerts(
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
            hmm_alert,
            current_alert,
            data["active_mask"],
            start,
            end,
        )
        for split, (start, end) in ranges.items()
    }
    valid = data["active_mask"] & np.isfinite(hmm_risk)
    correlation = float(
        np.corrcoef(hmm_risk[valid], triple_q[valid])[0, 1]
    )

    np.savez_compressed(
        OUT / f"s3_hmm_{block_name}_components.npz",
        evidence_raw=evidence_raw.astype(np.float32),
        normalized_evidence=normalized.astype(np.float32),
        triple_quantile=triple_q,
        hmm_risk=hmm_risk,
        hmm_signal=signal,
        hmm_aligned_iforest=auxiliary,
        current_dynamic=current_dynamic,
        current_alert_k10=current_alert,
        hmm_alert_k10=hmm_alert,
        active_mask=data["active_mask"],
        initial=model["initial"],
        transition=model["transition"],
        means=model["means"],
        variances=model["variances"],
        likelihood_history=model["likelihood_history"],
    )
    summary = {
        "block": block_name,
        "method": "two-state diagonal Gaussian HMM",
        "states": ["stable", "elevated"],
        "model": (
            "Z_t Markov; X_t | Z_t=k ~ N(mean_k, diag(var_k)); "
            "risk_t=P(Z_t=elevated | X_1:t)"
        ),
        "training": "Baum-Welch EM on normal training sequences",
        "labels_used_in_risk": False,
        "seed": HMM_SEED,
        "iterations": model["iterations"],
        "log_likelihood": model["log_likelihood"],
        "transition": model["transition"].tolist(),
        "means": model["means"].tolist(),
        "variances": model["variances"].tolist(),
        "training_sequences": len(training_sequences),
        "primary_k": PRIMARY_K,
        "correlation_with_triple": correlation,
        "top10_overlap_with_current_risk": overlap,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s3_hmm_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
                "iterations": model["iterations"],
                "log_likelihood": model["log_likelihood"],
                "transition": model["transition"].tolist(),
                "means": model["means"].tolist(),
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
