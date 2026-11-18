from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
from production_protocol_r42 import build_alerts  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from s3_dfm_v12 import (  # noqa: E402
    OUT,
    prepare_r42,
)
from v2_m_pipeline import add_utilities, evaluate_block  # noqa: E402


SEEDS = [20261111, 20261112, 20261113]
PRIMARY_K = 10
N_FACTORS = 2


def static_factor_posteriors(
    values: np.ndarray,
    observed: np.ndarray,
    mu: np.ndarray,
    loading: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    users, time_steps, _ = values.shape
    posterior_means = np.full(
        (users, time_steps, N_FACTORS),
        np.nan,
        dtype=np.float64,
    )
    for time_index in range(time_steps):
        for user_index in range(users):
            current = observed[user_index, time_index]
            if not current.any():
                continue
            loading_observed = loading[current]
            noise_observed = np.diag(noise[current])
            observation_cov = (
                loading_observed @ loading_observed.T
                + noise_observed
            )
            gain = (
                loading_observed.T
                @ np.linalg.pinv(observation_cov)
            )
            residual = (
                values[user_index, time_index, current]
                - mu[current]
            )
            posterior_means[user_index, time_index] = (
                gain @ residual
            )
    return posterior_means


def fit_ar1_from_static_scores(
    scores: np.ndarray,
    observed_any: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phi = np.zeros(N_FACTORS)
    for factor_index in range(N_FACTORS):
        numerator = 0.0
        denominator = 0.0
        for user_index in range(scores.shape[0]):
            times = np.flatnonzero(
                observed_any[user_index, : scores.shape[1] - 1]
                & observed_any[user_index, 1:]
                & fit_mask[user_index, 1:]
            )
            if len(times) == 0:
                continue
            current = scores[
                user_index, times + 1, factor_index
            ]
            previous = scores[
                user_index, times, factor_index
            ]
            numerator += float(np.dot(previous, current))
            denominator += float(np.dot(previous, previous))
        phi[factor_index] = float(
            np.clip(
                numerator / max(denominator, 1e-12),
                -0.95,
                0.95,
            )
        )
    innovation_variance = np.maximum(1.0 - phi**2, 1e-4)
    return phi, innovation_variance


def filter_two_factor(
    values: np.ndarray,
    observed: np.ndarray,
    mu: np.ndarray,
    loading: np.ndarray,
    noise: np.ndarray,
    phi: np.ndarray,
    innovation_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users, time_steps, _ = values.shape
    mean = np.zeros((users, N_FACTORS), dtype=np.float64)
    covariance = np.tile(
        np.eye(N_FACTORS), (users, 1, 1)
    )
    mean_output = np.full(
        values.shape[:2] + (N_FACTORS,), np.nan
    )
    norm_output = np.full(values.shape[:2], np.nan)
    factor_percentile = np.full(values.shape[:2], np.nan)
    transition = np.diag(phi)
    innovation = np.diag(innovation_variance)

    for time_index in range(time_steps):
        predicted_mean = mean @ transition.T
        predicted_covariance = (
            transition @ covariance @ transition.T
            + innovation[None, :, :]
        )
        for user_index in range(users):
            current = observed[user_index, time_index]
            if not current.any():
                mean_output[user_index, time_index] = (
                    predicted_mean[user_index]
                )
                norm_output[user_index, time_index] = float(
                    np.linalg.norm(predicted_mean[user_index])
                )
                mean[user_index] = predicted_mean[user_index]
                covariance[user_index] = (
                    predicted_covariance[user_index]
                )
                continue
            loading_observed = loading[current]
            noise_observed = np.diag(noise[current])
            observation_cov = (
                loading_observed
                @ predicted_covariance[user_index]
                @ loading_observed.T
                + noise_observed
            )
            gain = (
                predicted_covariance[user_index]
                @ loading_observed.T
                @ np.linalg.pinv(observation_cov)
            )
            residual = (
                values[user_index, time_index, current]
                - mu[current]
                - loading_observed
                @ predicted_mean[user_index]
            )
            filtered_mean = (
                predicted_mean[user_index] + gain @ residual
            )
            filtered_covariance = (
                predicted_covariance[user_index]
                - gain
                @ loading_observed
                @ predicted_covariance[user_index]
            )
            mean[user_index] = filtered_mean
            covariance[user_index] = filtered_covariance
            mean_output[user_index, time_index] = filtered_mean
            norm_output[user_index, time_index] = float(
                np.linalg.norm(filtered_mean)
            )

    # Convert causal factor norm to a daily percentile for queueing.
    factor_percentile = daily_percentile(
        norm_output.astype(np.float32),
        observed.any(axis=-1),
    ).astype(np.float32)
    return (
        mean_output,
        norm_output.astype(np.float32),
        factor_percentile,
    )


def main() -> None:
    prepared = prepare_r42()
    values = prepared["transformed"]
    observed = prepared["observed"]
    fit_mask = prepared["fit_mask"]
    complete_fit = fit_mask & observed.all(axis=-1)
    fit_rows = values[complete_fit]

    factor_model = FactorAnalysis(
        n_components=N_FACTORS,
        max_iter=5000,
        tol=1e-6,
        random_state=20261111,
    ).fit(fit_rows)
    mu = fit_rows.mean(axis=0)
    loading = np.asarray(factor_model.components_.T)
    noise = np.maximum(
        np.asarray(factor_model.noise_variance_), 1e-6
    )

    static_scores = static_factor_posteriors(
        values,
        observed,
        mu,
        loading,
        noise,
    )
    observed_any = observed.any(axis=-1)
    phi, innovation_variance = fit_ar1_from_static_scores(
        static_scores,
        observed_any,
        fit_mask,
    )
    filtered_mean, factor_norm, signal = filter_two_factor(
        values,
        observed,
        mu,
        loading,
        noise,
        phi,
        innovation_variance,
    )

    data = load_r42()
    original, augmented = fit_view_scores(
        data,
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        factor_norm,
        data["personal"],
        20261109,
    )
    auxiliary = np.maximum(original, augmented)
    metrics = add_utilities(
        evaluate_block(
            "r42_original",
            data,
            signal,
            auxiliary,
            "S3_DFM2_factor_norm",
            external=False,
        )
    )
    metrics_file = OUT / "s3_dfm2_r42_original_metrics.csv"
    metrics.to_csv(
        metrics_file,
        index=False,
        encoding="utf-8",
    )

    masked_q = prepared["evidence"][:, :, 0]
    manifold_q = prepared["evidence"][:, :, 1]
    channel_q = prepared["evidence"][:, :, 2]
    triple_q = (masked_q + manifold_q + channel_q) / 3.0
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
    factor_alert = build_alerts(
        signal, data["active_mask"], PRIMARY_K, 7
    )
    overlaps = {}
    for split, start, end in (
        ("validation", 159, 302),
        ("test", 303, 500),
    ):
        values_overlap = []
        for time_index in range(start, end + 1):
            first = set(
                np.flatnonzero(
                    factor_alert[:, time_index]
                ).tolist()
            )
            second = set(
                np.flatnonzero(
                    current_alert[:, time_index]
                ).tolist()
            )
            denominator = max(
                min(
                    PRIMARY_K,
                    int(data["active_mask"][:, time_index].sum()),
                ),
                1,
            )
            values_overlap.append(
                len(first & second) / denominator
            )
        overlaps[split] = float(np.mean(values_overlap))

    np.savez_compressed(
        OUT / "s3_dfm2_r42_original_components.npz",
        loading=loading.astype(np.float32),
        noise=noise.astype(np.float32),
        mu=mu.astype(np.float32),
        phi=phi.astype(np.float32),
        innovation_variance=innovation_variance.astype(
            np.float32
        ),
        filtered_factor_mean=filtered_mean.astype(np.float32),
        factor_norm=factor_norm,
        factor_signal=signal,
        factor_aligned_iforest=auxiliary,
        current_signal=current_signal,
        factor_alert_k10=factor_alert,
        current_alert_k10=current_alert,
        active_mask=data["active_mask"],
    )
    summary = {
        "model": "exploratory two-factor AR(1) DFM2",
        "status": "pilot only; no v1.2 confirmation eligibility",
        "n_factors_selected_by_bic": N_FACTORS,
        "factor_analysis_loadings": loading.tolist(),
        "factor_noise": noise.tolist(),
        "mu": mu.tolist(),
        "ar1_phi": phi.tolist(),
        "innovation_variance": innovation_variance.tolist(),
        "risk": "L2 norm of causal posterior factor mean",
        "labels_used_in_model_fit": False,
        "top10_overlap_with_current_risk": overlaps,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / "s3_dfm2_r42_original_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
