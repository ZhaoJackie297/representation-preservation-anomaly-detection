from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numba import njit
from scipy.optimize import minimize
from scipy.special import ndtri
from sklearn.decomposition import FactorAnalysis


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "innovation_sim"
OUT = DATA / "s3_dfm_v12"
OUT.mkdir(parents=True, exist_ok=True)

DATASET = ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
USER_DAY = ROOT / "processed_r42" / "r4.2_user_day_features.csv"
MASKED = DATA / "e17_r42_M0_masked_quantile.npy"
MANIFOLD_FILES = [
    DATA / f"e19_r42_learned_manifold_quantile_{seed}.npy"
    for seed in (20261111, 20261112, 20261113)
]
CHANNEL_FILE = DATA / "channel_evidence_arrays.npz"
CHANNEL_KEY = "r42_original_channel_top2_quantile"

VALIDATION_START = 159
VALIDATION_END = 302
FIT_START = 159
FIT_END = 230
H_SECOND_HALF_START = 231
H_SECOND_HALF_END = 302
DEVELOPMENT_TEST_START = 303
DEVELOPMENT_TEST_END = 500


def load_evidence() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = np.load(DATASET, allow_pickle=False)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    masked = np.load(MASKED)
    manifold = np.mean(
        np.stack(
            [np.load(path) for path in MANIFOLD_FILES],
            axis=0,
        ),
        axis=0,
    )
    with np.load(CHANNEL_FILE, allow_pickle=False) as store:
        channel = store[CHANNEL_KEY]
    evidence = np.stack(
        [masked, manifold, channel], axis=-1
    ).astype(np.float32)
    return evidence, active, dataset


def empirical_normal_transform(
    evidence: np.ndarray,
    observed_mask: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    transformed = np.full(
        evidence.shape, np.nan, dtype=np.float64
    )
    for channel_index in range(evidence.shape[-1]):
        calibration = np.sort(
            evidence[:, :, channel_index][
                fit_mask
                & observed_mask[:, :, channel_index]
            ].astype(np.float64)
        )
        if len(calibration) < 10:
            raise RuntimeError(
                f"Channel {channel_index} has too few fit values."
            )
        active = observed_mask[:, :, channel_index]
        values = evidence[:, :, channel_index][active]
        left = np.searchsorted(calibration, values, side="left")
        right = np.searchsorted(calibration, values, side="right")
        ecdf = (
            left
            + 0.5 * (right - left)
            + 0.5
        ) / (len(calibration) + 1.0)
        transformed[:, :, channel_index][active] = ndtri(
            np.clip(ecdf, 1e-8, 1.0 - 1e-8)
        )
    return transformed


def unpack_parameters(
    vector: np.ndarray,
    *,
    equal_structure: bool,
    fix_phi: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    def positive(value: float) -> float:
        return float(np.exp(np.clip(value, -20.0, 20.0)))

    mu = vector[:3]
    if equal_structure:
        loading = np.full(3, positive(vector[3]))
        noise = np.full(3, positive(vector[4]))
        phi_index = 5
    else:
        loading = np.exp(np.clip(vector[3:6], -20.0, 20.0))
        noise = np.exp(np.clip(vector[6:9], -20.0, 20.0))
        phi_index = 9
    phi = (
        0.0
        if fix_phi
        else float(np.clip(np.tanh(vector[phi_index]), -0.995, 0.995))
    )
    return mu, loading, noise, phi


def pack_parameters(
    mu: np.ndarray,
    loading: np.ndarray,
    noise: np.ndarray,
    phi: float,
    *,
    equal_structure: bool,
    fix_phi: bool,
) -> np.ndarray:
    safe_loading = np.clip(loading, 1e-5, None)
    safe_noise = np.clip(noise, 1e-5, None)
    values = [*mu]
    if equal_structure:
        values.extend(
            [
                np.log(float(np.mean(safe_loading))),
                np.log(float(np.mean(safe_noise))),
            ]
        )
    else:
        values.extend(np.log(safe_loading))
        values.extend(np.log(safe_noise))
    if not fix_phi:
        values.append(np.arctanh(np.clip(phi, -0.99, 0.99)))
    return np.asarray(values, dtype=np.float64)


def optimizer_start_combinations(
    n_starts: int,
) -> list[tuple[float, float, float]]:
    phi_grid = [0.0, 0.2, 0.4, 0.6, 0.8]
    primary = [(phi, 1.0, 1.0) for phi in phi_grid]
    extra = [
        (phi, loading_scale, noise_scale)
        for phi in phi_grid
        for loading_scale in (0.7, 1.0)
        for noise_scale in (0.7, 1.3)
        if (loading_scale, noise_scale) != (1.0, 1.0)
    ]
    return (primary + extra)[:n_starts]


def filter_dfm(
    values: np.ndarray,
    observed: np.ndarray,
    parameters: tuple[
        np.ndarray, np.ndarray, np.ndarray, float
    ],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    mu, loading, noise, phi = parameters
    users, time_steps, _ = values.shape
    centered = np.where(
        observed,
        values - mu[None, None, :],
        0.0,
    )
    mean = np.zeros(users, dtype=np.float64)
    variance = np.ones(users, dtype=np.float64)
    filtered_mean = np.full(
        (users, time_steps), np.nan, dtype=np.float64
    )
    filtered_variance = np.full(
        (users, time_steps), np.nan, dtype=np.float64
    )
    predicted_mean = np.full(
        (users, time_steps), np.nan, dtype=np.float64
    )
    predicted_variance = np.full(
        (users, time_steps), np.nan, dtype=np.float64
    )
    log_likelihood_by_time = np.zeros(
        time_steps, dtype=np.float64
    )

    for time_index in range(time_steps):
        mean_prediction = phi * mean
        variance_prediction = (
            phi**2 * variance + (1.0 - phi**2)
        )
        current_observed = observed[:, time_index, :]
        observation_count = current_observed.sum(axis=1)
        residual = np.where(
            current_observed,
            centered[:, time_index, :]
            - loading[None, :] * mean_prediction[:, None],
            0.0,
        )
        inverse_noise = 1.0 / noise
        a_term = (
            current_observed
            * loading[None, :] ** 2
            * inverse_noise[None, :]
        ).sum(axis=1)
        b_term = (
            current_observed
            * loading[None, :]
            * inverse_noise[None, :]
            * residual
        ).sum(axis=1)
        denominator = 1.0 + variance_prediction * a_term
        quadratic = (
            current_observed
            * residual**2
            * inverse_noise[None, :]
        ).sum(axis=1)
        quadratic -= (
            variance_prediction * b_term**2
        ) / denominator
        log_determinant = (
            current_observed
            * np.log(noise)[None, :]
        ).sum(axis=1) + np.log(denominator)
        has_observation = observation_count > 0
        ll = np.zeros(users, dtype=np.float64)
        ll[has_observation] = -0.5 * (
            observation_count[has_observation]
            * np.log(2.0 * np.pi)
            + log_determinant[has_observation]
            + quadratic[has_observation]
        )
        log_likelihood_by_time[time_index] = ll.sum()

        updated_variance = 1.0 / (
            1.0 / variance_prediction + a_term
        )
        updated_mean = updated_variance * (
            mean_prediction / variance_prediction + b_term
        )
        predicted_mean[:, time_index] = mean_prediction
        predicted_variance[:, time_index] = variance_prediction
        filtered_mean[:, time_index] = mean
        filtered_variance[:, time_index] = updated_variance
        mean = updated_mean
        variance = updated_variance

    return (
        filtered_mean,
        filtered_variance,
        predicted_mean,
        predicted_variance,
        log_likelihood_by_time,
    )


@njit(cache=True)
def _dfm_negative_loglik_numba(
    values: np.ndarray,
    observed: np.ndarray,
    mu: np.ndarray,
    loading: np.ndarray,
    noise: np.ndarray,
    phi: float,
) -> float:
    users = values.shape[0]
    time_steps = values.shape[1]
    dimensions = values.shape[2]
    mean = np.zeros(users)
    variance = np.ones(users)
    total = 0.0
    for time_index in range(time_steps):
        for user_index in range(users):
            mean_prediction = phi * mean[user_index]
            variance_prediction = (
                phi * phi * variance[user_index]
                + 1.0 - phi * phi
            )
            a_term = 0.0
            b_term = 0.0
            quadratic = 0.0
            log_determinant = 0.0
            observation_count = 0
            for channel_index in range(dimensions):
                if not observed[user_index, time_index, channel_index]:
                    continue
                observation_count += 1
                residual = (
                    values[user_index, time_index, channel_index]
                    - mu[channel_index]
                    - loading[channel_index] * mean_prediction
                )
                inverse_noise = 1.0 / noise[channel_index]
                a_term += (
                    loading[channel_index]
                    * loading[channel_index]
                    * inverse_noise
                )
                b_term += (
                    loading[channel_index]
                    * inverse_noise
                    * residual
                )
                quadratic += residual * residual * inverse_noise
                log_determinant += np.log(
                    noise[channel_index]
                )
            if observation_count == 0:
                mean[user_index] = mean_prediction
                variance[user_index] = variance_prediction
                continue
            denominator = 1.0 + variance_prediction * a_term
            quadratic -= (
                variance_prediction * b_term * b_term
            ) / denominator
            log_determinant += np.log(denominator)
            total -= 0.5 * (
                observation_count * 5.756462732485115
                + log_determinant
                + quadratic
            )
            updated_variance = 1.0 / (
                1.0 / variance_prediction + a_term
            )
            updated_mean = updated_variance * (
                mean_prediction / variance_prediction + b_term
            )
            mean[user_index] = updated_mean
            variance[user_index] = updated_variance
    return -total


def fit_dfm_numba(
    values: np.ndarray,
    observed: np.ndarray,
    fit_mask: np.ndarray,
    *,
    equal_structure: bool,
    fix_phi: bool,
    n_starts: int,
    random_seed: int = 20261111,
) -> dict:
    complete_rows = values[
        fit_mask & observed.all(axis=-1)
    ]
    if len(complete_rows) < 50:
        raise RuntimeError("Too few complete observations to fit.")
    initializer = FactorAnalysis(
        n_components=1,
        max_iter=1000,
        random_state=random_seed,
    ).fit(complete_rows)
    loading = np.asarray(
        initializer.components_[0], dtype=np.float64
    )
    if loading[0] < 0:
        loading = -loading
    if equal_structure:
        loading = np.full(
            3, float(np.mean(np.abs(loading)))
        )
    noise = np.maximum(
        np.asarray(
            initializer.noise_variance_,
            dtype=np.float64,
        ),
        0.05,
    )
    if equal_structure:
        noise = np.full(3, float(np.mean(noise)))
    mu = complete_rows.mean(axis=0)
    combinations = optimizer_start_combinations(n_starts)
    starts = [
        pack_parameters(
            mu,
            loading * loading_scale,
            noise * noise_scale,
            phi,
            equal_structure=equal_structure,
            fix_phi=fix_phi,
        )
        for phi, loading_scale, noise_scale in combinations
    ]

    def objective(vector: np.ndarray) -> float:
        parameters = unpack_parameters(
            vector,
            equal_structure=equal_structure,
            fix_phi=fix_phi,
        )
        return _dfm_negative_loglik_numba(
            values,
            observed,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
        )

    best = None
    runs = []
    for start_index, start in enumerate(starts):
        try:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                options={
                    "maxiter": 250,
                    "ftol": 1e-9,
                    "gtol": 1e-5,
                    "maxls": 30,
                },
            )
            runs.append(
                {
                    "start": start_index,
                    "success": bool(result.success),
                    "fun": float(result.fun),
                    "nit": int(result.nit),
                }
            )
            if best is None or result.fun < best.fun:
                best = result
        except Exception as error:
            runs.append(
                {
                    "start": start_index,
                    "success": False,
                    "error": str(error),
                }
            )
    if best is None:
        raise RuntimeError("Numba DFM fit failed.")
    parameters = unpack_parameters(
        best.x,
        equal_structure=equal_structure,
        fix_phi=fix_phi,
    )
    return {
        "unconstrained": best.x,
        "mu": parameters[0],
        "loading": parameters[1],
        "noise": parameters[2],
        "phi": parameters[3],
        "negative_log_likelihood": float(best.fun),
        "success": bool(best.success),
        "message": str(best.message),
        "runs": runs,
    }


def fit_dfm(
    values: np.ndarray,
    observed: np.ndarray,
    fit_mask: np.ndarray,
    *,
    equal_structure: bool,
    fix_phi: bool,
    n_starts: int,
    random_seed: int = 20261111,
) -> dict:
    complete_rows = values[
        fit_mask & observed.all(axis=-1)
    ]
    if len(complete_rows) < 50:
        raise RuntimeError("Too few complete observations to fit.")
    initializer = FactorAnalysis(
        n_components=1,
        max_iter=1000,
        random_state=random_seed,
    ).fit(complete_rows)
    loading = np.asarray(
        initializer.components_[0], dtype=np.float64
    )
    if loading[0] < 0:
        loading = -loading
    if equal_structure:
        loading = np.full(
            3, float(np.mean(np.abs(loading)))
        )
    noise = np.maximum(
        np.asarray(
            initializer.noise_variance_,
            dtype=np.float64,
        ),
        0.05,
    )
    if equal_structure:
        noise = np.full(3, float(np.mean(noise)))
    mu = complete_rows.mean(axis=0)

    start_vectors = []
    combinations = optimizer_start_combinations(n_starts)
    for phi, loading_scale, noise_scale in combinations:
        start_vectors.append(
            pack_parameters(
                mu,
                loading * loading_scale,
                noise * noise_scale,
                phi,
                equal_structure=equal_structure,
                fix_phi=fix_phi,
            )
        )

    def objective(vector: np.ndarray) -> float:
        params = unpack_parameters(
            vector,
            equal_structure=equal_structure,
            fix_phi=fix_phi,
        )
        *_, ll_by_time = filter_dfm(
            values, observed, params
        )
        return -float(ll_by_time.sum())

    best_result = None
    runs = []
    for start in start_vectors:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={
                "maxiter": 400,
                "ftol": 1e-10,
                "gtol": 1e-6,
                "maxls": 40,
            },
        )
        runs.append(
            {
                "success": bool(result.success),
                "fun": float(result.fun),
                "nit": int(result.nit),
                "message": str(result.message),
            }
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    if best_result is None:
        raise RuntimeError("DFM optimization did not run.")
    parameters = unpack_parameters(
        best_result.x,
        equal_structure=equal_structure,
        fix_phi=fix_phi,
    )
    return {
        "unconstrained": best_result.x,
        "mu": parameters[0],
        "loading": parameters[1],
        "noise": parameters[2],
        "phi": parameters[3],
        "negative_log_likelihood": float(best_result.fun),
        "success": bool(best_result.success),
        "message": str(best_result.message),
        "runs": runs,
    }


def unpack_parameters_torch(
    vector: torch.Tensor,
    *,
    equal_structure: bool,
    fix_phi: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    mu = vector[:3]
    if equal_structure:
        loading = torch.full(
            (3,),
            torch.exp(torch.clamp(vector[3], -20.0, 20.0)).item(),
            dtype=vector.dtype,
            device=vector.device,
        )
        noise = torch.full(
            (3,),
            torch.exp(torch.clamp(vector[4], -20.0, 20.0)).item(),
            dtype=vector.dtype,
            device=vector.device,
        )
        phi_index = 5
    else:
        loading = torch.exp(torch.clamp(vector[3:6], -20.0, 20.0))
        noise = torch.exp(torch.clamp(vector[6:9], -20.0, 20.0))
        phi_index = 9
    phi = (
        torch.zeros((), dtype=vector.dtype, device=vector.device)
        if fix_phi
        else torch.clamp(
            torch.tanh(vector[phi_index]), -0.995, 0.995
        )
    )
    return mu, loading, noise, phi


def filter_dfm_torch(
    values: torch.Tensor,
    observed: torch.Tensor,
    parameters: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    mu, loading, noise, phi = parameters
    users, time_steps, _ = values.shape
    centered = torch.where(
        observed,
        values - mu[None, None, :],
        torch.zeros((), dtype=values.dtype, device=values.device),
    )
    mean = torch.zeros(users, dtype=values.dtype, device=values.device)
    variance = torch.ones(
        users, dtype=values.dtype, device=values.device
    )
    filtered_mean = torch.full(
        (users, time_steps),
        torch.nan,
        dtype=values.dtype,
        device=values.device,
    )
    filtered_variance = torch.full_like(
        filtered_mean, torch.nan
    )
    predicted_mean = torch.empty_like(filtered_mean)
    predicted_variance = torch.empty_like(filtered_mean)
    log_likelihood = torch.zeros(
        time_steps, dtype=values.dtype, device=values.device
    )
    inverse_noise = 1.0 / noise
    log_noise = torch.log(noise)

    for time_index in range(time_steps):
        mean_prediction = phi * mean
        variance_prediction = (
            phi**2 * variance + (1.0 - phi**2)
        )
        current_observed = observed[:, time_index, :]
        observation_count = current_observed.sum(dim=1)
        residual = torch.where(
            current_observed,
            centered[:, time_index, :]
            - loading[None, :] * mean_prediction[:, None],
            torch.zeros((), dtype=values.dtype, device=values.device),
        )
        a_term = (
            current_observed
            * loading[None, :] ** 2
            * inverse_noise[None, :]
        ).sum(dim=1)
        b_term = (
            current_observed
            * loading[None, :]
            * inverse_noise[None, :]
            * residual
        ).sum(dim=1)
        denominator = 1.0 + variance_prediction * a_term
        quadratic = (
            current_observed
            * residual**2
            * inverse_noise[None, :]
        ).sum(dim=1)
        quadratic = quadratic - (
            variance_prediction * b_term**2
        ) / denominator
        log_determinant = (
            current_observed * log_noise[None, :]
        ).sum(dim=1) + torch.log(denominator)
        has_observation = observation_count > 0
        ll = torch.where(
            has_observation,
            -0.5
            * (
                observation_count.to(values.dtype)
                * np.log(2.0 * np.pi)
                + log_determinant
                + quadratic
            ),
            torch.zeros((), dtype=values.dtype, device=values.device),
        )
        updated_variance = 1.0 / (
            1.0 / variance_prediction + a_term
        )
        updated_mean = updated_variance * (
            mean_prediction / variance_prediction + b_term
        )
        predicted_mean[:, time_index] = mean_prediction
        predicted_variance[:, time_index] = variance_prediction
        filtered_mean[:, time_index] = mean
        filtered_variance[:, time_index] = updated_variance
        mean = updated_mean
        variance = updated_variance
        log_likelihood[time_index] = ll.sum()
    return (
        filtered_mean,
        filtered_variance,
        predicted_mean,
        predicted_variance,
        log_likelihood,
    )


def dfm_loglik_torch(
    values: torch.Tensor,
    observed: torch.Tensor,
    parameters: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[torch.Tensor, torch.Tensor]:
    mu, loading, noise, phi = parameters
    users, time_steps, _ = values.shape
    centered = torch.where(
        observed,
        values - mu[None, None, :],
        torch.zeros((), dtype=values.dtype, device=values.device),
    )
    mean = torch.zeros(users, dtype=values.dtype, device=values.device)
    variance = torch.ones(
        users, dtype=values.dtype, device=values.device
    )
    total_log_likelihood = torch.zeros(
        (), dtype=values.dtype, device=values.device
    )
    inverse_noise = 1.0 / noise
    log_noise = torch.log(noise)
    for time_index in range(time_steps):
        mean_prediction = phi * mean
        variance_prediction = (
            phi**2 * variance + (1.0 - phi**2)
        )
        current_observed = observed[:, time_index, :]
        observation_count = current_observed.sum(dim=1)
        residual = torch.where(
            current_observed,
            centered[:, time_index, :]
            - loading[None, :] * mean_prediction[:, None],
            torch.zeros((), dtype=values.dtype, device=values.device),
        )
        a_term = (
            current_observed
            * loading[None, :] ** 2
            * inverse_noise[None, :]
        ).sum(dim=1)
        b_term = (
            current_observed
            * loading[None, :]
            * inverse_noise[None, :]
            * residual
        ).sum(dim=1)
        denominator = 1.0 + variance_prediction * a_term
        quadratic = (
            current_observed
            * residual**2
            * inverse_noise[None, :]
        ).sum(dim=1)
        quadratic = quadratic - (
            variance_prediction * b_term**2
        ) / denominator
        log_determinant = (
            current_observed * log_noise[None, :]
        ).sum(dim=1) + torch.log(denominator)
        has_observation = observation_count > 0
        ll = torch.where(
            has_observation,
            -0.5
            * (
                observation_count.to(values.dtype)
                * np.log(2.0 * np.pi)
                + log_determinant
                + quadratic
            ),
            torch.zeros((), dtype=values.dtype, device=values.device),
        )
        total_log_likelihood = total_log_likelihood + ll.sum()
        updated_variance = 1.0 / (
            1.0 / variance_prediction + a_term
        )
        updated_mean = updated_variance * (
            mean_prediction / variance_prediction + b_term
        )
        mean = updated_mean
        variance = updated_variance
    return total_log_likelihood, mean


def fit_dfm_torch(
    values: np.ndarray,
    observed: np.ndarray,
    fit_mask: np.ndarray,
    *,
    equal_structure: bool,
    fix_phi: bool,
    n_starts: int,
    random_seed: int = 20261111,
    device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    dtype = torch.float64
    complete_rows = values[
        fit_mask & observed.all(axis=-1)
    ]
    if len(complete_rows) < 50:
        raise RuntimeError("Too few complete observations to fit.")
    initializer = FactorAnalysis(
        n_components=1,
        max_iter=1000,
        random_state=random_seed,
    ).fit(complete_rows)
    loading = np.asarray(
        initializer.components_[0], dtype=np.float64
    )
    if loading[0] < 0:
        loading = -loading
    if equal_structure:
        loading = np.full(
            3, float(np.mean(np.abs(loading)))
        )
    noise = np.maximum(
        np.asarray(
            initializer.noise_variance_,
            dtype=np.float64,
        ),
        0.05,
    )
    if equal_structure:
        noise = np.full(3, float(np.mean(noise)))
    mu = complete_rows.mean(axis=0)
    combinations = optimizer_start_combinations(n_starts)
    start_vectors = [
        pack_parameters(
            mu,
            loading * loading_scale,
            noise * noise_scale,
            phi,
            equal_structure=equal_structure,
            fix_phi=fix_phi,
        )
        for phi, loading_scale, noise_scale in combinations
    ]
    values_t = torch.as_tensor(
        np.nan_to_num(values, nan=0.0),
        dtype=dtype,
        device=torch_device,
    )
    observed_t = torch.as_tensor(
        observed, dtype=torch.bool, device=torch_device
    )

    runs = []
    best = None
    for run_index, start in enumerate(start_vectors):
        vector = torch.tensor(
            start,
            dtype=dtype,
            device=torch_device,
            requires_grad=True,
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            params = unpack_parameters_torch(
                vector,
                equal_structure=equal_structure,
                fix_phi=fix_phi,
            )
            ll, _ = dfm_loglik_torch(
                values_t, observed_t, params
            )
            loss = -ll
            loss.backward()
            return loss

        optimizer = torch.optim.LBFGS(
            [vector],
            lr=0.5,
            max_iter=250,
            max_eval=300,
            tolerance_grad=1e-7,
            tolerance_change=1e-10,
            history_size=30,
            line_search_fn="strong_wolfe",
        )
        try:
            optimizer.step(closure)
            with torch.no_grad():
                params = unpack_parameters_torch(
                    vector,
                    equal_structure=equal_structure,
                    fix_phi=fix_phi,
                )
                ll, _ = dfm_loglik_torch(
                    values_t, observed_t, params
                )
                objective = float(-ll.item())
            runs.append(
                {
                    "start": run_index,
                    "success": True,
                    "fun": objective,
                }
            )
            if best is None or objective < best[0]:
                best = (
                    objective,
                    vector.detach().cpu().numpy().copy(),
                )
        except Exception as error:
            runs.append(
                {
                    "start": run_index,
                    "success": False,
                    "error": str(error),
                }
            )
    if best is None:
        raise RuntimeError("GPU DFM optimization failed for all starts.")
    parameters = unpack_parameters(
        best[1],
        equal_structure=equal_structure,
        fix_phi=fix_phi,
    )
    return {
        "unconstrained": best[1],
        "mu": parameters[0],
        "loading": parameters[1],
        "noise": parameters[2],
        "phi": parameters[3],
        "negative_log_likelihood": float(best[0]),
        "success": True,
        "message": "torch LBFGS",
        "runs": runs,
    }


def prepare_r42() -> dict:
    evidence, active, dataset = load_evidence()
    with np.load(OUT / "r42_masks.npz", allow_pickle=False) as masks:
        observed = masks["observed_mask"].astype(bool)
        rank_mask = masks["rank_mask"].astype(bool)
    fit_mask = np.zeros(active.shape, dtype=bool)
    fit_mask[:, FIT_START : FIT_END + 1] = True
    fit_mask &= rank_mask
    holdout_mask = np.zeros(active.shape, dtype=bool)
    holdout_mask[:, H_SECOND_HALF_START :] = True
    holdout_mask &= rank_mask
    second_half_mask = np.zeros(active.shape, dtype=bool)
    second_half_mask[
        :, H_SECOND_HALF_START : H_SECOND_HALF_END + 1
    ] = True
    second_half_mask &= rank_mask
    development_test_mask = np.zeros(active.shape, dtype=bool)
    development_test_mask[
        :, DEVELOPMENT_TEST_START : DEVELOPMENT_TEST_END + 1
    ] = True
    development_test_mask &= rank_mask
    transformed = empirical_normal_transform(
        evidence, observed, fit_mask
    )
    return {
        "evidence": evidence,
        "active": active,
        "dataset": dataset,
        "observed": observed,
        "rank_mask": rank_mask,
        "fit_mask": fit_mask,
        "holdout_mask": holdout_mask,
        "second_half_mask": second_half_mask,
        "development_test_mask": development_test_mask,
        "transformed": transformed,
    }


def fit_real_models(n_starts: int, backend: str) -> None:
    if backend == "torch":
        fit_function = fit_dfm_torch
    elif backend == "numba":
        fit_function = fit_dfm_numba
    else:
        fit_function = fit_dfm
    prepared = prepare_r42()
    values = prepared["transformed"]
    observed = prepared["observed"]
    fit_mask = prepared["fit_mask"]
    free = fit_function(
        values,
        observed,
        fit_mask,
        equal_structure=False,
        fix_phi=False,
        n_starts=n_starts,
    )
    equal = fit_function(
        values,
        observed,
        fit_mask,
        equal_structure=True,
        fix_phi=False,
        n_starts=n_starts,
    )
    static = fit_function(
        values,
        observed,
        fit_mask,
        equal_structure=False,
        fix_phi=True,
        n_starts=n_starts,
    )
    second_half = fit_function(
        values,
        observed,
        prepared["second_half_mask"],
        equal_structure=False,
        fix_phi=False,
        n_starts=n_starts,
    )
    result = {
        "fit_period": [
            FIT_START,
            FIT_END,
        ],
        "n_starts": n_starts,
        "backend": backend,
        "free": {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in free.items()
            if key != "runs"
        },
        "equal": {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in equal.items()
            if key != "runs"
        },
        "static": {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in static.items()
            if key != "runs"
        },
        "second_half": {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in second_half.items()
            if key != "runs"
        },
        "labels_read": False,
    }
    (OUT / "s1_fit_models.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def fit_free_model_confirmation(n_starts: int) -> None:
    prepared = prepare_r42()
    fitted = fit_dfm_numba(
        prepared["transformed"],
        prepared["observed"],
        prepared["fit_mask"],
        equal_structure=False,
        fix_phi=False,
        n_starts=n_starts,
    )
    result = {
        key: (
            value.tolist()
            if isinstance(value, np.ndarray)
            else value
        )
        for key, value in fitted.items()
    }
    result["fit_period"] = [FIT_START, FIT_END]
    result["n_starts"] = n_starts
    result["labels_read"] = False
    (OUT / "s1_free20_confirmation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def write_s1_gate_report() -> None:
    free20 = json.loads(
        (OUT / "s1_free20_confirmation.json").read_text(
            encoding="utf-8"
        )
    )
    all_models = json.loads(
        (OUT / "s1_fit_models.json").read_text(
            encoding="utf-8"
        )
    )
    loading = np.asarray(free20["loading"], dtype=np.float64)
    noise = np.asarray(free20["noise"], dtype=np.float64)
    standardized_loading = loading / np.sqrt(
        loading**2 + noise
    )
    v1 = bool(
        (loading > 0).all()
        and (standardized_loading >= 0.2).all()
    )
    v2 = bool((noise >= 0.01).all())
    v3 = bool(abs(float(free20["phi"])) <= 0.95)
    report = {
        "fit_period": free20["fit_period"],
        "n_starts": free20["n_starts"],
        "labels_read": False,
        "free_model_parameters": {
            "mu": free20["mu"],
            "loading": free20["loading"],
            "noise": free20["noise"],
            "phi": free20["phi"],
            "negative_log_likelihood": free20[
                "negative_log_likelihood"
            ],
        },
        "standardized_loading": standardized_loading.tolist(),
        "V1_loading_pass": v1,
        "V2_noise_pass": v2,
        "V3_phi_pass": v3,
        "V4_V5_evaluated": False,
        "model_applicable": bool(v1 and v2 and v3),
        "stop_rule_triggered": bool(not (v1 and v2 and v3)),
        "reason": (
            "channel_top2 loading collapsed below 0.2 and "
            "masked noise collapsed below 0.01; the one-factor "
            "model is a boundary solution."
        ),
        "equal_model": all_models["equal"],
        "static_model": all_models["static"],
        "second_half_model": all_models["second_half"],
    }
    (OUT / "s1_gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def numerical_check(backend: str) -> None:
    if backend == "torch":
        fit_function = fit_dfm_torch
    elif backend == "numba":
        fit_function = fit_dfm_numba
    else:
        fit_function = fit_dfm
    rng = np.random.default_rng(20261111)
    users, time_steps = 80, 120
    phi = 0.6
    loading = np.asarray([0.8, 0.6, 0.5])
    noise = np.maximum(1.0 - loading**2, 0.05)
    mu = np.asarray([0.1, -0.1, 0.0])
    states = np.zeros((users, time_steps))
    states[:, 0] = rng.normal(size=users)
    innovation_scale = np.sqrt(1.0 - phi**2)
    for t in range(1, time_steps):
        states[:, t] = (
            phi * states[:, t - 1]
            + innovation_scale * rng.normal(size=users)
        )
    values = (
        mu[None, None, :]
        + states[:, :, None] * loading[None, None, :]
        + rng.normal(
            scale=np.sqrt(noise)[None, None, :],
            size=(users, time_steps, 3),
        )
    )
    observed = np.ones_like(values, dtype=bool)
    # Add deterministic partial missingness.
    observed[::11, 20:25, 1] = False
    observed[::17, 40:44, 2] = False
    fit_mask = np.zeros((users, time_steps), dtype=bool)
    fit_mask[:, :80] = True
    recovered = fit_function(
        values,
        observed,
        fit_mask,
        equal_structure=False,
        fix_phi=False,
        n_starts=5,
    )
    report = {
        "truth": {
            "phi": phi,
            "loading": loading.tolist(),
            "noise": noise.tolist(),
            "mu": mu.tolist(),
        },
        "recovered": {
            "phi": recovered["phi"],
            "loading": recovered["loading"].tolist(),
            "noise": recovered["noise"].tolist(),
            "mu": recovered["mu"].tolist(),
            "negative_log_likelihood": recovered[
                "negative_log_likelihood"
            ],
            "success": recovered["success"],
        },
    }
    (OUT / "numerical_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def audit_s0() -> None:
    evidence, active, dataset = load_evidence()
    observed = np.isfinite(evidence)
    eligible = active & observed.any(axis=-1)
    observed_count = observed.sum(axis=-1)
    partial_ratio = float(
        (
            eligible
            & (observed_count > 0)
            & (observed_count < evidence.shape[-1])
        ).sum()
        / max(eligible.sum(), 1)
    )
    k_min = 3 if partial_ratio < 0.005 else 2
    rank_mask = eligible & (observed_count >= k_min)

    values = dataset["values"]
    feature_names = dataset["features"].astype(str)
    zero_vector = np.nan_to_num(values).sum(axis=-1) == 0
    no_event = (
        np.nan_to_num(values[:, :, 0], nan=-1.0)
        < 0.0
    )

    user_day = pd.read_csv(
        USER_DAY, parse_dates=["date"]
    )
    csv_keys = set(
        zip(
            user_day["user_id"].astype(str),
            user_day["date"].dt.strftime("%Y-%m-%d"),
        )
    )
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    zero_with_csv = 0
    zero_without_csv = 0
    for user_index, time_index in zip(*np.nonzero(zero_vector & active)):
        key = (
            users[user_index],
            dates[time_index].strftime("%Y-%m-%d"),
        )
        if key in csv_keys:
            zero_with_csv += 1
        else:
            zero_without_csv += 1

    repeated_fraction = {}
    for channel_index, name in enumerate(
        ("masked", "manifold", "channel")
    ):
        channel_values = evidence[:, :, channel_index][eligible]
        counts = pd.Series(channel_values).value_counts(
            normalize=True
        )
        repeated_fraction[name] = float(counts.iloc[0])

    np.savez_compressed(
        OUT / "r42_masks.npz",
        active_mask=active,
        eligible_mask=eligible,
        observed_mask=observed,
        rank_mask=rank_mask,
        observed_count=observed_count,
    )
    audit = {
        "dataset": str(DATASET),
        "active_user_days": int(active.sum()),
        "inactive_user_days": int((~active).sum()),
        "eligible_user_days": int(eligible.sum()),
        "rank_user_days": int(rank_mask.sum()),
        "partial_observed_ratio": partial_ratio,
        "k_min": k_min,
        "observed_count_distribution": {
            str(int(key)): int(value)
            for key, value in pd.Series(
                observed_count[eligible]
            ).value_counts().sort_index().items()
        },
        "zero_vector_active_user_days": int(
            (zero_vector & active).sum()
        ),
        "zero_vector_with_csv_row": int(zero_with_csv),
        "zero_vector_without_csv_row": int(zero_with_csv_no := zero_without_csv),
        "no_event_active_user_days": int(
            (no_event & eligible).sum()
        ),
        "no_event_ratio_active": float(
            (no_event & eligible).sum() / max(eligible.sum(), 1)
        ),
        "most_repeated_value_fraction_by_channel": repeated_fraction,
        "feature_names": feature_names.tolist(),
        "train_end_index": int(dataset["train_end_index"]),
        "validation_start_index": int(dataset["val_start_index"]),
        "validation_end_index": int(dataset["val_end_index"]),
        "test_start_index": int(dataset["test_start_index"]),
        "test_end_index": int(dataset["test_end_index"]),
        "labels_read": False,
        "mask_file": str(OUT / "r42_masks.npz"),
    }
    (OUT / "s0_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "s0",
            "numcheck",
            "s1fit",
            "s1free20",
            "s1report",
        ],
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--backend",
        choices=["numba", "torch", "cpu"],
        default="numba",
    )
    args = parser.parse_args()
    if args.stage == "s0":
        audit_s0()
    elif args.stage == "numcheck":
        numerical_check(args.backend)
    elif args.stage == "s1fit":
        fit_real_models(args.n_starts, args.backend)
    elif args.stage == "s1free20":
        fit_free_model_confirmation(20)
    elif args.stage == "s1report":
        write_s1_gate_report()


if __name__ == "__main__":
    main()
