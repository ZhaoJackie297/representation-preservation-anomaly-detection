from __future__ import annotations

import numpy as np


def sanitize_user_day_features(
    values: np.ndarray,
    *,
    count_dimensions: int,
) -> np.ndarray:
    """Apply the shared count and timing-feature missing-value contract."""
    output = np.asarray(values, dtype=np.float64).copy()
    counts = output[:, :, :count_dimensions]
    output[:, :, :count_dimensions] = np.where(
        np.isfinite(counts),
        np.log1p(np.maximum(counts, 0.0)),
        np.nan,
    )
    timing = output[:, :, count_dimensions:]
    output[:, :, count_dimensions:] = np.where(
        np.isfinite(timing) & (timing >= 0.0),
        timing,
        np.nan,
    )
    return output.astype(np.float32)


def build_user_profiles(
    raw: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    """Build standardized profiles strictly from the training prefix."""
    if raw.shape[:2] != active_mask.shape:
        raise ValueError("raw and active_mask must share user/day dimensions")
    if train_end < 0 or train_end >= raw.shape[1]:
        raise ValueError("train_end must index an existing day")

    profiles = []
    for user_index in range(raw.shape[0]):
        mask = active_mask[user_index, : train_end + 1]
        values = raw[user_index, : train_end + 1][mask]
        if len(values) == 0:
            values = raw[user_index, : train_end + 1]
        median = np.nanmedian(values, axis=0)
        q90 = np.nanquantile(values, 0.9, axis=0)
        active_fraction = np.asarray([mask.mean()])
        profiles.append(np.concatenate([median, q90, active_fraction]))

    profile = np.stack(profiles)
    population_median = np.nanmedian(profile, axis=0)
    population_median = np.where(
        np.isfinite(population_median),
        population_median,
        0.0,
    )
    missing_rows, missing_columns = np.nonzero(~np.isfinite(profile))
    profile[missing_rows, missing_columns] = population_median[
        missing_columns
    ]
    mean = profile.mean(axis=0)
    std = profile.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return ((profile - mean) / std).astype(np.float64)
