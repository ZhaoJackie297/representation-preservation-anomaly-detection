from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
ARRAYS_FILE = OUT / "ts2vec_dual_score_arrays.npz"

K_GRID = [3, 5, 10, 20, 30, 50]
PRIMARY_K = 10
ALPHA = 0.4
COOLDOWN_DAYS = 7
DRIFT_QUANTILE = 0.95

sys.path.insert(0, str(ROOT))
from application_utility_analysis import PROFILES  # noqa: E402
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from production_protocol_r42 import build_alerts  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    daily_percentile,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)
from risk_topk_r42 import compute_risk  # noqa: E402


def tail_anomaly_score(value: float, history: np.ndarray) -> float:
    finite = history[np.isfinite(history)]
    if len(finite) == 0:
        return 0.0
    ordered = np.sort(finite)
    rank = int(np.searchsorted(ordered, value, side="right"))
    n = len(ordered)
    tail_probability = (n - rank + 1) / (n + 1)
    return max(0.0, -math.log(tail_probability))


def rolling_personal_quantile(
    raw_scores: np.ndarray,
    active_mask: np.ndarray,
    window_days: int,
) -> np.ndarray:
    calibrated = np.full(raw_scores.shape, np.nan, dtype=np.float32)
    for user_index in range(raw_scores.shape[0]):
        active_indices = np.flatnonzero(active_mask[user_index])
        raw_history: list[float] = []
        history_indices: list[int] = []
        history_cursor = 0
        for time_index in active_indices:
            value = float(raw_scores[user_index, time_index])
            if not np.isfinite(value):
                continue
            while (
                history_cursor < len(history_indices)
                and history_indices[history_cursor]
                <= time_index - window_days - 1
            ):
                history_cursor += 1
            reference = np.asarray(
                raw_history[history_cursor:], dtype=np.float64
            )
            calibrated[user_index, time_index] = tail_anomaly_score(
                value, reference
            )
            raw_history.append(value)
            history_indices.append(int(time_index))
    return calibrated


def combine_quantiles(
    masked_quantile: np.ndarray,
    latent_quantile: np.ndarray,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = (masked_quantile + latent_quantile) / 2.0
    dynamic = compute_risk(users, dates, base, active_mask)
    signal = daily_percentile(dynamic, active_mask)
    return signal.astype(np.float32), dynamic.astype(np.float32)


def population_drift_signal(
    values: np.ndarray,
    active_mask: np.ndarray,
    window_days: int,
    train_end: int,
) -> np.ndarray:
    users, time_steps, dimensions = values.shape
    daily_profiles = np.full(
        (time_steps, dimensions), np.nan, dtype=np.float64
    )
    for time_index in range(time_steps):
        active = active_mask[:, time_index]
        if not active.any():
            continue
        daily_profiles[time_index] = np.nanmean(
            values[active, time_index], axis=0
        )

    training_finite_days = np.isfinite(
        daily_profiles[: train_end + 1]
    ).all(axis=1)
    if not training_finite_days.any():
        return np.zeros(time_steps, dtype=np.float32)
    reference_rows = daily_profiles[: train_end + 1][
        training_finite_days
    ]
    center = reference_rows.mean(axis=0)
    scale = reference_rows.std(axis=0)
    scale = np.where(
        np.isfinite(scale) & (scale >= 1e-8),
        scale,
        1.0,
    )
    normalized = (daily_profiles - center) / scale
    drift = np.full(time_steps, np.nan, dtype=np.float64)
    for time_index in range(window_days, time_steps):
        reference_window = normalized[
            time_index - window_days : time_index
        ]
        valid = np.isfinite(reference_window).all(axis=1)
        if not valid.any():
            continue
        mean_shift = reference_window[valid].mean(axis=0)
        drift[time_index] = np.sqrt(
            np.mean(np.square(mean_shift))
        )
    return drift.astype(np.float32)


def drift_threshold_and_state(
    drift: np.ndarray,
    train_end: int,
    window_days: int,
) -> tuple[float, np.ndarray]:
    training_values = drift[
        window_days : train_end + 1
    ][np.isfinite(drift[window_days : train_end + 1])]
    if len(training_values) == 0:
        raise RuntimeError("No finite training drift values.")
    threshold = float(np.quantile(training_values, DRIFT_QUANTILE))
    active = np.isfinite(drift) & (drift > threshold)
    return threshold, active.astype(bool)


def gated_rolling_quantile(
    fixed_quantile: np.ndarray,
    rolling_quantile: np.ndarray,
    drift_active: np.ndarray,
) -> np.ndarray:
    calibrated = fixed_quantile.copy()
    calibrated[:, drift_active] = rolling_quantile[:, drift_active]
    return calibrated.astype(np.float32)


def add_application_utilities(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    for profile_name, weights in PROFILES.items():
        metrics[profile_name] = sum(
            metrics[column] * weight
            for column, weight in weights.items()
        )
    return metrics


def evaluate_signal(
    data: dict,
    signal: np.ndarray,
    auxiliary: np.ndarray,
    method: str,
    *,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        q = q_from_alpha(k, ALPHA)
        alerts = build_unique_quota_alerts(
            signal,
            auxiliary,
            data["active_mask"],
            k,
            q,
        )
        result = evaluate_alert_system(
            data,
            alerts,
            method,
            k,
            external=external,
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "k": k,
                    "q": q,
                    **compact_metric(
                        result, None if external else split
                    ),
                }
            )
    return pd.DataFrame(rows)


def evaluate_single_signal(
    data: dict,
    signal: np.ndarray,
    method: str,
    *,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        alerts = build_alerts(
            signal,
            data["active_mask"],
            k,
            COOLDOWN_DAYS,
        )
        result = evaluate_alert_system(
            data,
            alerts,
            method,
            k,
            external=external,
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "k": k,
                    "q": 0,
                    **compact_metric(
                        result, None if external else split
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_rankings(k10: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (block, split), group in k10.groupby(["data_block", "split"]):
        for profile in PROFILES:
            ordered = group.sort_values(
                [profile, "precision_at_k"],
                ascending=False,
            ).copy()
            ordered["rank"] = np.arange(1, len(ordered) + 1)
            for row in ordered.itertuples(index=False):
                rows.append(
                    {
                        "data_block": block,
                        "split": split,
                        "profile": profile,
                        "method": row.method,
                        "utility": float(getattr(row, profile)),
                        "rank": int(row.rank),
                    }
                )
    return pd.DataFrame(rows)


def summarize_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    maxima = rankings.groupby(
        ["data_block", "split", "profile"]
    )["utility"].transform("max")
    rankings = rankings.copy()
    rankings["relative_utility"] = rankings["utility"] / maxima
    rankings["regret"] = 1.0 - rankings["relative_utility"]
    return (
        rankings.groupby("method", as_index=False)
        .agg(
            appearances=("rank", "size"),
            first_place=(
                "rank",
                lambda values: int((values == 1).sum()),
            ),
            top_three=(
                "rank",
                lambda values: int((values <= 3).sum()),
            ),
            mean_rank=("rank", "mean"),
            worst_rank=("rank", "max"),
            mean_utility=("utility", "mean"),
            mean_relative_utility=("relative_utility", "mean"),
            worst_relative_utility=("relative_utility", "min"),
            mean_regret=("regret", "mean"),
            worst_regret=("regret", "max"),
        )
        .sort_values(
            ["mean_rank", "mean_regret"],
            ascending=[True, True],
        )
    )


def main() -> None:
    arrays = np.load(ARRAYS_FILE, allow_pickle=False)
    r42 = load_r42()
    r52 = load_r52()
    r42_dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    r52_dataset = np.load(
        ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
        allow_pickle=False,
    )

    blocks = [
        {
            "name": "r42_original",
            "data": r42,
            "external": False,
            "values": r42_dataset["values"].astype(np.float32),
            "train_end": int(r42_dataset["train_end_index"]),
            "existing_signal": arrays["r42_original_ts_dual_mean_signal"],
        },
        {
            "name": "r42_retrained",
            "data": r42,
            "external": False,
            "values": r42_dataset["values"].astype(np.float32),
            "train_end": int(r42_dataset["train_end_index"]),
            "existing_signal": arrays[
                "r42_retrained_ts_dual_mean_signal"
            ],
        },
        {
            "name": "r52_external",
            "data": r52,
            "external": True,
            "values": r52_dataset["values"].astype(np.float32),
            "train_end": int(r52_dataset["train_end_index"]),
            "existing_signal": arrays["r52_external_ts_dual_mean_signal"],
        },
    ]

    metric_frames = []
    drift_rows = []
    saved_arrays = {}
    diagnostics = []
    for block in blocks:
        name = str(block["name"])
        data = block["data"]
        active = data["active_mask"]
        users = data["users"]
        dates = data["dates"]
        train_end = int(block["train_end"])
        masked_raw = arrays[f"{name}_masked_error"].astype(np.float32)
        latent_raw = arrays[f"{name}_latent_raw"].astype(np.float32)
        auxiliary = arrays[f"{name}_dual_view_iforest"].astype(np.float32)

        fixed_mask = per_user_quantile_scores(
            masked_raw, active, train_end
        )
        fixed_latent = per_user_quantile_scores(
            latent_raw, active, train_end
        )
        fixed_signal, _ = combine_quantiles(
            fixed_mask, fixed_latent, users, dates, active
        )
        fixed_diff = float(
            np.nanmax(
                np.abs(fixed_signal - block["existing_signal"])
            )
        )

        rolling_mask30 = rolling_personal_quantile(
            masked_raw, active, 30
        )
        rolling_latent30 = rolling_personal_quantile(
            latent_raw, active, 30
        )
        rolling_signal30, _ = combine_quantiles(
            rolling_mask30,
            rolling_latent30,
            users,
            dates,
            active,
        )
        rolling_mask60 = rolling_personal_quantile(
            masked_raw, active, 60
        )
        rolling_latent60 = rolling_personal_quantile(
            latent_raw, active, 60
        )
        rolling_signal60, _ = combine_quantiles(
            rolling_mask60,
            rolling_latent60,
            users,
            dates,
            active,
        )

        drift = population_drift_signal(
            block["values"], active, 30, train_end
        )
        drift_threshold, drift_active = drift_threshold_and_state(
            drift, train_end, 30
        )
        gated_mask30 = gated_rolling_quantile(
            fixed_mask, rolling_mask30, drift_active
        )
        gated_latent30 = gated_rolling_quantile(
            fixed_latent, rolling_latent30, drift_active
        )
        gated_signal30, _ = combine_quantiles(
            gated_mask30,
            gated_latent30,
            users,
            dates,
            active,
        )

        variants = {
            "fixed_dual": fixed_signal,
            "rolling30_dual": rolling_signal30,
            "rolling60_dual": rolling_signal60,
            "drift30_dual": gated_signal30,
        }
        for method, signal in variants.items():
            frame = evaluate_signal(
                data,
                signal,
                auxiliary,
                method,
                external=bool(block["external"]),
            )
            frame.insert(0, "data_block", name)
            frame["external"] = int(bool(block["external"]))
            metric_frames.append(frame)
            saved_arrays[f"{name}_{method}_signal"] = signal.astype(
                np.float32
            )

        iforest = evaluate_single_signal(
            data,
            auxiliary,
            "dual_iforest_complete",
            external=bool(block["external"]),
        )
        iforest.insert(0, "data_block", name)
        iforest["external"] = int(bool(block["external"]))
        metric_frames.append(iforest)

        for time_index in range(len(dates)):
            drift_rows.append(
                {
                    "data_block": name,
                    "date": dates[time_index].strftime("%Y-%m-%d"),
                    "split": (
                        "train"
                        if time_index <= train_end
                        else "external_test"
                        if block["external"]
                        else "validation"
                        if time_index <= data["val_end"]
                        else "test"
                    ),
                    "drift_value": float(drift[time_index])
                    if np.isfinite(drift[time_index])
                    else np.nan,
                    "drift_threshold": drift_threshold,
                    "drift_active": int(drift_active[time_index]),
                }
            )

        saved_arrays[f"{name}_drift_value"] = drift
        saved_arrays[f"{name}_drift_active"] = drift_active.astype(
            np.uint8
        )
        diagnostics.append(
            {
                "data_block": name,
                "fixed_signal_max_abs_diff_vs_existing": fixed_diff,
                "drift_threshold": drift_threshold,
                "drift_active_days": int(drift_active.sum()),
                "drift_active_validation_days": int(
                    drift_active[
                        train_end + 1 : data.get("val_end", train_end)
                        + 1
                    ].sum()
                ),
                "drift_active_test_days": int(
                    drift_active[
                        (data.get("val_end", train_end) + 1) :
                    ].sum()
                )
                if not block["external"]
                else int(drift_active[data["test_start"] :].sum()),
            }
        )

    metrics = add_application_utilities(
        pd.concat(metric_frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "drift_calibration_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(drift_rows).to_csv(
        OUT / "drift_calibration_daily_drift.csv",
        index=False,
        encoding="utf-8",
    )
    k10 = metrics[metrics["k"] == PRIMARY_K].copy()
    rankings = build_rankings(k10)
    stability = summarize_stability(rankings)
    rankings.to_csv(
        OUT / "drift_calibration_k10_rankings.csv",
        index=False,
        encoding="utf-8",
    )
    stability.to_csv(
        OUT / "drift_calibration_k10_rank_stability.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "drift_calibration_arrays.npz",
        **saved_arrays,
    )

    summary = {
        "primary_k": PRIMARY_K,
        "q": q_from_alpha(PRIMARY_K, ALPHA),
        "drift_quantile": DRIFT_QUANTILE,
        "drift_window_days": 30,
        "rolling_windows": [30, 60],
        "variants": [
            "fixed_dual",
            "rolling30_dual",
            "rolling60_dual",
            "drift30_dual",
        ],
        "labels_used_for_calibration": False,
        "labels_used_for_drift_detection": False,
        "diagnostics": diagnostics,
        "outputs": {
            "metrics": str(OUT / "drift_calibration_metrics.csv"),
            "daily_drift": str(
                OUT / "drift_calibration_daily_drift.csv"
            ),
            "k10_rankings": str(
                OUT / "drift_calibration_k10_rankings.csv"
            ),
            "k10_rank_stability": str(
                OUT / "drift_calibration_k10_rank_stability.csv"
            ),
            "arrays": str(OUT / "drift_calibration_arrays.npz"),
        },
    }
    (OUT / "drift_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nK=10 rank stability", flush=True)
    print(stability.to_string(index=False), flush=True)
    print("\nK=10 winners", flush=True)
    print(
        rankings[rankings["rank"] == 1].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
