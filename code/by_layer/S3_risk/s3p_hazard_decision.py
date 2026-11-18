from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
SEEDS = [20261111, 20261112, 20261113]
PRIMARY_K = 10
HORIZON_DAYS = 3

sys.path.insert(0, str(ROOT))
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42,
)
from v2_m_pipeline import add_utilities, evaluate_block  # noqa: E402


def load_triple_evidence(
    data: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = np.load(
        OUT / "e17_r42_M0_masked_quantile.npy"
    ).astype(np.float32)
    manifold = np.mean(
        np.stack(
            [
                np.load(
                    OUT
                    / f"e19_r42_learned_manifold_quantile_{seed}.npy"
                )
                for seed in SEEDS
            ],
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)
    with np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    ) as store:
        channel = store[
            "r42_original_channel_top2_quantile"
        ].astype(np.float32)
    return masked, manifold, channel


def rolling_features(
    masked: np.ndarray,
    manifold: np.ndarray,
    channel: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    users, time_steps = masked.shape
    severity = (masked + manifold + channel) / 3.0
    features = np.full(
        (users, time_steps, 8),
        np.nan,
        dtype=np.float32,
    )

    for user_index in range(users):
        active = np.flatnonzero(active_mask[user_index])
        high_streak = 0
        for time_index in range(time_steps):
            if not active_mask[user_index, time_index]:
                high_streak = 0
                continue
            left = max(0, time_index - 6)
            past = np.flatnonzero(
                active_mask[user_index, left : time_index + 1]
            ) + left
            values = severity[user_index, past]
            mean3 = float(
                values[-3:].mean() if len(values) else 0.0
            )
            mean7 = float(values.mean() if len(values) else 0.0)
            max7 = float(values.max() if len(values) else 0.0)
            count_high = int((values > 3.0).sum())
            high_streak = (
                high_streak + 1
                if float(severity[user_index, time_index]) > 3.0
                else 0
            )
            evidence = np.asarray(
                [
                    masked[user_index, time_index],
                    manifold[user_index, time_index],
                    channel[user_index, time_index],
                ],
                dtype=np.float32,
            )
            features[user_index, time_index] = (
                severity[user_index, time_index],
                mean3,
                mean7,
                max7,
                count_high,
                high_streak,
                severity[user_index, time_index] - mean3,
                float(np.std(evidence)),
            )
    return severity, features


def build_future_event_target(
    data: dict,
) -> np.ndarray:
    users = data["users"].astype(str)
    dates = pd.to_datetime(data["dates"]).normalize()
    active_mask = data["active_mask"]
    user_index = {user: i for i, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): i
        for i, date in enumerate(dates)
    }
    target = np.zeros(active_mask.shape, dtype=np.uint8)
    incidents = data["incidents"].copy()
    incidents["start_date"] = pd.to_datetime(
        incidents["start"]
    ).dt.normalize()
    for incident in incidents.itertuples(index=False):
        start = date_index.get(
            incident.start_date.strftime("%Y-%m-%d")
        )
        if start is None:
            continue
        account_names = str(incident.observed_accounts).split("|")
        for offset in range(1, HORIZON_DAYS + 1):
            prior = start - offset
            if prior < 0:
                continue
            for account in account_names:
                user = user_index.get(account)
                if user is not None and active_mask[user, prior]:
                    target[user, prior] = 1
    return target


def fit_hazard_model(
    features: np.ndarray,
    target: np.ndarray,
    active_mask: np.ndarray,
    train_days: tuple[int, int],
    calibration_days: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    h = HORIZON_DAYS
    train_start, train_end = train_days
    cal_start, cal_end = calibration_days
    train_rows = np.zeros(active_mask.shape, dtype=bool)
    train_rows[
        :, train_start : train_end - h + 1
    ] = True
    train_rows &= active_mask
    calibration_rows = np.zeros(
        active_mask.shape, dtype=bool
    )
    calibration_rows[
        :, cal_start : cal_end - h + 1
    ] = True
    calibration_rows &= active_mask

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            solver="lbfgs",
            max_iter=2000,
            class_weight=None,
            random_state=20261111,
        ),
    )
    model.fit(
        features[train_rows],
        target[train_rows],
    )
    prediction_features = features.copy()
    prediction_features[~active_mask] = 0.0
    raw_probability = model.predict_proba(
        prediction_features.reshape(
            -1, features.shape[-1]
        )
    )[:, 1].reshape(features.shape[:2])
    calibration_probability = raw_probability[
        calibration_rows
    ]
    calibration_target = target[calibration_rows]
    # Platt calibration on validation second half.
    calibration_model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=20261111,
    )
    calibration_model.fit(
        np.log(
            np.clip(calibration_probability, 1e-6, 1 - 1e-6)
            / (1 - np.clip(
                calibration_probability, 1e-6, 1 - 1e-6
            ))
        ).reshape(-1, 1),
        calibration_target,
    )
    raw_logit = np.log(
        np.clip(raw_probability, 1e-6, 1 - 1e-6)
        / (1 - np.clip(raw_probability, 1e-6, 1 - 1e-6))
    )
    raw_for_calibration = raw_probability.copy()
    raw_for_calibration[~active_mask] = 0.0
    raw_logit_for_calibration = np.log(
        np.clip(raw_for_calibration, 1e-6, 1 - 1e-6)
        / (1 - np.clip(
            raw_for_calibration, 1e-6, 1 - 1e-6
        ))
    )
    calibrated = calibration_model.predict_proba(
        raw_logit_for_calibration.reshape(-1, 1)
    )[:, 1].reshape(raw_probability.shape)
    calibrated[~active_mask] = np.nan
    brier = float(
        np.mean((calibration_probability - calibration_target) ** 2)
    )
    return calibrated, {
        "train_rows": int(train_rows.sum()),
        "train_positive": int(target[train_rows].sum()),
        "calibration_rows": int(calibration_rows.sum()),
        "calibration_positive": int(
            calibration_target.sum()
        ),
        "base_model": "L2 logistic hazard",
        "C": 0.1,
        "platt_slope": float(
            calibration_model.coef_[0, 0]
        ),
        "platt_intercept": float(
            calibration_model.intercept_[0]
        ),
        "calibration_brier": brier,
        "target": (
            f"incident start in next {HORIZON_DAYS} days "
            "for an observed account"
        ),
    }


def main() -> None:
    data = load_r42()
    masked, manifold, channel = load_triple_evidence(data)
    target = build_future_event_target(data)
    severity, features = rolling_features(
        masked,
        manifold,
        channel,
        data["active_mask"],
    )
    hazard, fit_summary = fit_hazard_model(
        features,
        target,
        data["active_mask"],
        train_days=(159, 230),
        calibration_days=(231, 302),
    )
    signal = daily_percentile(
        hazard, data["active_mask"]
    ).astype(np.float32)

    original, augmented = fit_view_scores(
        data,
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        hazard,
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
            "S3-P_hazard_decision",
            external=False,
        )
    )
    metrics.to_csv(
        OUT / "s3p_hazard_r42_original_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    test_metrics = metrics[
        (metrics["split"] == "test")
        & (metrics["k"] == PRIMARY_K)
    ]
    baseline = pd.read_csv(
        OUT / "v2m_s_metrics.csv"
    )
    baseline = baseline[
        (baseline["data_block"] == "r42_original")
        & (baseline["split"] == "test")
        & (baseline["k"] == PRIMARY_K)
        & (baseline["variant"] == "aligned_iforest")
    ]
    summary = {
        "model": "S3-P hazard + budgeted decision pilot",
        "target": fit_summary["target"],
        "horizon_days": HORIZON_DAYS,
        "fit": fit_summary,
        "primary_k": PRIMARY_K,
        "test_metrics": test_metrics.to_dict(
            orient="records"
        ),
        "baseline_v2m_s": baseline.to_dict(
            orient="records"
        ),
        "labels_used_in_model_fit": True,
        "confirmation_status": (
            "Pilot only: validation labels used for calibration; "
            "test is a single development holdout, not independent "
            "confirmation."
        ),
    }
    (OUT / "s3p_hazard_r42_original_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "s3p_hazard_r42_original_components.npz",
        feature_names=np.asarray(
            [
                "severity",
                "mean3",
                "mean7",
                "max7",
                "count_high7",
                "high_streak",
                "trend3",
                "evidence_std",
            ]
        ),
        features=features,
        target=target,
        hazard=hazard,
        hazard_signal=signal,
        hazard_aligned_iforest=auxiliary,
        active_mask=data["active_mask"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
