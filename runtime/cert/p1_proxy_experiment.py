from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
K = 10
Q_LIMIT = 4
COOLDOWN_DAYS = 7

COUNT_COLUMNS = [
    "logon_count",
    "usb_connect_count",
    "file_copy_count",
    "email_count",
    "http_count",
]

sys.path.insert(0, str(ROOT))
from application_utility_analysis import PROFILES  # noqa: E402
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    daily_percentile,
    evaluate_alert_system,
    load_r42,
    load_r52,
)
from risk_topk_r42 import compute_risk  # noqa: E402


def load_raw_features(
    path: Path,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    return frame.set_index(["user_id", "date"]).reindex(index)


def build_proxy_scores(
    frame: pd.DataFrame,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-6
    total = frame[COUNT_COLUMNS].sum(axis=1).clip(lower=0)
    modality = (frame[COUNT_COLUMNS] > 0).sum(axis=1)
    active_span = frame["active_span_minutes"].clip(lower=0)
    max_channel = frame[COUNT_COLUMNS].max(axis=1)
    after_hours = (
        frame["after_hours_logon_count"]
        + frame["after_hours_email_count"]
        + frame["after_hours_http_count"]
    )

    midas = pd.DataFrame(
        {
            "log_total": np.log1p(total),
            "modality_count": modality,
            "event_rate": total / (active_span + 1.0),
            "max_channel_share": max_channel / (total + eps),
            "after_hours_share": after_hours / (total + eps),
            "other_pc_count": frame["other_pc_logon_count"],
        }
    )
    relationship = pd.DataFrame(
        {
            "external_email_ratio": frame["external_recipient_count"]
            / (frame["email_recipient_count"] + 1.0),
            "attachment_ratio": frame["email_attachment_count"]
            / (frame["email_count"] + 1.0),
            "file_email_ratio": frame["file_copy_count"]
            / (frame["email_count"] + 1.0),
            "http_logon_ratio": frame["http_count"]
            / (frame["logon_count"] + 1.0),
            "external_http_ratio": frame["external_http_count"]
            / (frame["http_count"] + 1.0),
            "archive_file_ratio": frame["archive_copy_count"]
            / (frame["file_copy_count"] + 1.0),
        }
    )

    def robust_channel_scores(channel_frame: pd.DataFrame) -> np.ndarray:
        values = channel_frame.to_numpy(dtype=np.float64).reshape(
            active_mask.shape[0],
            active_mask.shape[1],
            -1,
        )
        scores = np.full(active_mask.shape, np.nan, dtype=np.float32)
        for user_index in range(values.shape[0]):
            train_indices = np.flatnonzero(
                active_mask[user_index, : train_end + 1]
                & np.isfinite(
                    values[user_index, : train_end + 1]
                ).all(axis=1)
            )
            if len(train_indices) == 0:
                continue
            train_rows = values[user_index, train_indices]
            median = np.median(train_rows, axis=0)
            mad = np.median(np.abs(train_rows - median), axis=0)
            scale = 1.4826 * mad
            std = np.std(train_rows, axis=0)
            scale = np.where(
                scale >= 1e-8,
                scale,
                np.where(std >= 1e-8, std, 1.0),
            )
            active_indices = np.flatnonzero(active_mask[user_index])
            z = np.abs(
                (values[user_index, active_indices] - median)
                / scale
            )
            top_count = min(3, z.shape[1])
            scores[user_index, active_indices] = np.sort(
                z, axis=1
            )[:, -top_count:].mean(axis=1)
        return scores

    return (
        robust_channel_scores(midas),
        robust_channel_scores(relationship),
    )


def dynamic_daily(
    data: dict,
    quantile: np.ndarray,
) -> np.ndarray:
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        quantile.astype(np.float32),
        data["active_mask"],
    )
    return daily_percentile(
        dynamic, data["active_mask"]
    ).astype(np.float32)


def evaluate(
    data: dict,
    signal: np.ndarray,
    auxiliary: np.ndarray,
    method: str,
    *,
    external: bool,
) -> pd.DataFrame:
    alerts = build_unique_quota_alerts(
        signal,
        auxiliary,
        data["active_mask"],
        K,
        Q_LIMIT,
    )
    result = evaluate_alert_system(
        data, alerts, method, K, external=external
    )
    splits = ["external_test"] if external else ["validation", "test"]
    rows = []
    for split in splits:
        rows.append(
            {
                "method": method,
                "split": split,
                **compact_metric(
                    result, None if external else split
                ),
            }
        )
    return pd.DataFrame(rows)


def add_utilities(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for profile, weights in PROFILES.items():
        frame[profile] = sum(
            frame[column] * weight
            for column, weight in weights.items()
        )
    return frame


def main() -> None:
    channel = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    aligned = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    r42 = load_r42()
    r52 = load_r52()
    blocks = [
        {
            "name": "r42_original",
            "data": r42,
            "feature_file": (
                ROOT / "processed_r42" / "r4.2_user_day_features.csv"
            ),
            "train_end": r42["train_end"],
            "external": False,
        },
        {
            "name": "r42_retrained",
            "data": r42,
            "feature_file": (
                ROOT / "processed_r42" / "r4.2_user_day_features.csv"
            ),
            "train_end": r42["train_end"],
            "external": False,
        },
        {
            "name": "r52_external",
            "data": r52,
            "feature_file": (
                ROOT / "processed_r52" / "r5.2_user_day_features.csv"
            ),
            "train_end": r52["test_start"] - 1,
            "external": True,
        },
    ]
    frames = []
    saved = {}
    for block in blocks:
        name = str(block["name"])
        data = block["data"]
        raw = load_raw_features(
            block["feature_file"], data["users"], data["dates"]
        )
        midas_raw, relationship_raw = build_proxy_scores(
            raw, data["active_mask"], int(block["train_end"])
        )
        midas_q = per_user_quantile_scores(
            midas_raw, data["active_mask"], int(block["train_end"])
        )
        relationship_q = per_user_quantile_scores(
            relationship_raw,
            data["active_mask"],
            int(block["train_end"]),
        )
        v2_q = channel[
            f"{name}_triple_mean_quantile"
        ].astype(np.float32)
        auxiliary = aligned[
            f"{name}_v2_aligned_dual_view_iforest"
        ]

        plus_midas_q = (v2_q * 3.0 + midas_q) / 4.0
        plus_relationship_q = (
            v2_q * 3.0 + relationship_q
        ) / 4.0
        plus_both_q = (
            v2_q * 3.0 + midas_q + relationship_q
        ) / 5.0
        variants = {
            "v2_base": channel[
                f"{name}_triple_mean_iforest_signal"
            ],
            "midas_only": dynamic_daily(data, midas_q),
            "relationship_only": dynamic_daily(
                data, relationship_q
            ),
            "v2_plus_midas": dynamic_daily(data, plus_midas_q),
            "v2_plus_relationship": dynamic_daily(
                data, plus_relationship_q
            ),
            "v2_plus_both": dynamic_daily(data, plus_both_q),
        }
        for method, signal in variants.items():
            frame = evaluate(
                data,
                signal,
                auxiliary,
                method,
                external=bool(block["external"]),
            )
            frame.insert(0, "data_block", name)
            frames.append(frame)
            saved[f"{name}_{method}_signal"] = signal.astype(np.float32)
        saved[f"{name}_midas_quantile"] = midas_q
        saved[f"{name}_relationship_quantile"] = relationship_q

    metrics = add_utilities(
        pd.concat(frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "p1_proxy_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        100
        * metrics.groupby("method")["balanced"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )
    summary.to_csv(
        OUT / "p1_proxy_mean.csv",
        header=["mean_balanced_utility_pct"],
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "p1_proxy_arrays.npz",
        **saved,
    )
    payload = {
        "primary_k": K,
        "labels_used_for_scores": False,
        "mean_utility": summary.to_dict(),
        "outputs": {
            "metrics": str(OUT / "p1_proxy_metrics.csv"),
            "summary": str(OUT / "p1_proxy_mean.csv"),
            "arrays": str(OUT / "p1_proxy_arrays.npz"),
        },
    }
    (OUT / "p1_proxy_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(), flush=True)
    print("\nBlock metrics", flush=True)
    print(
        metrics[
            [
                "data_block",
                "split",
                "method",
                "incident_hits",
                "incident_total",
                "balanced",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
