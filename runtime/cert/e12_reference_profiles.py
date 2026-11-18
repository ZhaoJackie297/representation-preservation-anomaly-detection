from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
K_GRID = [3, 5, 10, 20, 30, 50]
PRIMARY_K = 10
ALPHA = 0.4
MIN_PEER_COUNT = 10
PROFILE_HIGH_PERCENTILE = 0.90

FEATURES = [
    "logon_count",
    "logoff_count",
    "first_logon_minute",
    "last_logoff_minute",
    "active_span_minutes",
    "after_hours_logon_count",
    "other_pc_logon_count",
    "usb_connect_count",
    "usb_disconnect_count",
    "file_copy_count",
    "archive_copy_count",
    "executable_copy_count",
    "unique_file_extensions",
    "email_count",
    "email_recipient_count",
    "external_recipient_count",
    "email_attachment_count",
    "email_size_bytes",
    "after_hours_email_count",
    "http_count",
    "after_hours_http_count",
    "external_http_count",
    "unique_http_domains",
]
LOG1P_FEATURES = FEATURES[:2] + FEATURES[5:]

sys.path.insert(0, str(ROOT))
from application_utility_analysis import PROFILES  # noqa: E402
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from production_protocol_r42 import label_matrix  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    daily_percentile,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)
from risk_topk_r42 import compute_risk  # noqa: E402


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    return median, 1.4826 * mad


def build_group_raw(
    values: np.ndarray,
    active_mask: np.ndarray,
    functional_units: np.ndarray,
    roles: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_active = active_mask[:, : train_end + 1]
    global_median, global_scale = robust_location_scale(
        values[:, : train_end + 1][train_active]
    )
    global_scale = np.where(
        np.isfinite(global_scale) & (global_scale >= 1e-8),
        global_scale,
        1.0,
    )
    raw = np.full(values.shape[:2], np.nan, dtype=np.float32)
    levels = np.zeros(values.shape[:2], dtype=np.int8)
    peer_counts = np.zeros(values.shape[:2], dtype=np.int16)

    for time_index in range(values.shape[1]):
        active_users = np.flatnonzero(active_mask[:, time_index])
        if len(active_users) == 0:
            continue
        day_median, day_scale = robust_location_scale(
            values[active_users, time_index]
        )
        day_median = np.where(np.isfinite(day_median), day_median, 0.0)
        day_scale = np.where(
            np.isfinite(day_scale) & (day_scale >= 1e-8),
            day_scale,
            global_scale,
        )
        assigned = np.zeros(values.shape[0], dtype=bool)

        def assign(labels: np.ndarray, level: int) -> None:
            active_labels = labels[active_users]
            for label in np.unique(active_labels):
                if str(label) in {"Unknown", "nan", ""}:
                    continue
                group_users = active_users[active_labels == label]
                group_users = group_users[~assigned[group_users]]
                if len(group_users) < MIN_PEER_COUNT:
                    continue
                group_values = values[group_users, time_index]
                median = np.nanmedian(group_values, axis=0)
                mad = np.nanmedian(
                    np.abs(group_values - median), axis=0
                )
                scale = 1.4826 * mad
                scale = np.where(
                    np.isfinite(scale) & (scale >= 1e-8),
                    scale,
                    global_scale,
                )
                z = np.abs((group_values - median) / scale)
                z = np.clip(z, 0.0, 100.0)
                raw[group_users, time_index] = np.sort(
                    z, axis=1
                )[:, -3:].mean(axis=1)
                levels[group_users, time_index] = level
                peer_counts[group_users, time_index] = len(group_users)
                assigned[group_users] = True

        assign(functional_units[:, time_index], 1)
        assign(roles[:, time_index], 2)
        remaining = active_users[~assigned[active_users]]
        if len(remaining):
            z = np.abs(
                (values[remaining, time_index] - day_median)
                / day_scale
            )
            z = np.clip(z, 0.0, 100.0)
            raw[remaining, time_index] = np.sort(
                z, axis=1
            )[:, -3:].mean(axis=1)
            levels[remaining, time_index] = 3
            peer_counts[remaining, time_index] = len(active_users)
    return raw, levels, peer_counts


def load_group_arrays(
    feature_csv: Path,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(feature_csv, parse_dates=["date"])
    index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    aligned = frame.set_index(["user_id", "date"]).reindex(index)
    values = aligned[FEATURES].to_numpy(dtype=np.float64).reshape(
        len(users), len(dates), len(FEATURES)
    )
    for column_index, feature in enumerate(FEATURES):
        if feature in LOG1P_FEATURES:
            finite = np.isfinite(values[:, :, column_index])
            values[:, :, column_index][finite] = np.log1p(
                values[:, :, column_index][finite]
            )
    functional = aligned["functional_unit"].fillna(
        "Unknown"
    ).to_numpy().reshape(len(users), len(dates))
    roles = aligned["role"].fillna("Unknown").to_numpy().reshape(
        len(users), len(dates)
    )
    return values.astype(np.float32), functional, roles


def dynamic_daily(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    quantile: np.ndarray,
) -> np.ndarray:
    dynamic = compute_risk(
        users, dates, quantile.astype(np.float32), active_mask
    )
    return daily_percentile(dynamic, active_mask).astype(np.float32)


def add_utilities(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for profile, weights in PROFILES.items():
        frame[profile] = sum(
            frame[column] * weight
            for column, weight in weights.items()
        )
    return frame


def evaluate(
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


def profile_label(
    personal_high: bool,
    group_high: bool,
    context_high: bool,
) -> str:
    if personal_high and group_high:
        return "个人与群体同时高"
    if personal_high:
        return "仅个人高"
    if group_high:
        return "仅群体高"
    if context_high:
        return "仅上下文高"
    return "无强参照"


def main() -> None:
    dual = np.load(
        OUT / "ts2vec_dual_score_arrays.npz",
        allow_pickle=False,
    )
    aligned_iforest = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    channel = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    r42 = load_r42()
    r52 = load_r52()
    r42_labels = label_matrix(r42["users"], r42["dates"])
    r42_values, r42_functional, r42_roles = load_group_arrays(
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        r42["users"],
        r42["dates"],
    )
    r52_values, r52_functional, r52_roles = load_group_arrays(
        ROOT / "processed_r52" / "r5.2_user_day_features.csv",
        r52["users"],
        r52["dates"],
    )
    shared_groups = {}
    for name, data, values, functional, roles, train_end in (
        (
            "r42",
            r42,
            r42_values,
            r42_functional,
            r42_roles,
            r42["train_end"],
        ),
        (
            "r52",
            r52,
            r52_values,
            r52_functional,
            r52_roles,
            r52["test_start"] - 1,
        ),
    ):
        raw, levels, counts = build_group_raw(
            values,
            data["active_mask"],
            functional,
            roles,
            int(train_end),
        )
        shared_groups[name] = (raw, levels, counts)

    blocks = [
        {
            "name": "r42_original",
            "group_key": "r42",
            "data": r42,
            "labels": r42_labels,
            "external": False,
            "train_end": r42["train_end"],
        },
        {
            "name": "r42_retrained",
            "group_key": "r42",
            "data": r42,
            "labels": r42_labels,
            "external": False,
            "train_end": r42["train_end"],
        },
        {
            "name": "r52_external",
            "group_key": "r52",
            "data": r52,
            "labels": r52["labels"],
            "external": True,
            "train_end": r52["test_start"] - 1,
        },
    ]

    metric_frames = []
    profile_rows = []
    saved = {}
    diagnostics = []
    for block in blocks:
        name = str(block["name"])
        data = block["data"]
        active = data["active_mask"]
        users = data["users"]
        dates = data["dates"]
        train_end = int(block["train_end"])
        auxiliary = aligned_iforest[
            f"{name}_v2_aligned_dual_view_iforest"
        ]
        mask_q = dual[f"{name}_ts_mask_quantile"].astype(np.float32)
        latent_q = dual[f"{name}_ts_latent_quantile"].astype(np.float32)
        channel_q = channel[
            f"{name}_channel_top2_quantile"
        ].astype(np.float32)
        v2_q = (mask_q + latent_q + channel_q) / 3.0
        v2_signal = channel[
            f"{name}_triple_mean_iforest_signal"
        ].astype(np.float32)

        group_raw, group_levels, group_counts = shared_groups[
            block["group_key"]
        ]
        group_q = per_user_quantile_scores(
            group_raw, active, train_end
        )
        group_daily = dynamic_daily(
            users, dates, active, group_q
        )
        mask_daily = dynamic_daily(
            users, dates, active, mask_q
        )
        latent_daily = dynamic_daily(
            users, dates, active, latent_q
        )
        channel_daily = dynamic_daily(
            users, dates, active, channel_q
        )

        equal_q = (mask_q + latent_q + channel_q + group_q) / 4.0
        equal_dynamic = compute_risk(
            users, dates, equal_q, active
        )
        equal_signal = daily_percentile(
            equal_dynamic, active
        ).astype(np.float32)

        personal_high = latent_daily >= PROFILE_HIGH_PERCENTILE
        group_high = group_daily >= PROFILE_HIGH_PERCENTILE
        context_daily = np.maximum(mask_daily, channel_daily)
        context_high = context_daily >= PROFILE_HIGH_PERCENTILE
        gate = group_high & (personal_high | context_high)
        conditional_q = v2_q.copy()
        conditional_q[gate] = np.maximum(
            conditional_q[gate], group_q[gate]
        )
        conditional_dynamic = compute_risk(
            users, dates, conditional_q, active
        )
        conditional_signal = daily_percentile(
            conditional_dynamic, active
        ).astype(np.float32)

        variants = {
            "v2_triple_iforest": v2_signal,
            "v2_group_equal_iforest": equal_signal,
            "v2_group_conditional_iforest": conditional_signal,
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
            frame["external"] = int(bool(block["external"]))
            metric_frames.append(frame)
            saved[f"{name}_{method}_signal"] = signal.astype(np.float32)

        base_alerts = build_unique_quota_alerts(
            v2_signal,
            auxiliary,
            active,
            PRIMARY_K,
            q_from_alpha(PRIMARY_K, ALPHA),
        )
        for time_index in range(len(dates)):
            for user_index in np.flatnonzero(
                base_alerts[:, time_index]
            ):
                profile = profile_label(
                    bool(personal_high[user_index, time_index]),
                    bool(group_high[user_index, time_index]),
                    bool(context_high[user_index, time_index]),
                )
                profile_rows.append(
                    {
                        "data_block": name,
                        "date": dates[time_index].strftime("%Y-%m-%d"),
                        "user_id": str(users[user_index]),
                        "profile": profile,
                        "peer_level": int(
                            group_levels[user_index, time_index]
                        ),
                        "peer_count": int(
                            group_counts[user_index, time_index]
                        ),
                        "group_daily": float(
                            group_daily[user_index, time_index]
                        ),
                        "personal_daily": float(
                            latent_daily[user_index, time_index]
                        ),
                        "context_daily": float(
                            context_high[user_index, time_index]
                        ),
                        "audit_malicious_account_day": int(
                            block["labels"][user_index, time_index]
                        ),
                    }
                )

        saved[f"{name}_group_quantile"] = group_q.astype(np.float32)
        saved[f"{name}_group_daily"] = group_daily
        saved[f"{name}_personal_daily"] = latent_daily
        saved[f"{name}_context_daily"] = context_daily
        saved[f"{name}_group_gate"] = gate.astype(np.uint8)
        saved[f"{name}_conditional_quantile"] = conditional_q
        diagnostics.append(
            {
                "data_block": name,
                "peer_level_rows": {
                    str(level): int((group_levels == level).sum())
                    for level in sorted(np.unique(group_levels))
                },
                "group_gate_rows": int(gate.sum()),
                "group_gate_eval_rows": int(
                    gate[
                        :,
                        train_end + 1 :
                    ].sum()
                ),
            }
        )

    metrics = add_utilities(
        pd.concat(metric_frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "e12_reference_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    profile_frame = pd.DataFrame(profile_rows)
    profile_summary = (
        profile_frame.groupby(
            ["data_block", "profile"],
            as_index=False,
        )
        .agg(
            selected_alerts=("user_id", "size"),
            malicious_account_days=(
                "audit_malicious_account_day",
                "sum",
            ),
            mean_group_daily=("group_daily", "mean"),
            mean_personal_daily=("personal_daily", "mean"),
        )
    )
    profile_summary["audit_density"] = (
        profile_summary["malicious_account_days"]
        / profile_summary["selected_alerts"]
    )
    profile_summary.to_csv(
        OUT / "e12_reference_profile_summary.csv",
        index=False,
        encoding="utf-8",
    )
    k10 = metrics[metrics["k"] == PRIMARY_K]
    k10_summary = (
        k10.groupby("method", as_index=False)
        .agg(
            mean_utility=("balanced", "mean"),
            mean_recall=("event_recall_3d", "mean"),
            mean_scenario=("scenario_macro_recall_3d", "mean"),
            mean_precision=("precision_at_k", "mean"),
        )
        .sort_values("mean_utility", ascending=False)
    )
    k10_summary.to_csv(
        OUT / "e12_reference_k10_summary.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "e12_reference_arrays.npz",
        **saved,
    )
    summary = {
        "primary_k": PRIMARY_K,
        "q": q_from_alpha(PRIMARY_K, ALPHA),
        "profile_high_percentile": PROFILE_HIGH_PERCENTILE,
        "peer_rule": [
            "functional_unit with >=10 active peers",
            "role with >=10 active peers",
            "all active users fallback",
        ],
        "labels_used_for_scores": False,
        "labels_used_for_gate": False,
        "diagnostics": diagnostics,
        "outputs": {
            "metrics": str(OUT / "e12_reference_metrics.csv"),
            "profiles": str(
                OUT / "e12_reference_profile_summary.csv"
            ),
            "k10_summary": str(
                OUT / "e12_reference_k10_summary.csv"
            ),
            "arrays": str(OUT / "e12_reference_arrays.npz"),
        },
    }
    (OUT / "e12_reference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nK=10 summary", flush=True)
    print(k10_summary.to_string(index=False), flush=True)
    print("\nProfile summary", flush=True)
    print(profile_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
