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

sys.path.insert(0, str(ROOT))
from production_protocol_r42 import (  # noqa: E402
    label_matrix,
)
from quota_relationship_sim import (  # noqa: E402
    evaluate_alert_system,
    load_r42,
    load_r52,
)


def select_daily_queue(
    ts_signal: np.ndarray,
    auxiliary_signal: np.ndarray,
    original_iforest: np.ndarray,
    augmented_iforest: np.ndarray,
    mask_quantile: np.ndarray,
    latent_quantile: np.ndarray,
    active_mask: np.ndarray,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    labels: np.ndarray,
    *,
    start_index: int,
    evaluation_ranges: dict[str, tuple[int, int]],
    data_block: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    alerts = np.zeros(ts_signal.shape, dtype=np.uint8)
    rows = []
    last_alert = np.full(ts_signal.shape[0], -10_000, dtype=np.int32)
    evaluation_index = {
        time_index: split
        for split, (start, end) in evaluation_ranges.items()
        for time_index in range(start, end + 1)
    }

    for time_index in range(start_index, ts_signal.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= COOLDOWN_DAYS
        )
        candidates = np.flatnonzero(available)
        if len(candidates) == 0:
            continue

        ts_order = candidates[
            np.lexsort(
                (
                    candidates,
                    -ts_signal[candidates, time_index],
                )
            )
        ]
        aux_order = candidates[
            np.lexsort(
                (
                    candidates,
                    -auxiliary_signal[candidates, time_index],
                )
            )
        ]
        ts_rank = {
            int(user_index): rank
            for rank, user_index in enumerate(ts_order, start=1)
        }
        aux_rank = {
            int(user_index): rank
            for rank, user_index in enumerate(aux_order, start=1)
        }
        ts_top_k = set(int(user) for user in ts_order[:K])

        selected: list[tuple[int, str]] = [
            (int(user), "ts_main")
            for user in ts_order[: K - Q_LIMIT]
        ]
        selected_users = {user for user, _ in selected}
        for user_index in aux_order:
            user_index = int(user_index)
            if (
                user_index in selected_users
                or user_index in ts_top_k
            ):
                continue
            selected.append((user_index, "iforest_rescue"))
            selected_users.add(user_index)
            if len(selected) >= K:
                break
        if len(selected) < K:
            for user_index in ts_order:
                user_index = int(user_index)
                if user_index in selected_users:
                    continue
                selected.append((user_index, "ts_fallback"))
                selected_users.add(user_index)
                if len(selected) >= K:
                    break

        selected = selected[:K]
        selected_array = np.asarray(
            [user for user, _ in selected], dtype=np.int64
        )
        alerts[selected_array, time_index] = 1
        last_alert[selected_array] = time_index

        split = evaluation_index.get(time_index)
        if split is None:
            continue
        for rank, (user_index, source) in enumerate(selected, start=1):
            mask_value = float(mask_quantile[user_index, time_index])
            latent_value = float(
                latent_quantile[user_index, time_index]
            )
            if abs(mask_value - latent_value) < 0.10:
                dominant_ts = "双证据接近"
            elif mask_value > latent_value:
                dominant_ts = "遮罩差异主导"
            else:
                dominant_ts = "潜空间偏离主导"

            original_value = float(
                original_iforest[user_index, time_index]
            )
            augmented_value = float(
                augmented_iforest[user_index, time_index]
            )
            dominant_iforest = (
                "原始IForest主导"
                if original_value >= augmented_value
                else "增强IForest主导"
            )
            if source == "iforest_rescue":
                reason = (
                    "TS Top-K未覆盖，但双视角IForest排名靠前，"
                    "作为独有候选进入补位名额。"
                )
            elif source == "ts_fallback":
                reason = (
                    "IForest独有候选不足，剩余名额退回TS2Vec主路。"
                )
            else:
                reason = (
                    "位于TS2Vec主路前K-q名。"
                    f"{dominant_ts}。"
                )

            rows.append(
                {
                    "data_block": data_block,
                    "split": split,
                    "date": dates[time_index].strftime("%Y-%m-%d"),
                    "queue_rank": rank,
                    "user_id": str(users[user_index]),
                    "source": source,
                    "ts_rank": ts_rank[user_index],
                    "auxiliary_rank": aux_rank[user_index],
                    "ts_signal": float(
                        ts_signal[user_index, time_index]
                    ),
                    "ts_mask_quantile": mask_value,
                    "ts_latent_quantile": latent_value,
                    "ts_dual_mean_quantile": (
                        mask_value + latent_value
                    )
                    / 2.0,
                    "ts_evidence_gap": abs(mask_value - latent_value),
                    "dominant_ts_evidence": dominant_ts,
                    "iforest_original_signal": original_value,
                    "iforest_augmented_signal": augmented_value,
                    "iforest_view_signal": float(
                        auxiliary_signal[user_index, time_index]
                    ),
                    "dominant_iforest_evidence": dominant_iforest,
                    "reason": reason,
                    "audit_malicious_account_day": int(
                        labels[user_index, time_index]
                    ),
                }
            )
    return alerts, pd.DataFrame(rows)


def event_start_series(incidents: pd.DataFrame) -> pd.Series:
    if "start_date" in incidents.columns:
        return pd.to_datetime(incidents["start_date"]).dt.normalize()
    if "first_observed_date" in incidents.columns:
        return pd.to_datetime(
            incidents["first_observed_date"]
        ).dt.normalize()
    raise KeyError("No incident start-date column found.")


def extract_hit_ids(
    result: dict,
    split: str,
    external: bool,
) -> set[str]:
    section = (
        result["external_test"]
        if external
        else result["splits"][split]
    )
    if external:
        return set(section["horizons"]["3"]["incident_ids"])
    return set(section["incident_ids"])


def monthly_stability(
    data_block: str,
    data: dict,
    alerts: np.ndarray,
    labels: np.ndarray,
    *,
    external: bool,
    splits: list[str],
) -> pd.DataFrame:
    result = evaluate_alert_system(
        data,
        alerts,
        "frozen_dual_score_dual_iforest",
        K,
        external=external,
    )
    incidents = data["incidents"].copy()
    incidents["event_start"] = event_start_series(incidents)
    months = pd.period_range(
        data["dates"][0], data["dates"][-1], freq="M"
    )
    rows = []
    for split in splits:
        hit_ids = extract_hit_ids(result, split, external)
        if external:
            split_start = data["dates"][data["test_start"]].normalize()
            split_end = data["dates"][-1].normalize()
        elif split == "validation":
            split_start = data["dates"][
                data["train_end"] + 1
            ].normalize()
            split_end = data["dates"][data["val_end"]].normalize()
        else:
            split_start = data["dates"][
                data["val_end"] + 1
            ].normalize()
            split_end = data["dates"][-1].normalize()

        for month in months:
            month_start = max(
                month.start_time.normalize(),
                pd.Timestamp(split_start),
            )
            month_end = min(
                month.end_time.normalize(),
                pd.Timestamp(split_end),
            )
            if month_start > month_end:
                continue
            month_incidents = incidents[
                (incidents["event_start"] >= month_start)
                & (incidents["event_start"] <= month_end)
            ]
            if month_incidents.empty:
                continue
            incident_ids = set(
                month_incidents["incident_id"].astype(str)
            )
            hits = incident_ids & hit_ids
            recall = len(hits) / len(incident_ids)
            scenario_recalls = []
            for _, group in month_incidents.groupby(
                "scenario", sort=True
            ):
                group_ids = set(group["incident_id"].astype(str))
                scenario_recalls.append(
                    len(group_ids & hits) / len(group_ids)
                )
            scenario_macro = float(np.mean(scenario_recalls))

            date_mask = (
                (data["dates"] >= month_start)
                & (data["dates"] <= month_end)
            )
            month_alerts = alerts[:, date_mask]
            month_labels = labels[:, date_mask]
            alert_slots = int(month_alerts.sum())
            true_slots = int((month_alerts & month_labels).sum())
            precision = (
                true_slots / alert_slots if alert_slots else 0.0
            )
            utility = (
                0.50 * recall
                + 0.25 * scenario_macro
                + 0.25 * precision
            )
            rows.append(
                {
                    "data_block": data_block,
                    "split": split,
                    "month": str(month),
                    "incident_total": len(incident_ids),
                    "incident_hits": len(hits),
                    "event_recall_3d": recall,
                    "scenario_macro_recall_3d": scenario_macro,
                    "precision_at_k": precision,
                    "balanced_utility": utility,
                    "alert_slots": alert_slots,
                    "true_alert_slots": true_slots,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    arrays = np.load(
        OUT / "ts2vec_dual_score_arrays.npz",
        allow_pickle=False,
    )
    view_scores = np.load(
        OUT / "iforest_view_ensemble_scores.npz",
        allow_pickle=False,
    )
    r42 = load_r42()
    r52 = load_r52()
    r42_labels = label_matrix(r42["users"], r42["dates"])

    blocks = [
        {
            "name": "r42_original",
            "data": r42,
            "data_for_eval": r42,
            "labels": r42_labels,
            "external": False,
            "splits": ["validation", "test"],
            "start_index": 0,
            "evaluation_ranges": {
                "validation": (
                    r42["train_end"] + 1,
                    r42["val_end"],
                ),
                "test": (
                    r42["val_end"] + 1,
                    len(r42["dates"]) - 1,
                ),
            },
            "original_iforest": view_scores["r42_original"],
            "augmented_iforest": view_scores["r42_augmented"],
        },
        {
            "name": "r42_retrained",
            "data": r42,
            "data_for_eval": r42,
            "labels": r42_labels,
            "external": False,
            "splits": ["validation", "test"],
            "start_index": 0,
            "evaluation_ranges": {
                "validation": (
                    r42["train_end"] + 1,
                    r42["val_end"],
                ),
                "test": (
                    r42["val_end"] + 1,
                    len(r42["dates"]) - 1,
                ),
            },
            "original_iforest": view_scores["retrained_original"],
            "augmented_iforest": view_scores["retrained_augmented"],
        },
        {
            "name": "r52_external",
            "data": r52,
            "data_for_eval": r52,
            "labels": r52["labels"],
            "external": True,
            "splits": ["external_test"],
            "start_index": r52["test_start"],
            "evaluation_ranges": {
                "external_test": (
                    r52["test_start"],
                    len(r52["dates"]) - 1,
                )
            },
            "original_iforest": view_scores["r52_original"],
            "augmented_iforest": view_scores["r52_augmented"],
        },
    ]

    evidence_frames = []
    monthly_frames = []
    summaries = []
    for block in blocks:
        name = str(block["name"])
        data = block["data"]
        ts_signal = arrays[f"{name}_ts_dual_mean_signal"]
        auxiliary = arrays[f"{name}_dual_view_iforest"]
        original_iforest = block["original_iforest"]
        augmented_iforest = block["augmented_iforest"]
        mask_quantile = arrays[f"{name}_ts_mask_quantile"]
        latent_quantile = arrays[f"{name}_ts_latent_quantile"]
        alerts, evidence = select_daily_queue(
            ts_signal,
            auxiliary,
            original_iforest,
            augmented_iforest,
            mask_quantile,
            latent_quantile,
            data["active_mask"],
            block["data_for_eval"]["users"],
            data["dates"],
            block["labels"],
            start_index=int(block["start_index"]),
            evaluation_ranges=block["evaluation_ranges"],
            data_block=name,
        )
        evidence_frames.append(evidence)
        monthly_frames.append(
            monthly_stability(
                name,
                data,
                alerts,
                block["labels"],
                external=bool(block["external"]),
                splits=list(block["splits"]),
            )
        )

        source_counts = (
            evidence["source"].value_counts().to_dict()
            if not evidence.empty
            else {}
        )
        summaries.append(
            {
                "data_block": name,
                "selected_alerts": int(len(evidence)),
                "source_counts": source_counts,
                "audit_malicious_alerts": int(
                    evidence["audit_malicious_account_day"].sum()
                )
                if not evidence.empty
                else 0,
            }
        )

    evidence = pd.concat(evidence_frames, ignore_index=True)
    evidence.to_csv(
        OUT / "frozen_model_evidence_cards_k10.csv",
        index=False,
        encoding="utf-8",
    )
    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly.to_csv(
        OUT / "frozen_model_monthly_stability.csv",
        index=False,
        encoding="utf-8",
    )

    # Save a small audit sample without changing model selection.
    samples = []
    for block in blocks:
        block_name = str(block["name"])
        subset = evidence[evidence["data_block"] == block_name]
        rescue = subset[subset["source"] == "iforest_rescue"].head(4)
        main = subset[subset["source"] == "ts_main"].head(4)
        samples.append(pd.concat([main, rescue], ignore_index=True))
    sample = pd.concat(samples, ignore_index=True)
    sample.to_csv(
        OUT / "frozen_model_evidence_card_samples.csv",
        index=False,
        encoding="utf-8",
    )

    summary = {
        "model": "TS2Vec dual-mean + dual-view IForest conditional complement",
        "primary_k": K,
        "q": Q_LIMIT,
        "cooldown_days": COOLDOWN_DAYS,
        "labels_used_for_alert_selection": False,
        "audit_label_note": (
            "audit_malicious_account_day is included only for offline "
            "validation and is not used by queue construction."
        ),
        "blocks": summaries,
        "monthly_months_with_events": int(len(monthly)),
        "monthly_mean_balanced_utility": float(
            monthly["balanced_utility"].mean()
        ),
        "monthly_min_balanced_utility": float(
            monthly["balanced_utility"].min()
        ),
        "monthly_max_balanced_utility": float(
            monthly["balanced_utility"].max()
        ),
        "outputs": {
            "evidence_cards": str(
                OUT / "frozen_model_evidence_cards_k10.csv"
            ),
            "samples": str(
                OUT / "frozen_model_evidence_card_samples.csv"
            ),
            "monthly_stability": str(
                OUT / "frozen_model_monthly_stability.csv"
            ),
        },
    }
    (OUT / "frozen_model_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nMonthly stability", flush=True)
    print(monthly.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
