from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
K_GRID = [3, 5, 10, 20, 30, 50]
PRIMARY_K = 10
ALPHA = 0.4
SEED = 20261109

sys.path.insert(0, str(ROOT))
from application_utility_analysis import PROFILES  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)
from risk_topk_r42 import compute_risk  # noqa: E402


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
    ts_signal: np.ndarray,
    auxiliary: np.ndarray,
    method: str,
    *,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        q = q_from_alpha(k, ALPHA)
        alerts = build_unique_quota_alerts(
            ts_signal,
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
            mean_regret=("regret", "mean"),
            worst_regret=("regret", "max"),
        )
        .sort_values("mean_rank")
    )


def main() -> None:
    dual = np.load(
        OUT / "ts2vec_dual_score_arrays.npz",
        allow_pickle=False,
    )
    channel = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    r42 = load_r42()
    r52 = load_r52()
    blocks = [
        {
            "name": "r42_original",
            "data": r42,
            "feature_file": (
                ROOT
                / "processed_r42"
                / "r4.2_user_day_features.csv"
            ),
            "external": False,
        },
        {
            "name": "r42_retrained",
            "data": r42,
            "feature_file": (
                ROOT
                / "processed_r42"
                / "r4.2_user_day_features.csv"
            ),
            "external": False,
        },
        {
            "name": "r52_external",
            "data": r52,
            "feature_file": (
                ROOT
                / "processed_r52"
                / "r5.2_user_day_features.csv"
            ),
            "external": True,
        },
    ]

    frames = []
    saved = {}
    diagnostics = []
    for block in blocks:
        name = str(block["name"])
        data = block["data"]
        triple_q = channel[
            f"{name}_triple_mean_quantile"
        ].astype(np.float32)
        triple_signal = channel[
            f"{name}_triple_mean_iforest_signal"
        ].astype(np.float32)
        v2_dynamic = compute_risk(
            data["users"],
            data["dates"],
            triple_q,
            data["active_mask"],
        )
        original, augmented = fit_view_scores(
            data,
            block["feature_file"],
            v2_dynamic,
            data["personal"],
            SEED,
        )
        aligned_auxiliary = np.maximum(original, augmented)
        old_auxiliary = dual[f"{name}_dual_view_iforest"]

        variants = {
            "v2_old_iforest": old_auxiliary,
            "v2_aligned_iforest": aligned_auxiliary,
        }
        for method, auxiliary in variants.items():
            frame = evaluate(
                data,
                triple_signal,
                auxiliary,
                method,
                external=bool(block["external"]),
            )
            frame.insert(0, "data_block", name)
            frame["external"] = int(bool(block["external"]))
            frames.append(frame)
        saved[f"{name}_v2_aligned_original_iforest"] = original
        saved[f"{name}_v2_aligned_augmented_iforest"] = augmented
        saved[f"{name}_v2_aligned_dual_view_iforest"] = aligned_auxiliary
        diagnostics.append(
            {
                "data_block": name,
                "old_new_aux_mean_abs_diff": float(
                    np.nanmean(
                        np.abs(aligned_auxiliary - old_auxiliary)
                    )
                ),
            }
        )

    metrics = add_utilities(
        pd.concat(frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "v2_iforest_alignment_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    k10 = metrics[metrics["k"] == PRIMARY_K]
    rankings = build_rankings(k10)
    stability = summarize_stability(rankings)
    stability.to_csv(
        OUT / "v2_iforest_alignment_k10_rank_stability.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "v2_iforest_alignment_arrays.npz",
        **saved,
    )
    summary = {
        "primary_k": PRIMARY_K,
        "q": q_from_alpha(PRIMARY_K, ALPHA),
        "seed": SEED,
        "diagnostics": diagnostics,
        "k10_stability": stability.to_dict(orient="records"),
        "outputs": {
            "metrics": str(
                OUT / "v2_iforest_alignment_metrics.csv"
            ),
            "k10_stability": str(
                OUT / "v2_iforest_alignment_k10_rank_stability.csv"
            ),
            "arrays": str(
                OUT / "v2_iforest_alignment_arrays.npz"
            ),
        },
    }
    (OUT / "v2_iforest_alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nK=10", flush=True)
    print(
        k10[
            [
                "data_block",
                "split",
                "method",
                "incident_hits",
                "incident_total",
                "event_recall_3d",
                "scenario_macro_recall_3d",
                "precision_at_k",
                "balanced",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
