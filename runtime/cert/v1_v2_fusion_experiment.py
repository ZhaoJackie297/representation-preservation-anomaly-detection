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

sys.path.insert(0, str(ROOT))
from application_utility_analysis import PROFILES  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    daily_percentile,
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
            "external": False,
        },
        {
            "name": "r42_retrained",
            "data": r42,
            "external": False,
        },
        {
            "name": "r52_external",
            "data": r52,
            "external": True,
        },
    ]

    frames = []
    saved = {}
    for block in blocks:
        name = str(block["name"])
        data = block["data"]
        auxiliary = dual[f"{name}_dual_view_iforest"]
        mask_q = dual[f"{name}_ts_mask_quantile"].astype(np.float32)
        latent_q = dual[f"{name}_ts_latent_quantile"].astype(np.float32)
        channel_q = channel[
            f"{name}_channel_top2_quantile"
        ].astype(np.float32)

        v1_q = (mask_q + latent_q) / 2.0
        v2_q = (mask_q + latent_q + channel_q) / 3.0
        score_fusion_q = (v1_q + v2_q) / 2.0

        v1_dynamic = compute_risk(
            data["users"], data["dates"], v1_q, data["active_mask"]
        )
        v1_signal = daily_percentile(
            v1_dynamic, data["active_mask"]
        ).astype(np.float32)
        v2_dynamic = compute_risk(
            data["users"], data["dates"], v2_q, data["active_mask"]
        )
        v2_signal = daily_percentile(
            v2_dynamic, data["active_mask"]
        ).astype(np.float32)
        score_fusion_dynamic = compute_risk(
            data["users"],
            data["dates"],
            score_fusion_q,
            data["active_mask"],
        )
        score_fusion_signal = daily_percentile(
            score_fusion_dynamic, data["active_mask"]
        ).astype(np.float32)
        signal_fusion = (v1_signal + v2_signal) / 2.0

        variants = {
            "v1_dual_iforest": v1_signal,
            "v2_triple_iforest": v2_signal,
            "v1_v2_score_fusion_iforest": score_fusion_signal,
            "v1_v2_signal_fusion_iforest": signal_fusion,
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
            frames.append(frame)
            saved[f"{name}_{method}_signal"] = signal.astype(np.float32)

    metrics = add_utilities(
        pd.concat(frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "v1_v2_fusion_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    k10 = metrics[metrics["k"] == PRIMARY_K]
    summary = (
        k10.groupby("method", as_index=False)
        .agg(
            mean_utility=("balanced", "mean"),
            mean_recall=("event_recall_3d", "mean"),
            mean_scenario=("scenario_macro_recall_3d", "mean"),
            mean_precision=("precision_at_k", "mean"),
        )
        .sort_values("mean_utility", ascending=False)
    )
    summary.to_csv(
        OUT / "v1_v2_fusion_k10_summary.csv",
        index=False,
        encoding="utf-8",
    )
    by_k = (
        metrics.groupby(["k", "method"])["balanced"]
        .mean()
        .unstack()
    )
    by_k.to_csv(
        OUT / "v1_v2_fusion_budget_comparison.csv",
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "v1_v2_fusion_arrays.npz",
        **saved,
    )
    payload = {
        "primary_k": PRIMARY_K,
        "q": q_from_alpha(PRIMARY_K, ALPHA),
        "score_fusion_definition": (
            "0.5*V1 + 0.5*V2 = "
            "(2*mask + 2*latent + channel)/5"
        ),
        "signal_fusion_definition": (
            "0.5*V1_daily_percentile + "
            "0.5*V2_daily_percentile"
        ),
        "labels_used": False,
        "outputs": {
            "metrics": str(OUT / "v1_v2_fusion_metrics.csv"),
            "k10_summary": str(
                OUT / "v1_v2_fusion_k10_summary.csv"
            ),
            "budget_comparison": str(
                OUT / "v1_v2_fusion_budget_comparison.csv"
            ),
            "arrays": str(OUT / "v1_v2_fusion_arrays.npz"),
        },
    }
    (OUT / "v1_v2_fusion_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("K=10 summary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nBudget comparison", flush=True)
    print((100 * by_k).round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
