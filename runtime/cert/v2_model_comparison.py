from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
K_GRID = [3, 5, 10, 20, 30, 50]

sys.path.insert(0, str(ROOT))
from application_utility_analysis import PROFILES  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from production_protocol_r42 import build_alerts  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)


def add_utilities(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for profile, weights in PROFILES.items():
        frame[profile] = sum(
            frame[column] * weight
            for column, weight in weights.items()
        )
    return frame


def evaluate_signal(
    data: dict,
    signal,
    auxiliary,
    method: str,
    *,
    external: bool,
    use_iforest: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        q = q_from_alpha(k, 0.4) if use_iforest else 0
        alerts = (
            build_unique_quota_alerts(
                signal,
                auxiliary,
                data["active_mask"],
                k,
                q,
            )
            if use_iforest
            else build_alerts(signal, data["active_mask"], k, 7)
        )
        result = evaluate_alert_system(
            data, alerts, method, k, external=external
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
    r42 = load_r42()
    r52 = load_r52()
    aligned = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
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
    baseline_frames = []
    for block in blocks:
        name = str(block["name"])
        data = block["data"]
        auxiliary = aligned[
            f"{name}_v2_aligned_dual_view_iforest"
        ]
        for method, use_iforest in (
            ("zscore_complete", False),
            ("zscore_aligned_iforest", True),
        ):
            frame = evaluate_signal(
                data,
                data["personal"],
                auxiliary,
                method,
                external=bool(block["external"]),
                use_iforest=use_iforest,
            )
            frame.insert(0, "data_block", name)
            frame["external"] = int(bool(block["external"]))
            baseline_frames.append(frame)
    baselines = pd.concat(baseline_frames, ignore_index=True)

    source_specs = [
        (
            "ts2vec_dual_score_metrics.csv",
            {
                "iforest_complete": "original_iforest",
                "ts_mask_complete": "mask_only_ts",
                "ts_dual_mean_complete": "v1_complete_ts",
                "ts_dual_mean_dual_iforest": "v1_dual_iforest",
            },
        ),
        (
            "channel_evidence_metrics.csv",
            {
                "iforest_complete": "dual_view_iforest",
                "channel_top2_iforest": "channel_only_iforest",
                "triple_mean_complete": "v2_complete_ts",
                "triple_mean_iforest": "v2_old_aux_iforest",
            },
        ),
        (
            "v2_pipeline_metrics.csv",
            {"v2_aligned": "v2_official"},
        ),
    ]
    frames = [baselines]
    for filename, mapping in source_specs:
        frame = pd.read_csv(OUT / filename)
        frame = frame[frame["method"].isin(mapping)].copy()
        frame["method"] = frame["method"].map(mapping)
        frames.append(
            frame[
                [
                    "data_block",
                    "split",
                    "method",
                    "k",
                    "q",
                    "incident_hits",
                    "incident_total",
                    "event_recall_3d",
                    "event_recall_1d",
                    "precision_at_k",
                    "false_alerts_per_day",
                    "scenario_macro_recall_3d",
                ]
            ]
        )
    combined = add_utilities(
        pd.concat(frames, ignore_index=True)
    )
    combined.to_csv(
        OUT / "v2_model_comparison_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    k10 = combined[combined["k"] == 10]
    k10.to_csv(
        OUT / "v2_model_comparison_k10.csv",
        index=False,
        encoding="utf-8",
    )
    by_k = (
        100
        * combined.groupby(["k", "method"])["balanced"]
        .mean()
        .unstack()
        .round(2)
    )
    by_k.to_csv(
        OUT / "v2_model_comparison_by_k.csv",
        encoding="utf-8",
    )
    k10_mean = (
        100
        * k10.groupby("method")["balanced"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )
    k10_mean.to_csv(
        OUT / "v2_model_comparison_k10_mean.csv",
        header=["mean_balanced_utility_pct"],
        encoding="utf-8",
    )
    k10_blocks = (
        100
        * k10.pivot_table(
            index="method",
            columns=["data_block", "split"],
            values="balanced",
        ).round(2)
    )
    k10_blocks.to_csv(
        OUT / "v2_model_comparison_k10_blocks.csv",
        encoding="utf-8",
    )
    external = k10[k10["data_block"] == "r52_external"][
        [
            "method",
            "incident_hits",
            "incident_total",
            "event_recall_3d",
            "scenario_macro_recall_3d",
            "precision_at_k",
            "balanced",
        ]
    ].copy()
    external.to_csv(
        OUT / "v2_model_comparison_external_k10.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "k10_mean": k10_mean.to_dict(),
        "files": {
            "metrics": str(OUT / "v2_model_comparison_metrics.csv"),
            "k10": str(OUT / "v2_model_comparison_k10.csv"),
            "by_k": str(OUT / "v2_model_comparison_by_k.csv"),
            "k10_blocks": str(
                OUT / "v2_model_comparison_k10_blocks.csv"
            ),
            "external": str(
                OUT / "v2_model_comparison_external_k10.csv"
            ),
        },
    }
    (OUT / "v2_model_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("K=10 mean utility", flush=True)
    print(k10_mean.to_string(), flush=True)
    print("\nBudget comparison", flush=True)
    print(by_k.to_string(), flush=True)
    print("\nExternal event comparison", flush=True)
    print(external.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
