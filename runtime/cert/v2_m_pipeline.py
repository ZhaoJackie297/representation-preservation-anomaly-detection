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
from production_protocol_r42 import build_alerts  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)
from v2_pipeline import select_queue_with_sources  # noqa: E402


BLOCKS = {
    "r42_original": {
        "family": "r42",
        "prefix": "e19_r42",
        "external": False,
    },
    "r42_retrained": {
        "family": "r42",
        "prefix": "e19_r42_retrained",
        "external": False,
    },
    "r52_external": {
        "family": "r52",
        "prefix": "e19_r52",
        "external": True,
    },
}

METHODS = {
    "v2m_a0_control": "A0_latent_control",
    "v2m_a1_pca": "A1_pca_manifold",
    "v2m_a2_learned": "A2_learned_manifold",
    "v2m_a3_dual": "A3_dual_mean",
}


def add_utilities(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for profile, weights in PROFILES.items():
        frame[profile] = sum(
            frame[column] * weight
            for column, weight in weights.items()
        )
    return frame


def evaluate_block(
    block_name: str,
    data: dict,
    signal: np.ndarray,
    auxiliary: np.ndarray,
    method: str,
    *,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        for variant, use_iforest in (
            ("complete_ts", False),
            ("aligned_iforest", True),
        ):
            q = q_from_alpha(k, ALPHA) if use_iforest else 0
            if use_iforest:
                alerts = build_unique_quota_alerts(
                    signal,
                    auxiliary,
                    data["active_mask"],
                    k,
                    q,
                )
            else:
                alerts = build_alerts(
                    signal, data["active_mask"], k, 7
                )
            result = evaluate_alert_system(
                data,
                alerts,
                f"{method}_{variant}",
                k,
                external=external,
            )
            splits = (
                ["external_test"]
                if external
                else ["validation", "test"]
            )
            for split in splits:
                rows.append(
                    {
                        "data_block": block_name,
                        "split": split,
                        "method": method,
                        "variant": variant,
                        "k": k,
                        "q": q,
                        **compact_metric(
                            result,
                            None if external else split,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    r42 = load_r42()
    r52 = load_r52()
    families = {"r42": r42, "r52": r52}
    iforest = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    frames = []
    queues = []
    saved_signals = {}
    for block_name, config in BLOCKS.items():
        print(f"Evaluating {block_name}", flush=True)
        data = families[config["family"]]
        auxiliary = iforest[
            f"{block_name}_v2_aligned_dual_view_iforest"
        ].astype(np.float32)
        for method, prefix in METHODS.items():
            signal = np.load(
                OUT / f"{config['prefix']}_{prefix}_signal.npy"
            )
            saved_signals[f"{block_name}_{method}_signal"] = signal
            frames.append(
                evaluate_block(
                    block_name,
                    data,
                    signal,
                    auxiliary,
                    method,
                    external=bool(config["external"]),
                )
            )
        a3_signal = np.load(
            OUT
            / f"{config['prefix']}_A3_dual_mean_signal.npy"
        )
        alerts, queue = select_queue_with_sources(
            a3_signal,
            auxiliary,
            data["active_mask"],
            PRIMARY_K,
        )
        queue["data_block"] = block_name
        queue["date"] = queue["date_index"].map(
            lambda index: data["dates"][index].strftime("%Y-%m-%d")
        )
        queue["user_id"] = queue["user_index"].map(
            lambda index: str(data["users"][index])
        )
        if config["external"]:
            queue = queue[
                queue["date_index"] >= data["test_start"]
            ].copy()
            queue["split"] = "external_test"
        else:
            queue = queue[
                queue["date_index"] > data["train_end"]
            ].copy()
            queue["split"] = np.where(
                queue["date_index"] <= data["val_end"],
                "validation",
                "test",
            )
        queues.append(
            queue[
                [
                    "data_block",
                    "split",
                    "date",
                    "rank",
                    "user_id",
                    "source",
                    "ts_score",
                    "iforest_score",
                ]
            ]
        )
        saved_signals[
            f"{block_name}_aligned_iforest"
        ] = auxiliary
        saved_signals[
            f"{block_name}_a3_alerts_k10"
        ] = alerts

    metrics = add_utilities(
        pd.concat(frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "v2_m_pipeline_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    queue_frame = pd.concat(queues, ignore_index=True)
    queue_frame.to_csv(
        OUT / "v2_m_pipeline_daily_queue_k10.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "v2_m_pipeline_components.npz",
        **saved_signals,
    )

    k10 = metrics[metrics["k"] == PRIMARY_K]
    official = pd.read_csv(
        OUT / "v2_pipeline_metrics.csv"
    )
    official = official[official["k"] == PRIMARY_K].copy()
    official["variant"] = "official_v2"
    columns = [
        "data_block",
        "split",
        "method",
        "variant",
        "k",
        "q",
        "incident_hits",
        "incident_total",
        "event_recall_3d",
        "event_recall_1d",
        "precision_at_k",
        "false_alerts_per_day",
        "scenario_macro_recall_3d",
        "impact_first",
        "balanced",
        "workload_first",
    ]
    k10_comparison = pd.concat(
        [k10, official[columns]],
        ignore_index=True,
    )
    k10_comparison.to_csv(
        OUT / "v2_m_pipeline_k10_comparison.csv",
        index=False,
        encoding="utf-8",
    )
    means = k10_comparison.groupby(
        ["method", "variant"], as_index=False
    )["balanced"].mean()
    means["balanced"] = means["balanced"] * 100.0
    means = means.sort_values("balanced", ascending=False)
    means.to_csv(
        OUT / "v2_m_pipeline_k10_mean.csv",
        index=False,
        encoding="utf-8",
    )
    by_block = (
        100
        * k10_comparison.pivot_table(
            index=["method", "variant"],
            columns=["data_block", "split"],
            values="balanced",
        )
    )
    by_block.to_csv(
        OUT / "v2_m_pipeline_k10_blocks.csv",
        encoding="utf-8",
    )
    by_k = (
        100
        * metrics.groupby(["k", "method", "variant"])[
            "balanced"
        ]
        .mean()
        .unstack(["method", "variant"])
        .round(2)
    )
    by_k.to_csv(
        OUT / "v2_m_pipeline_by_k.csv",
        encoding="utf-8",
    )
    a0 = k10[
        (k10["method"] == "v2m_a0_control")
        & (k10["variant"] == "aligned_iforest")
    ][["data_block", "split", "balanced"]].rename(
        columns={"balanced": "a0"}
    )
    a3 = k10[
        (k10["method"] == "v2m_a3_dual")
        & (k10["variant"] == "aligned_iforest")
    ][["data_block", "split", "balanced"]].rename(
        columns={"balanced": "a3"}
    )
    paired = a0.merge(a3, on=["data_block", "split"])
    paired["difference"] = paired["a3"] - paired["a0"]
    paired.to_csv(
        OUT / "v2_m_pipeline_a3_vs_a0.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "model_version": "V2-M candidate",
        "default_candidate": "v2m_a3_dual + aligned_iforest",
        "primary_k": PRIMARY_K,
        "q": q_from_alpha(PRIMARY_K, ALPHA),
        "cooldown_days": 7,
        "k10_mean": means.to_dict(orient="records"),
        "a3_vs_a0": {
            "mean_difference": float(paired["difference"].mean()),
            "wins": int((paired["difference"] > 0).sum()),
            "ties": int((paired["difference"] == 0).sum()),
            "losses": int((paired["difference"] < 0).sum()),
        },
        "outputs": {
            "metrics": str(OUT / "v2_m_pipeline_metrics.csv"),
            "k10_comparison": str(
                OUT / "v2_m_pipeline_k10_comparison.csv"
            ),
            "k10_mean": str(OUT / "v2_m_pipeline_k10_mean.csv"),
            "k10_blocks": str(
                OUT / "v2_m_pipeline_k10_blocks.csv"
            ),
            "queue": str(
                OUT / "v2_m_pipeline_daily_queue_k10.csv"
            ),
            "components": str(
                OUT / "v2_m_pipeline_components.npz"
            ),
        },
    }
    (OUT / "v2_m_pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nK=10 mean", flush=True)
    print(means.to_string(index=False), flush=True)
    print("\nA3 vs A0", flush=True)
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
