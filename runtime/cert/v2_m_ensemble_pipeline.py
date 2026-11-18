from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
SEEDS = [20261111, 20261112, 20261113]
METHOD = "v2m_a2_ensemble"

sys.path.insert(0, str(ROOT))
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42,
    load_r52,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from v2_m_pipeline import (  # noqa: E402
    BLOCKS,
    add_utilities,
    evaluate_block,
)
from v2_m_aligned_pipeline import (  # noqa: E402
    build_aligned_auxiliary,
)
from v2_pipeline import select_queue_with_sources  # noqa: E402


def quantile_paths(block_name: str) -> list[Path]:
    if block_name == "r42_original":
        return [
            OUT / f"e19_r42_learned_manifold_quantile_{seed}.npy"
            for seed in SEEDS
        ]
    if block_name == "r42_retrained":
        return [
            OUT
            / f"e19_r42_retrained_learned_manifold_quantile_{seed}.npy"
            for seed in SEEDS
        ]
    return [
        OUT / f"e19_r52_learned_manifold_quantile_{seed}.npy"
        for seed in SEEDS
    ]


def masked_quantile(block_name: str) -> np.ndarray:
    if block_name == "r42_original":
        return np.load(OUT / "e17_r42_M0_masked_quantile.npy")
    if block_name == "r42_retrained":
        return np.load(
            OUT / "e19_r42_retrained_masked_quantile.npy"
        )
    return np.load(OUT / "e17_r52_M0_masked_quantile.npy")


def main() -> None:
    r42 = load_r42()
    r52 = load_r52()
    families = {"r42": r42, "r52": r52}
    feature_files = {
        "r42": ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        "r52": ROOT / "processed_r52" / "r5.2_user_day_features.csv",
    }
    channel_arrays = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    metric_frames = []
    queues = []
    saved = {}
    for block_name, config in BLOCKS.items():
        print(f"Evaluating ensemble for {block_name}", flush=True)
        data = families[config["family"]]
        quantiles = [
            np.load(path) for path in quantile_paths(block_name)
        ]
        ensemble_q = np.mean(
            np.stack(quantiles, axis=0),
            axis=0,
        ).astype(np.float32)
        channel_q = channel_arrays[
            f"{block_name}_channel_top2_quantile"
        ].astype(np.float32)
        triple_q = (
            masked_quantile(block_name)
            + ensemble_q
            + channel_q
        ) / 3.0
        dynamic = compute_risk(
            data["users"],
            data["dates"],
            triple_q,
            data["active_mask"],
        )
        signal = daily_percentile(
            dynamic, data["active_mask"]
        ).astype(np.float32)
        auxiliary, original, augmented = build_aligned_auxiliary(
            data,
            feature_files[config["family"]],
            triple_q,
        )
        metric_frames.append(
            evaluate_block(
                block_name,
                data,
                signal,
                auxiliary,
                METHOD,
                external=bool(config["external"]),
            )
        )
        alerts, queue = select_queue_with_sources(
            signal,
            auxiliary,
            data["active_mask"],
            10,
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
        saved[f"{block_name}_ensemble_quantile"] = ensemble_q
        saved[f"{block_name}_triple_quantile"] = triple_q
        saved[f"{block_name}_signal"] = signal
        saved[f"{block_name}_aligned_iforest"] = auxiliary
        saved[f"{block_name}_alerts_k10"] = alerts

    metrics = add_utilities(
        pd.concat(metric_frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "v2_m_ensemble_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    pd.concat(queues, ignore_index=True).to_csv(
        OUT / "v2_m_ensemble_daily_queue_k10.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "v2_m_ensemble_components.npz",
        **saved,
    )
    k10 = metrics[metrics["k"] == 10].copy()
    official = pd.read_csv(OUT / "v2_pipeline_metrics.csv")
    official = official[official["k"] == 10].copy()
    official["variant"] = "official_v2"
    single = pd.read_csv(OUT / "v2_m_aligned_metrics.csv")
    single = single[
        (single["k"] == 10)
        & (single["method"] == "v2m_a2_learned")
    ].copy()
    columns = list(k10.columns)
    merged_frames = [k10, single]
    for frame in merged_frames[1:]:
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
    comparison = pd.concat(
        [k10, single[columns], official[columns]],
        ignore_index=True,
    )
    comparison.to_csv(
        OUT / "v2_m_ensemble_k10_comparison.csv",
        index=False,
        encoding="utf-8",
    )
    means = comparison.groupby(
        ["method", "variant"], as_index=False
    )["balanced"].mean()
    means["balanced"] *= 100.0
    means = means.sort_values("balanced", ascending=False)
    means.to_csv(
        OUT / "v2_m_ensemble_k10_mean.csv",
        index=False,
        encoding="utf-8",
    )
    by_block = (
        100
        * comparison.pivot_table(
            index=["method", "variant"],
            columns=["data_block", "split"],
            values="balanced",
        )
    )
    by_block.to_csv(
        OUT / "v2_m_ensemble_k10_blocks.csv",
        encoding="utf-8",
    )
    official_blocks = comparison[
        comparison["method"] == "v2_aligned"
    ][["data_block", "split", "balanced"]].rename(
        columns={"balanced": "v2"}
    )
    ensemble_blocks = comparison[
        (comparison["method"] == METHOD)
        & (comparison["variant"] == "aligned_iforest")
    ][["data_block", "split", "balanced"]].rename(
        columns={"balanced": "ensemble"}
    )
    paired = official_blocks.merge(
        ensemble_blocks, on=["data_block", "split"]
    )
    paired["difference"] = paired["ensemble"] - paired["v2"]
    paired.to_csv(
        OUT / "v2_m_ensemble_vs_v2.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "model_version": "V2-M-A2 seed ensemble",
        "seeds": SEEDS,
        "primary_k": 10,
        "q": 4,
        "k10_mean": means.to_dict(orient="records"),
        "vs_v2": {
            "mean_difference": float(
                paired["difference"].mean()
            ),
            "wins": int((paired["difference"] > 0).sum()),
            "ties": int((paired["difference"] == 0).sum()),
            "losses": int((paired["difference"] < 0).sum()),
        },
        "outputs": {
            "metrics": str(OUT / "v2_m_ensemble_metrics.csv"),
            "k10_mean": str(
                OUT / "v2_m_ensemble_k10_mean.csv"
            ),
            "k10_blocks": str(
                OUT / "v2_m_ensemble_k10_blocks.csv"
            ),
            "queue": str(
                OUT / "v2_m_ensemble_daily_queue_k10.csv"
            ),
            "components": str(
                OUT / "v2_m_ensemble_components.npz"
            ),
        },
    }
    (OUT / "v2_m_ensemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(means.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
