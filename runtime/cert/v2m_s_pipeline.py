from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
METHOD = "v2m_s_a2_ensemble"
PRIMARY_K = 10

sys.path.insert(0, str(ROOT))
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42_cached,
    load_r52_cached,
    q_from_alpha,
)
from risk_topk_r42 import compute_severity_risk  # noqa: E402
from v2_m_aligned_pipeline import (  # noqa: E402
    build_aligned_auxiliary,
)
from v2_m_ensemble_pipeline import (  # noqa: E402
    masked_quantile,
    quantile_paths,
)
from v2_m_pipeline import (  # noqa: E402
    BLOCKS,
    add_utilities,
    evaluate_block,
)
from v2_pipeline import select_queue_with_sources  # noqa: E402


def main() -> None:
    r42 = load_r42_cached()
    r52 = load_r52_cached()
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
        print(f"Evaluating V2-M-S for {block_name}", flush=True)
        data = families[config["family"]]
        manifold_q = np.mean(
            np.stack(
                [np.load(path) for path in quantile_paths(block_name)],
                axis=0,
            ),
            axis=0,
        ).astype(np.float32)
        channel_q = channel_arrays[
            f"{block_name}_channel_top2_quantile"
        ].astype(np.float32)
        triple_q = (
            masked_quantile(block_name)
            + manifold_q
            + channel_q
        ) / 3.0

        severity_risk = compute_severity_risk(
            data["users"],
            data["dates"],
            triple_q,
            data["active_mask"],
        )
        signal = daily_percentile(
            severity_risk, data["active_mask"]
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
        saved[f"{block_name}_severity_quantile"] = triple_q
        saved[f"{block_name}_signal"] = signal
        saved[f"{block_name}_aligned_iforest"] = auxiliary
        saved[f"{block_name}_alerts_k10"] = alerts

    metrics = add_utilities(
        pd.concat(metric_frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "v2m_s_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    pd.concat(queues, ignore_index=True).to_csv(
        OUT / "v2m_s_daily_queue_k10.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "v2m_s_components.npz",
        **saved,
    )

    k10 = metrics[metrics["k"] == PRIMARY_K].copy()
    means = k10.groupby("variant", as_index=False)[
        "balanced"
    ].mean()
    means["balanced"] *= 100.0
    means.to_csv(
        OUT / "v2m_s_k10_mean.csv",
        index=False,
        encoding="utf-8",
    )
    blocks = (
        100
        * k10.pivot_table(
            index="variant",
            columns=["data_block", "split"],
            values="balanced",
        )
    )
    blocks.to_csv(
        OUT / "v2m_s_k10_blocks.csv",
        encoding="utf-8",
    )
    by_k = (
        100
        * metrics.groupby(["k", "variant"])["balanced"]
        .mean()
        .unstack("variant")
    )
    by_k.to_csv(
        OUT / "v2m_s_by_k.csv",
        encoding="utf-8",
    )

    old_path = OUT / "v2_m_ensemble_metrics.csv"
    comparison_rows = []
    if old_path.exists():
        old = pd.read_csv(old_path)
        old = old[old["k"] == PRIMARY_K].copy()
        old = old[old["method"] == "v2m_a2_ensemble"]
        key = ["data_block", "split", "variant"]
        old = old[key + ["balanced"]].rename(
            columns={"balanced": "legacy_weighted"}
        )
        new = k10[key + ["balanced"]].rename(
            columns={"balanced": "severity_only"}
        )
        comparison = old.merge(new, on=key)
        comparison["difference_pp"] = (
            comparison["severity_only"]
            - comparison["legacy_weighted"]
        ) * 100.0
        comparison.to_csv(
            OUT / "v2m_s_vs_legacy_k10.csv",
            index=False,
            encoding="utf-8",
        )
        comparison_rows = comparison.to_dict(orient="records")

    summary = {
        "model_version": "V2-M-S",
        "current_s3_risk": "severity_only",
        "legacy_weighted_formula_active": False,
        "primary_k": PRIMARY_K,
        "q": q_from_alpha(PRIMARY_K, 0.4),
        "iforest_seed": 20261109,
        "k10_mean": means.to_dict(orient="records"),
        "by_k": by_k.to_dict(),
        "versus_legacy_weighted": comparison_rows,
        "outputs": {
            "metrics": str(OUT / "v2m_s_metrics.csv"),
            "queue": str(OUT / "v2m_s_daily_queue_k10.csv"),
            "components": str(OUT / "v2m_s_components.npz"),
            "k10_mean": str(OUT / "v2m_s_k10_mean.csv"),
            "k10_blocks": str(OUT / "v2m_s_k10_blocks.csv"),
        },
    }
    (OUT / "v2m_s_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(means.to_string(index=False))


if __name__ == "__main__":
    main()
