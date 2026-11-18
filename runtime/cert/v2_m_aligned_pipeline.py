from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
PRIMARY_K = 10
ALPHA = 0.4
IFOREST_SEED = 20261109
MANIFOLD_SUFFIX = {
    "v2m_a1_pca": "pca_manifold",
    "v2m_a2_learned": "learned_manifold",
    "v2m_a3_dual": "dual_mean_manifold",
}

sys.path.insert(0, str(ROOT))
from channel_evidence_experiment import encode_plain  # noqa: E402
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42,
    load_r52,
    q_from_alpha,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from ts2vec_dual_score_experiment import (  # noqa: E402
    ENCODE_ARGS,
    load_model,
)
from v2_m_pipeline import (  # noqa: E402
    BLOCKS,
    METHODS,
    add_utilities,
    evaluate_block,
)
from v2_pipeline import select_queue_with_sources  # noqa: E402


def ensure_retrained_masked_q(
    data: dict,
    values: np.ndarray,
    train_end: int,
) -> np.ndarray:
    path = OUT / "e19_r42_retrained_masked_quantile.npy"
    if path.exists():
        return np.load(path)
    print("Computing retrained masked quantile", flush=True)
    model = load_model(
        (
            ROOT
            / "processed_r42"
            / "innovation_sim"
            / "r4.2_ts2vec_quota_retrain.pth"
        ),
        values.shape[-1],
    )
    plain = encode_plain(model, values)
    masked = np.asarray(
        model.encode(values, mask="mask_last", **ENCODE_ARGS),
        dtype=np.float32,
    )
    raw = np.abs(masked - plain).sum(axis=-1)
    quantile = per_user_quantile_scores(
        raw, data["active_mask"], train_end
    )
    np.save(path, quantile)
    return quantile


def method_triple_quantile(
    block_name: str,
    method: str,
    data: dict,
    values: np.ndarray,
    train_end: int,
    channel_arrays,
    v2_components,
) -> np.ndarray:
    if method == "v2m_a0_control":
        return v2_components[
            f"{block_name}_triple_quantile"
        ].astype(np.float32)
    if block_name == "r42_original":
        masked_q = np.load(
            OUT / "e17_r42_M0_masked_quantile.npy"
        )
    elif block_name == "r52_external":
        masked_q = np.load(
            OUT / "e17_r52_M0_masked_quantile.npy"
        )
    else:
        masked_q = ensure_retrained_masked_q(
            data, values, train_end
        )
    manifold_prefix = MANIFOLD_SUFFIX[method]
    if block_name == "r42_original":
        manifold_path = (
            OUT / f"e19_r42_{manifold_prefix}_quantile.npy"
        )
    elif block_name == "r52_external":
        manifold_path = (
            OUT / f"e19_r52_{manifold_prefix}_quantile.npy"
        )
    else:
        manifold_path = (
            OUT
            / f"e19_r42_retrained_{manifold_prefix}_quantile.npy"
        )
    manifold_q = np.load(manifold_path)
    channel_q = channel_arrays[
        f"{block_name}_channel_top2_quantile"
    ].astype(np.float32)
    return (masked_q + manifold_q + channel_q) / 3.0


def build_aligned_auxiliary(
    data: dict,
    feature_file: Path,
    triple_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        triple_q,
        data["active_mask"],
    )
    original, augmented = fit_view_scores(
        data,
        feature_file,
        dynamic,
        data["personal"],
        IFOREST_SEED,
    )
    return np.maximum(original, augmented), original, augmented


def main() -> None:
    np.random.seed(IFOREST_SEED)
    torch.manual_seed(IFOREST_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    r42 = load_r42()
    r52 = load_r52()
    families = {"r42": r42, "r52": r52}
    feature_files = {
        "r42": ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        "r52": ROOT / "processed_r52" / "r5.2_user_day_features.csv",
    }
    sequence_files = {
        "r42": ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        "r52": ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
    }
    channel_arrays = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    v2_components = np.load(
        OUT / "v2_pipeline_components.npz",
        allow_pickle=False,
    )
    metric_frames = []
    queue_frames = []
    saved_arrays = {}
    diagnostic_rows = []

    for block_name, config in BLOCKS.items():
        print(f"Aligning IForest for {block_name}", flush=True)
        data = families[config["family"]]
        dataset = np.load(
            sequence_files[config["family"]],
            allow_pickle=False,
        )
        values = dataset["values"].astype(np.float32)
        train_end = int(dataset["train_end_index"])
        for method, manifold_prefix in METHODS.items():
            triple_q = method_triple_quantile(
                block_name,
                method,
                data,
                values,
                train_end,
                channel_arrays,
                v2_components,
            )
            signal = daily_percentile(
                compute_risk(
                    data["users"],
                    data["dates"],
                    triple_q,
                    data["active_mask"],
                ),
                data["active_mask"],
            ).astype(np.float32)
            auxiliary, original, augmented = build_aligned_auxiliary(
                data,
                feature_files[config["family"]],
                triple_q,
            )
            diagnostic_rows.append(
                {
                    "data_block": block_name,
                    "method": method,
                    "original_mean": float(np.nanmean(original)),
                    "augmented_mean": float(np.nanmean(augmented)),
                    "auxiliary_mean": float(np.nanmean(auxiliary)),
                }
            )
            metric_frames.append(
                evaluate_block(
                    block_name,
                    data,
                    signal,
                    auxiliary,
                    method,
                    external=bool(config["external"]),
                )
            )
            saved_arrays[
                f"{block_name}_{method}_triple_quantile"
            ] = triple_q
            saved_arrays[
                f"{block_name}_{method}_signal"
            ] = signal
            saved_arrays[
                f"{block_name}_{method}_aligned_iforest"
            ] = auxiliary

        a3_signal = saved_arrays[f"{block_name}_v2m_a3_dual_signal"]
        a3_auxiliary = saved_arrays[
            f"{block_name}_v2m_a3_dual_aligned_iforest"
        ]
        alerts, queue = select_queue_with_sources(
            a3_signal,
            a3_auxiliary,
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
        queue_frames.append(
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
        saved_arrays[f"{block_name}_v2m_a3_alerts_k10"] = alerts

    metrics = add_utilities(
        pd.concat(metric_frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "v2_m_aligned_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    pd.concat(queue_frames, ignore_index=True).to_csv(
        OUT / "v2_m_aligned_daily_queue_k10.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "v2_m_aligned_components.npz",
        **saved_arrays,
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        OUT / "v2_m_aligned_iforest_diagnostics.csv",
        index=False,
        encoding="utf-8",
    )

    k10 = metrics[metrics["k"] == PRIMARY_K].copy()
    official = pd.read_csv(OUT / "v2_pipeline_metrics.csv")
    official = official[official["k"] == PRIMARY_K].copy()
    official["variant"] = "official_v2"
    columns = list(k10.columns)
    for column in columns:
        if column not in official.columns:
            official[column] = np.nan
    comparison = pd.concat(
        [k10, official[columns]],
        ignore_index=True,
    )
    comparison.to_csv(
        OUT / "v2_m_aligned_k10_comparison.csv",
        index=False,
        encoding="utf-8",
    )
    means = comparison.groupby(
        ["method", "variant"], as_index=False
    )["balanced"].mean()
    means["balanced"] = means["balanced"] * 100.0
    means = means.sort_values("balanced", ascending=False)
    means.to_csv(
        OUT / "v2_m_aligned_k10_mean.csv",
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
        OUT / "v2_m_aligned_k10_blocks.csv",
        encoding="utf-8",
    )
    by_k = metrics.groupby(
        ["k", "method", "variant"], as_index=False
    )["balanced"].mean()
    by_k["balanced"] = by_k["balanced"] * 100.0
    by_k.to_csv(
        OUT / "v2_m_aligned_by_k.csv",
        index=False,
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
    official_aligned = official[
        official["method"] == "v2_aligned"
    ][["data_block", "split", "balanced"]].rename(
        columns={"balanced": "v2"}
    )
    paired = a0.merge(a3, on=["data_block", "split"]).merge(
        official_aligned,
        on=["data_block", "split"],
        how="left",
    )
    paired["a3_minus_a0"] = paired["a3"] - paired["a0"]
    paired["a3_minus_v2"] = paired["a3"] - paired["v2"]
    paired.to_csv(
        OUT / "v2_m_aligned_paired_comparison.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "model_version": "V2-M aligned candidate",
        "primary_k": PRIMARY_K,
        "q": q_from_alpha(PRIMARY_K, ALPHA),
        "iforest_seed": IFOREST_SEED,
        "k10_mean": means.to_dict(orient="records"),
        "a3_vs_a0": {
            "mean_difference": float(
                paired["a3_minus_a0"].mean()
            ),
            "wins": int((paired["a3_minus_a0"] > 0).sum()),
            "losses": int((paired["a3_minus_a0"] < 0).sum()),
        },
        "a3_vs_v2": {
            "mean_difference": float(
                paired["a3_minus_v2"].mean()
            ),
            "wins": int((paired["a3_minus_v2"] > 0).sum()),
            "losses": int((paired["a3_minus_v2"] < 0).sum()),
        },
        "outputs": {
            "metrics": str(OUT / "v2_m_aligned_metrics.csv"),
            "k10_mean": str(OUT / "v2_m_aligned_k10_mean.csv"),
            "k10_blocks": str(
                OUT / "v2_m_aligned_k10_blocks.csv"
            ),
            "paired": str(
                OUT / "v2_m_aligned_paired_comparison.csv"
            ),
            "queue": str(
                OUT / "v2_m_aligned_daily_queue_k10.csv"
            ),
            "components": str(
                OUT / "v2_m_aligned_components.npz"
            ),
        },
    }
    (OUT / "v2_m_aligned_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nK=10 mean", flush=True)
    print(means.to_string(index=False), flush=True)
    print("\nPaired", flush=True)
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
