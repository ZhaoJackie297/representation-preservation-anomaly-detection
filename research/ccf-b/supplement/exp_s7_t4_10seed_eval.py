"""Evaluate unified 10-seed T3/T4 checkpoints on CERT r4.2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_DIR = REPO_ROOT / "runtime" / "cert"
sys.path.insert(0, str(CERT_DIR))

from e0_e1_representation_preservation import (  # noqa: E402
    align_raw_features,
    global_train_z,
)
from e3_multichannel_amplitude import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    aggregate_components,
    evaluate_variant,
)
from e4_joint_instance_pai import (  # noqa: E402
    bank_scores,
    load_soft_model,
)
from quota_relationship_sim import load_r42  # noqa: E402


SEEDS = list(range(20261116, 20261126))
OUT_ROOT = (
    REPO_ROOT
    / "research"
    / "ccf-b"
    / "supplement_results"
    / "exp_s7_10seed_unified"
)
SCORE_ROOT = Path(
    r"E:\MarketRisker\work\cert\external\unify_10seed_v1"
    r"\cert_t4_10seed_scores"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-tag", default="r42_10seed_unified")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["T4_full_soft", "T3_instance_soft"],
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SCORE_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = np.load(
        CERT_DIR / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    data = load_r42()
    raw = align_raw_features(
        CERT_DIR / "processed_r42" / "r4.2_user_day_features.csv",
        users,
        dates,
    )
    g_raw, q_raw = aggregate_components(
        raw,
        active,
        train_end,
        user_specific=True,
        aggregation="channel_mean",
    )
    g_z = global_train_z(g_raw, active, train_end)
    q_z = global_train_z(q_raw, active, train_end)
    rows: list[dict] = []

    for variant in args.variants:
        for seed in args.seeds:
            score_path = SCORE_ROOT / f"{variant}_seed_{seed}_scores.npz"
            if score_path.exists() and not args.force:
                with np.load(score_path, allow_pickle=False) as stored:
                    scores = {
                        key: stored[key].astype(np.float32)
                        for key in stored.files
                    }
            else:
                checkpoint = (
                    CERT_DIR
                    / "experiments"
                    / "representation_preservation"
                    / "e2_instance"
                    / args.instance_tag
                    / variant
                    / f"seed_{seed}"
                    / "model.pth"
                )
                model, device = load_soft_model(checkpoint)
                encode_args = {
                    "causal": True,
                    "sliding_length": 1,
                    "sliding_padding": 30,
                    "batch_size": 256,
                }
                masked = np.asarray(
                    model.encode(values, mask="mask_last", **encode_args),
                    dtype=np.float32,
                )
                plain = np.asarray(
                    model.encode(values, **encode_args),
                    dtype=np.float32,
                )
                original = np.abs(masked - plain).sum(axis=-1)
                cosine, euclidean = bank_scores(
                    plain,
                    plain,
                    active,
                    train_end,
                    device,
                )
                original_z = global_train_z(original, active, train_end)
                cosine_z = global_train_z(cosine, active, train_end)
                euclidean_z = global_train_z(euclidean, active, train_end)
                scores = {
                    "mask_original": original_z,
                    "bank_cosine": cosine_z,
                    "bank_euclidean": euclidean_z,
                    "pai_a1": (
                        euclidean_z + PAI_W_G * g_z + PAI_W_Q * q_z
                    ).astype(np.float32),
                    "pai_official": (
                        0.6 * euclidean_z + 0.4 * g_z + 0.2 * q_z
                    ).astype(np.float32),
                }
                np.savez_compressed(score_path, **scores)
            for scorer, signal in scores.items():
                method = f"{variant}__{scorer}"
                rows.extend(
                    evaluate_variant(
                        method,
                        signal,
                        data,
                        active,
                        train_end,
                        val_end,
                        seed,
                    )
                )
            print(f"[DONE] {variant} seed={seed}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT_ROOT / "t4_10seed_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        frame.groupby(["method", "split", "k"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            event_recall_3d_std=("event_recall_3d", "std"),
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_std=("incident_hits", "std"),
        )
        .sort_values(["split", "k", "method"])
    )
    summary.to_csv(
        OUT_ROOT / "t4_10seed_summary.csv",
        index=False,
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
