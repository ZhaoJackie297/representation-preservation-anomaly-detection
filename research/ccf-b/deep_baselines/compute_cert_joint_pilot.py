from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
RUNTIME = REPO / "runtime" / "cert"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

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


PAI_OFFICIAL_EU = 0.6
PAI_OFFICIAL_G = 0.4
PAI_OFFICIAL_Q = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse saved CERT T0/T4 models to compute euclidean-bank and "
            "PAI-A1 scores for the exploratory paired pilot."
        )
    )
    parser.add_argument("--cert-root", type=Path, required=True)
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--t0-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20261201, 20261202, 20261203],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = args.cert_root / "processed_r42"
    dataset = np.load(
        processed / "r4.2_sequence_dataset.npz",
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
        processed / "r4.2_user_day_features.csv",
        users,
        dates,
    )
    g_a1, q_a1 = aggregate_components(
        raw,
        active,
        train_end,
        user_specific=True,
        aggregation="channel_mean",
    )
    g_a1_z = global_train_z(g_a1, active, train_end)
    q_a1_z = global_train_z(q_a1, active, train_end)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, model_root in (
        ("T0_matched", args.t0_run_root),
        ("T4_full_soft", args.instance_root / "T4_full_soft"),
    ):
        for seed in args.seeds:
            model_path = (
                model_root / f"r42_T0_seed{seed}" / "model.pth"
                if variant == "T0_matched"
                else model_root / f"seed_{seed}" / "model.pth"
            )
            model, device = load_soft_model(model_path)
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
            original_z = global_train_z(
                np.abs(masked - plain).sum(axis=-1),
                active,
                train_end,
            )
            _, euc_score = bank_scores(
                plain,
                plain,
                active,
                train_end,
                device,
            )
            euc_z = global_train_z(euc_score, active, train_end)
            pai_a1 = euc_z + PAI_W_G * g_a1_z + PAI_W_Q * q_a1_z
            pai_official = (
                PAI_OFFICIAL_EU * euc_z
                + PAI_OFFICIAL_G * g_a1_z
                + PAI_OFFICIAL_Q * q_a1_z
            )
            scores = {
                "original": original_z,
                "euclidean": euc_z,
                "pai_a1": pai_a1,
                "pai_official": pai_official,
            }
            for score_name, score in scores.items():
                rows.extend(
                    evaluate_variant(
                        f"{variant}_{score_name}",
                        score,
                        data,
                        active,
                        train_end,
                        val_end,
                        seed,
                    )
                )
            np.savez_compressed(
                args.output_dir / f"{variant}_seed_{seed}_scores.npz",
                **scores,
                mag_g_z=g_a1_z,
                t2_z=q_a1_z,
            )
            print(f"{variant} seed={seed} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(
        args.output_dir / "metrics_by_seed.csv",
        index=False,
    )
    summary = (
        frame.groupby(["method", "split", "k"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            event_recall_3d_std=("event_recall_3d", "std"),
            precision_at_k_mean=("precision_at_k", "mean"),
            incident_hits_mean=("incident_hits", "mean"),
        )
        .sort_values(["split", "k", "method"])
    )
    summary.to_csv(
        args.output_dir / "metric_summary.csv",
        index=False,
    )
    protocol = {
        "status": "exploratory_reuse_of_saved_models",
        "seeds": args.seeds,
        "repository_bank": {
            "clusters": "min(500, max(1, round(0.1 * active training days)))",
            "query": "mean of three nearest euclidean centers",
        },
        "pai_a1_weights": {
            "euclidean": 1.0,
            "mag_g": PAI_W_G,
            "t2": PAI_W_Q,
        },
        "training_reused": True,
        "t4_configuration": {
            "lambda": 0.5,
            "soft_instance": True,
            "soft_temporal": True,
            "tau_temp": 1.5,
        },
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
