from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e19_models"
SEEDS = [20261111, 20261112, 20261113]

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from e19_user_manifold import (  # noqa: E402
    build_signal,
    dataset_config,
    evaluate_method,
    fit_user_manifolds,
    load_components,
    project_with_model,
)
from e19_user_manifold import UserMetricProjector  # noqa: E402
from quota_relationship_sim import daily_percentile  # noqa: E402
from risk_topk_r42 import compute_risk  # noqa: E402


def make_signal(data: dict, scores: np.ndarray) -> np.ndarray:
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        scores,
        data["active_mask"],
    )
    return daily_percentile(
        dynamic, data["active_mask"]
    ).astype(np.float32)


def run_dataset(dataset_name: str) -> None:
    config = dataset_config(dataset_name)
    data = config["data"]
    dataset = np.load(
        config["dataset_file"], allow_pickle=False
    )
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    plain, masked_q, _, channel_q = load_components(dataset_name)
    pca_q = np.load(
        OUT / f"e19_{dataset_name}_pca_manifold_quantile.npy"
    )
    burst_q = np.load(
        OUT / f"e19_{dataset_name}_hourly_burst_quantile.npy"
    )
    relation_q = np.load(
        OUT / f"e19_{dataset_name}_relation_quantile.npy"
    )
    auxiliary_all = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    auxiliary = auxiliary_all[config["auxiliary_key"]]
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    rows = []
    for seed in SEEDS:
        print(
            f"{dataset_name} evaluating seed {seed}",
            flush=True,
        )
        model = UserMetricProjector(
            plain.shape[-1], 24
        ).to(device)
        model.load_state_dict(
            torch.load(
                MODEL_DIR / f"{dataset_name}_user_metric_{seed}.pth",
                map_location=device,
                weights_only=True,
            )
        )
        projected = project_with_model(model, plain)
        _, _, raw, _ = fit_user_manifolds(
            projected, active_mask, train_end
        )
        learned_q = per_user_quantile_scores(
            raw, active_mask, train_end
        )
        dual_q = (pca_q + learned_q) / 2.0
        core_q = (masked_q + dual_q + channel_q) / 3.0
        consensus_q = np.minimum(burst_q, relation_q)
        final_q = np.maximum(core_q, consensus_q)
        signals = {
            "A3_dual_mean": build_signal(
                data, masked_q, dual_q, channel_q
            ),
            "C2_burst_relation_consensus": make_signal(
                data, final_q
            ),
        }
        for method, signal in signals.items():
            metrics = evaluate_method(
                data,
                signal,
                auxiliary,
                f"E19_c2seed{seed}_{method}",
                external=config["external"],
            )
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "metrics": metrics,
                }
            )
        del model
    summary = {
        "dataset": dataset_name,
        "seeds": SEEDS,
        "rows": rows,
    }
    path = OUT / f"e19_{dataset_name}_c2_seed_study.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["r42", "r52", "all"], default="all"
    )
    args = parser.parse_args()
    if args.dataset in {"r42", "all"}:
        run_dataset("r42")
    if args.dataset in {"r52", "all"}:
        run_dataset("r52")


if __name__ == "__main__":
    main()
