from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


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
    train_user_metric,
)


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
    auxiliary_all = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    auxiliary = auxiliary_all[config["auxiliary_key"]]
    rows = []
    for seed in SEEDS:
        print(
            f"{dataset_name} metric seed {seed}",
            flush=True,
        )
        model, losses = train_user_metric(
            plain,
            active_mask,
            train_end,
            MODEL_DIR / f"{dataset_name}_user_metric_{seed}.pth",
            seed=seed,
        )
        projected = project_with_model(model, plain)
        _, _, raw, diagnostics = fit_user_manifolds(
            projected, active_mask, train_end
        )
        learned_q = per_user_quantile_scores(
            raw, active_mask, train_end
        )
        dual_q = (pca_q + learned_q) / 2.0
        methods = {
            "A2_learned_manifold": build_signal(
                data, masked_q, learned_q, channel_q
            ),
            "A3_dual_mean": build_signal(
                data, masked_q, dual_q, channel_q
            ),
        }
        for method, signal in methods.items():
            metrics = evaluate_method(
                data,
                signal,
                auxiliary,
                f"E19_seed{seed}_{method}",
                external=config["external"],
            )
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "losses": losses,
                    "diagnostics": diagnostics,
                    "metrics": metrics,
                }
            )
        del model
    summary = {
        "dataset": dataset_name,
        "seeds": SEEDS,
        "rows": rows,
    }
    path = OUT / f"e19_{dataset_name}_metric_seed_study.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["r42", "r52"], default="r42"
    )
    args = parser.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
