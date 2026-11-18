from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"

sys.path.insert(0, str(ROOT))
from e19_user_manifold import (  # noqa: E402
    build_signal,
    dataset_config,
    evaluate_method,
    load_components,
)


def run_dataset(dataset_name: str) -> None:
    config = dataset_config(dataset_name)
    data = config["data"]
    dataset = np.load(
        config["dataset_file"], allow_pickle=False
    )
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    auxiliary_all = np.load(
        OUT / "v2_iforest_alignment_arrays.npz",
        allow_pickle=False,
    )
    auxiliary = auxiliary_all[config["auxiliary_key"]]
    _, masked_q, _, channel_q = load_components(dataset_name)
    pca_q = np.load(
        OUT / f"e19_{dataset_name}_pca_manifold_quantile.npy"
    )
    learned_q = np.load(
        OUT
        / f"e19_{dataset_name}_learned_manifold_quantile.npy"
    )
    dual_mean_q = (pca_q + learned_q) / 2.0
    dual_max_q = np.maximum(pca_q, learned_q)
    methods = {
        "A1_pca_manifold": build_signal(
            data, masked_q, pca_q, channel_q
        ),
        "A2_learned_manifold": build_signal(
            data, masked_q, learned_q, channel_q
        ),
        "A3_dual_mean": build_signal(
            data, masked_q, dual_mean_q, channel_q
        ),
        "A4_dual_max_sensitivity": build_signal(
            data, masked_q, dual_max_q, channel_q
        ),
    }
    results = {}
    for method, signal in methods.items():
        results[method] = evaluate_method(
            data,
            signal,
            auxiliary,
            f"E19_{method}",
            external=config["external"],
        )
        np.save(
            OUT
            / f"e19_{dataset_name}_{method}_signal.npy",
            signal,
        )
    np.save(
        OUT / f"e19_{dataset_name}_dual_mean_manifold_quantile.npy",
        dual_mean_q,
    )
    np.save(
        OUT / f"e19_{dataset_name}_dual_max_manifold_quantile.npy",
        dual_max_q,
    )
    summary = {
        "dataset": dataset_name,
        "fusion": {
            "default": "A3_dual_mean",
            "mean_rule": "(PCA manifold quantile + learned manifold quantile)/2",
            "max_rule": "sensitivity only, not selected by external results",
            "weights_searched": False,
        },
        "results": results,
    }
    path = OUT / f"e19_{dataset_name}_dual_summary.json"
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
