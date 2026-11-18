from __future__ import annotations

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
    UserMetricProjector,
    fit_user_manifolds,
    load_components,
    project_with_model,
)


def export_dataset(dataset_name: str) -> None:
    sequence_path = (
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        if dataset_name == "r42"
        else ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
    )
    dataset = np.load(sequence_path, allow_pickle=False)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    plain, _, _, _ = load_components(dataset_name)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    for seed in SEEDS:
        print(f"Exporting {dataset_name} seed {seed}", flush=True)
        model_path = (
            MODEL_DIR / f"{dataset_name}_user_metric_{seed}.pth"
        )
        if not model_path.exists():
            continue
        model = UserMetricProjector(
            plain.shape[-1], 24
        ).to(device)
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
                weights_only=True,
            )
        )
        projected = project_with_model(model, plain)
        _, _, raw, _ = fit_user_manifolds(
            projected, active_mask, train_end
        )
        quantile = per_user_quantile_scores(
            raw, active_mask, train_end
        )
        np.save(
            OUT
            / f"e19_{dataset_name}_learned_manifold_quantile_{seed}.npy",
            quantile,
        )


if __name__ == "__main__":
    export_dataset("r42")
    export_dataset("r52")
