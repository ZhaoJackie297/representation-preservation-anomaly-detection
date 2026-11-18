from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e19_models"
SEEDS = [20261111, 20261112, 20261113]

sys.path.insert(0, str(ROOT))
from channel_evidence_experiment import encode_plain  # noqa: E402
from e19_user_manifold import (  # noqa: E402
    fit_user_manifolds,
    project_with_model,
    train_user_metric,
)
from quota_relationship_sim import load_r42  # noqa: E402
from ts2vec_dual_score_experiment import load_model  # noqa: E402


def main() -> None:
    data = load_r42()
    dataset_path = (
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
    )
    model_path = (
        ROOT
        / "processed_r42"
        / "innovation_sim"
        / "r4.2_ts2vec_quota_retrain.pth"
    )
    dataset = np.load(dataset_path, allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    plain_path = OUT / "e19_r42_retrained_plain.npy"
    if plain_path.exists():
        plain = np.load(plain_path, mmap_mode="r")
    else:
        print("Encoding retrained plain representations", flush=True)
        model = load_model(model_path, values.shape[-1])
        plain = encode_plain(model, values)
        np.save(plain_path, plain)
    rows = []
    for seed in SEEDS:
        print(f"Training retrained metric seed {seed}", flush=True)
        model, losses = train_user_metric(
            plain,
            active_mask,
            train_end,
            MODEL_DIR
            / f"r42_retrained_user_metric_{seed}.pth",
            seed=seed,
        )
        projected = project_with_model(model, plain)
        _, _, raw, diagnostics = fit_user_manifolds(
            projected, active_mask, train_end
        )
        from deep_compare_per_user_r42 import (
            per_user_quantile_scores,
        )

        quantile = per_user_quantile_scores(
            raw, active_mask, train_end
        )
        np.save(
            OUT
            / f"e19_r42_retrained_learned_manifold_quantile_{seed}.npy",
            quantile,
        )
        rows.append(
            {
                "seed": seed,
                "losses": losses,
                "diagnostics": diagnostics,
            }
        )
    summary = {
        "dataset": "r42_retrained",
        "seeds": SEEDS,
        "rows": rows,
    }
    (OUT / "e19_r42_retrained_metric_seed_training.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
