from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e19_models"
DATASET_NAME = "r42_retrained"

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from e17_multigranular_mask import (  # noqa: E402
    build_application_scores,
    evaluate_signal,
    load_base_student,
)
from e19_user_manifold import (  # noqa: E402
    build_signal,
    fit_global_pca,
    fit_user_manifolds,
    project_in_chunks,
    project_with_model,
    train_user_metric,
)
from quota_relationship_sim import load_r42  # noqa: E402


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
    train_end = int(dataset["train_end_index"])
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    student, device = load_base_student(
        model_path, values.shape[-1]
    )
    print("Building retrained V2 components", flush=True)
    a0_signal, arrays = build_application_scores(
        student,
        values,
        data,
        train_end,
        device,
    )
    plain = arrays["plain"]
    masked_q = arrays["masked_quantile"]
    channel_q = arrays["channel_top2_quantile"]

    print("Building PCA manifold", flush=True)
    pca_mean, pca_components, pca_explained = fit_global_pca(
        plain, active_mask, train_end, 16
    )
    pca_representation = project_in_chunks(
        plain, pca_mean, pca_components
    )
    _, _, pca_raw, pca_diagnostics = fit_user_manifolds(
        pca_representation, active_mask, train_end
    )
    pca_q = per_user_quantile_scores(
        pca_raw, active_mask, train_end
    )

    print("Training learned user metric", flush=True)
    metric_path = (
        MODEL_DIR / f"{DATASET_NAME}_user_metric.pth"
    )
    metric_model, metric_losses = train_user_metric(
        plain,
        active_mask,
        train_end,
        metric_path,
    )
    metric_representation = project_with_model(
        metric_model, plain
    )
    _, _, metric_raw, metric_diagnostics = fit_user_manifolds(
        metric_representation, active_mask, train_end
    )
    metric_q = per_user_quantile_scores(
        metric_raw, active_mask, train_end
    )
    dual_q = (pca_q + metric_q) / 2.0

    methods = {
        "A0_latent_control": a0_signal,
        "A1_pca_manifold": build_signal(
            data, masked_q, pca_q, channel_q
        ),
        "A2_learned_manifold": build_signal(
            data, masked_q, metric_q, channel_q
        ),
        "A3_dual_mean": build_signal(
            data, masked_q, dual_q, channel_q
        ),
    }
    results = {}
    for method, signal in methods.items():
        results[method] = evaluate_signal(
            data,
            signal,
            None,
            f"E19_{DATASET_NAME}_{method}",
            external=False,
            use_iforest=False,
        )
        np.save(
            OUT / f"e19_{DATASET_NAME}_{method}_signal.npy",
            signal,
        )
    np.save(
        OUT / f"e19_{DATASET_NAME}_pca_manifold_quantile.npy",
        pca_q,
    )
    np.save(
        OUT / f"e19_{DATASET_NAME}_learned_manifold_quantile.npy",
        metric_q,
    )
    np.save(
        OUT / f"e19_{DATASET_NAME}_dual_mean_manifold_quantile.npy",
        dual_q,
    )
    summary = {
        "dataset": DATASET_NAME,
        "model_file": str(model_path),
        "protocol": {
            "evaluation": "complete TS only",
            "iforest_included": False,
            "labels_used_for_training": False,
        },
        "pca_explained_variance": pca_explained.tolist(),
        "pca_diagnostics": pca_diagnostics,
        "learned_diagnostics": metric_diagnostics,
        "metric_losses": metric_losses,
        "results": results,
    }
    path = OUT / f"e19_{DATASET_NAME}_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
