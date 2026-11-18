from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"

sys.path.insert(0, str(ROOT))
from e17_multigranular_mask import (  # noqa: E402
    build_application_scores,
    evaluate_signal,
    load_base_student,
)
from e19_relation_signature import build_quad_signal  # noqa: E402
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
    student, device = load_base_student(
        model_path, values.shape[-1]
    )
    print("Building retrained V2 components", flush=True)
    _, arrays = build_application_scores(
        student,
        values,
        data,
        train_end,
        device,
    )
    dual_q = np.load(
        OUT / "e19_r42_retrained_dual_mean_manifold_quantile.npy"
    )
    relation_q = np.load(
        OUT / "e19_r42_relation_quantile.npy"
    )
    stage_b_signal = build_quad_signal(
        data,
        arrays["masked_quantile"],
        dual_q,
        arrays["channel_top2_quantile"],
        relation_q,
    )
    stage_a_signal = np.load(
        OUT / "e19_r42_retrained_A3_dual_mean_signal.npy"
    )
    results = {
        "A3_dual_mean": evaluate_signal(
            data,
            stage_a_signal,
            None,
            "E19_r42_retrained_A3",
            external=False,
            use_iforest=False,
        ),
        "B1_relation_equal": evaluate_signal(
            data,
            stage_b_signal,
            None,
            "E19_r42_retrained_B1",
            external=False,
            use_iforest=False,
        ),
    }
    np.save(
        OUT / "e19_r42_retrained_B1_relation_equal_signal.npy",
        stage_b_signal,
    )
    summary = {
        "dataset": "r42_retrained",
        "protocol": {
            "evaluation": "complete TS only",
            "iforest_included": False,
            "fusion": (
                "masked + dual_manifold + channel_top2 + relation"
                " equal-weight daily quantiles"
            ),
            "labels_used_for_training": False,
        },
        "results": results,
    }
    path = OUT / "e19_r42_retrained_relation_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
