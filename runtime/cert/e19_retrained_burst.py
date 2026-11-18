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
from quota_relationship_sim import daily_percentile, load_r42  # noqa: E402
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
    burst_q = np.load(
        OUT / "e19_r42_hourly_burst_quantile.npy"
    )
    relation_q = np.load(
        OUT / "e19_r42_relation_quantile.npy"
    )
    core_q = (
        arrays["masked_quantile"]
        + dual_q
        + arrays["channel_top2_quantile"]
    ) / 3.0
    consensus_q = np.minimum(burst_q, relation_q)
    final_q = np.maximum(core_q, consensus_q)
    signals = {
        "A3_dual_mean": np.load(
            OUT / "e19_r42_retrained_A3_dual_mean_signal.npy"
        ),
        "C2_burst_relation_consensus": make_signal(
            data, final_q
        ),
    }
    results = {}
    for method, signal in signals.items():
        results[method] = evaluate_signal(
            data,
            signal,
            None,
            f"E19_r42_retrained_{method}",
            external=False,
            use_iforest=False,
        )
        np.save(
            OUT
            / f"e19_r42_retrained_{method}_signal.npy",
            signal,
        )
    summary = {
        "dataset": "r42_retrained",
        "protocol": {
            "evaluation": "complete TS only",
            "iforest_included": False,
            "fusion": (
                "max(core quantile, min(burst quantile, "
                "relation quantile))"
            ),
            "labels_used_for_training": False,
        },
        "results": results,
    }
    path = OUT / "e19_r42_retrained_burst_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
