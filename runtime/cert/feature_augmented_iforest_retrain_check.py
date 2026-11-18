from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "processed_r42" / "innovation_sim"

sys.path.insert(0, str(ROOT))
from feature_augmented_iforest_sim import (  # noqa: E402
    K_GRID,
    augment,
    build_unique_quota_alerts,
    evaluate_variant,
    load_feature_values,
    train_iforest,
)
from quota_relationship_sim import (  # noqa: E402
    load_r42,
    q_from_alpha,
)


def main() -> None:
    data = load_r42()
    retrained = np.load(
        OUT / "r4.2_quota_retrained_scores.npz",
        allow_pickle=False,
    )
    calibrated = np.load(
        ROOT / "processed_r42" / "r4.2_calibrated_method_scores.npz",
        allow_pickle=False,
    )
    base_values = load_feature_values(
        data,
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
    )
    personal_signal = data["personal"]
    augmented = augment(
        base_values,
        [
            retrained["ts2vec_dynamic"][..., None],
            personal_signal[..., None],
        ],
    )

    import feature_augmented_iforest_sim as source

    source.users = data["users"]
    source.dates = data["dates"]
    aux_score = train_iforest(
        augmented,
        data["active_mask"],
        data["train_end"],
        20261109,
    )
    data["ts"] = retrained["ts2vec_signal"].astype(np.float32)

    metrics = evaluate_variant(
        data,
        data["ts"],
        aux_score,
        "iforest_tsdynamic_personal25",
        external=False,
    )
    metrics.to_csv(
        OUT / "feature_augmented_iforest_retrain_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "experiment": "Retrained TS2Vec with augmented IForest auxiliary",
        "variant": "IForest(23 features + TS dynamic + personal)",
        "retrain_threshold": "TS2Vec seed 20261111",
        "q_relation": "q=min(max(1,round(0.4K)),floor(K/2))",
        "selection_data": "r4.2 original validation variant selected previously",
        "test_used_for_selection": False,
        "metrics": str(
            OUT / "feature_augmented_iforest_retrain_metrics.csv"
        ),
    }
    (OUT / "feature_augmented_iforest_retrain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
