from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
RETRAIN = ROOT / "processed_r42" / "innovation_sim"
SEED = 20261109
_ORIGINAL_VIEW_CACHE: dict[tuple, np.ndarray] = {}

sys.path.insert(0, str(ROOT))
from feature_augmented_iforest_sim import (  # noqa: E402
    K_GRID,
    augment,
    build_unique_quota_alerts,
    load_feature_values,
    train_iforest,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)


def fit_view_scores(
    data: dict,
    feature_file: Path,
    ts_dynamic: np.ndarray,
    personal_signal: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    import feature_augmented_iforest_sim as source

    source.users = data["users"]
    source.dates = data["dates"]
    base_values = load_feature_values(data, feature_file)
    train_end = (
        data["train_end"]
        if "train_end" in data
        else data["test_start"] - 1
    )
    cache_key = (
        str(Path(feature_file).resolve()),
        str(data["users"][0]),
        str(data["users"][-1]),
        tuple(data["active_mask"].shape),
        str(pd.Timestamp(data["dates"][0])),
        str(pd.Timestamp(data["dates"][-1])),
        int(train_end),
        int(seed),
    )
    original = _ORIGINAL_VIEW_CACHE.get(cache_key)
    if original is None:
        original = train_iforest(
            base_values,
            data["active_mask"],
            train_end,
            seed,
        )
        _ORIGINAL_VIEW_CACHE[cache_key] = original
    augmented_values = augment(
        base_values,
        [ts_dynamic[..., None], personal_signal[..., None]],
    )
    augmented = train_iforest(
        augmented_values,
        data["active_mask"],
        train_end,
        seed,
    )
    return original, augmented


def combine_views(
    original: np.ndarray,
    augmented: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "mean":
        return (original + augmented) / 2.0
    if method == "max":
        return np.maximum(original, augmented)
    if method == "min":
        return np.minimum(original, augmented)
    raise ValueError(method)


def evaluate_aux(
    data: dict,
    ts_signal: np.ndarray,
    aux_signal: np.ndarray,
    method: str,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        q = q_from_alpha(k, 0.4)
        alerts = build_unique_quota_alerts(
            ts_signal,
            aux_signal,
            data["active_mask"],
            k,
            q,
        )
        result = evaluate_alert_system(
            data, alerts, method, k, external=external
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "k": k,
                    "q": q,
                    **compact_metric(
                        result, None if external else split
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    r42 = load_r42()
    calibrated42 = np.load(
        ROOT / "processed_r42" / "r4.2_calibrated_method_scores.npz",
        allow_pickle=False,
    )
    original42, augmented42 = fit_view_scores(
        r42,
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        calibrated42["ts2vec_quantile_dynamic"].astype(np.float32),
        r42["personal"],
        SEED,
    )

    combination_methods = {
        "iforest_original": original42,
        "iforest_augmented": augmented42,
        "iforest_view_mean": combine_views(
            original42, augmented42, "mean"
        ),
        "iforest_view_max": combine_views(
            original42, augmented42, "max"
        ),
        "iforest_view_min": combine_views(
            original42, augmented42, "min"
        ),
    }

    validation_rows = []
    validation_results = {}
    for method, aux_signal in combination_methods.items():
        metrics = evaluate_aux(
            r42, r42["ts"], aux_signal, method, external=False
        )
        validation = metrics[metrics["split"] == "validation"]
        validation_results[method] = metrics
        validation_rows.append(
            {
                "method": method,
                "mean_validation_recall": validation[
                    "event_recall_3d"
                ].mean(),
                "mean_validation_precision": validation[
                    "precision_at_k"
                ].mean(),
                "mean_validation_macro": validation[
                    "scenario_macro_recall_3d"
                ].mean(),
            }
        )
    validation_summary = pd.DataFrame(validation_rows).sort_values(
        [
            "mean_validation_recall",
            "mean_validation_precision",
            "mean_validation_macro",
        ],
        ascending=False,
    )
    selected_method = str(validation_summary.iloc[0]["method"])
    validation_summary.to_csv(
        OUT / "iforest_view_ensemble_validation_summary.csv",
        index=False,
        encoding="utf-8",
    )

    r52 = load_r52()
    scores52 = np.load(
        ROOT / "processed_r52" / "r5.2_conditional_combination_scores.npz",
        allow_pickle=False,
    )
    original52, augmented52 = fit_view_scores(
        r52,
        ROOT / "processed_r52" / "r5.2_user_day_features.csv",
        scores52["ts2vec_dynamic"].astype(np.float32),
        scores52["zscore_signal"].astype(np.float32),
        SEED,
    )
    selected_external = evaluate_aux(
        r52,
        r52["ts"],
        (
            original52
            if selected_method == "iforest_original"
            else augmented52
            if selected_method == "iforest_augmented"
            else combine_views(
                original52,
                augmented52,
                selected_method.replace("iforest_view_", ""),
            )
        ),
        selected_method,
        external=True,
    )

    retrained = np.load(
        RETRAIN / "r4.2_quota_retrained_scores.npz",
        allow_pickle=False,
    )
    original_retrain, augmented_retrain = fit_view_scores(
        r42,
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        retrained["ts2vec_dynamic"].astype(np.float32),
        r42["personal"],
        SEED,
    )
    retrain_data = dict(r42)
    retrain_data["ts"] = retrained["ts2vec_signal"].astype(np.float32)
    selected_retrain = evaluate_aux(
        retrain_data,
        retrain_data["ts"],
        (
            original_retrain
            if selected_method == "iforest_original"
            else augmented_retrain
            if selected_method == "iforest_augmented"
            else combine_views(
                original_retrain,
                augmented_retrain,
                selected_method.replace("iforest_view_", ""),
            )
        ),
        selected_method,
        external=False,
    )

    final = pd.concat(
        [
            validation_results[selected_method],
            selected_external,
            selected_retrain.assign(
                method="retrained_" + selected_method
            ),
        ],
        ignore_index=True,
    )
    final.to_csv(
        OUT / "iforest_view_ensemble_final_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "iforest_view_ensemble_scores.npz",
        r42_users=r42["users"],
        r42_dates=r42["dates"].strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        r42_original=original42,
        r42_augmented=augmented42,
        r52_users=r52["users"],
        r52_dates=r52["dates"].strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        r52_original=original52,
        r52_augmented=augmented52,
        retrained_original=original_retrain,
        retrained_augmented=augmented_retrain,
    )
    summary = {
        "selected_method": selected_method,
        "selection_split": "r4.2 original validation",
        "q_relation": "q=min(max(1,round(0.4K)),floor(K/2))",
        "test_external_used_for_selection": False,
        "labels_used_for_model_fit": False,
        "validation_summary": str(
            OUT / "iforest_view_ensemble_validation_summary.csv"
        ),
        "final_metrics": str(
            OUT / "iforest_view_ensemble_final_metrics.csv"
        ),
        "scores_file": str(
            OUT / "iforest_view_ensemble_scores.npz"
        ),
    }
    (OUT / "iforest_view_ensemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(validation_summary.to_string(index=False), flush=True)
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
