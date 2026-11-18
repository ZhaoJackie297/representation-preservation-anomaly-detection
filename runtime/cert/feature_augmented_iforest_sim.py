from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
OUT.mkdir(parents=True, exist_ok=True)

K_GRID = [3, 5, 10, 20, 30, 50]
ALPHA = 0.4
SEED = 20261109

sys.path.insert(0, str(ROOT))
from baseline_isolation_forest_r42 import FEATURES, LOG1P_FEATURES  # noqa: E402
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from risk_topk_r42 import compute_risk  # noqa: E402
from production_protocol_r42 import label_matrix  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    daily_percentile,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)


def standardize_augmented_features(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    train_active = active_mask[:, : train_end + 1]
    train_rows = values[:, : train_end + 1][train_active]
    mean = train_rows.mean(axis=0)
    std = train_rows.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return ((values - mean) / std).astype(np.float32)


def train_iforest(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    seed: int,
) -> np.ndarray:
    normalized = standardize_augmented_features(
        values, active_mask, train_end
    )
    train_active = active_mask[:, : train_end + 1]
    train_rows = normalized[:, : train_end + 1][train_active]
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(train_rows)
    raw_score = np.full(
        active_mask.shape, np.nan, dtype=np.float32
    )
    raw_score[active_mask] = -model.score_samples(
        normalized[active_mask]
    )
    quantile = per_user_quantile_scores(
        raw_score, active_mask, train_end
    )
    dynamic = compute_risk(
        users,
        dates,
        quantile,
        active_mask,
    )
    return daily_percentile(dynamic, active_mask)


def build_unique_quota_alerts(
    ts_score: np.ndarray,
    aux_score: np.ndarray,
    active_mask: np.ndarray,
    k: int,
    q_limit: int,
) -> np.ndarray:
    alerts = np.zeros(ts_score.shape, dtype=np.uint8)
    last_alert = np.full(ts_score.shape[0], -10_000, dtype=np.int32)
    for time_index in range(ts_score.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= 7
        )
        candidates = np.flatnonzero(available)
        if len(candidates) == 0:
            continue
        ts_order = candidates[
            np.lexsort(
                (candidates, -ts_score[candidates, time_index])
            )
        ]
        aux_order = candidates[
            np.lexsort(
                (candidates, -aux_score[candidates, time_index])
            )
        ]
        selected = ts_order[: k - q_limit].astype(int).tolist()
        ts_top_k = set(ts_order[:k].tolist())
        for user_index in aux_order:
            user_index = int(user_index)
            if user_index in selected or user_index in ts_top_k:
                continue
            selected.append(user_index)
            if len(selected) >= k:
                break
        if len(selected) < k:
            for user_index in ts_order:
                if int(user_index) not in selected:
                    selected.append(int(user_index))
                if len(selected) >= k:
                    break
        selected_array = np.asarray(selected[:k], dtype=np.int64)
        alerts[selected_array, time_index] = 1
        last_alert[selected_array] = time_index
    return alerts


def evaluate_variant(
    data: dict,
    ts_score: np.ndarray,
    aux_score: np.ndarray,
    variant: str,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        q = q_from_alpha(k, ALPHA)
        alerts = build_unique_quota_alerts(
            ts_score, aux_score, data["active_mask"], k, q
        )
        result = evaluate_alert_system(
            data,
            alerts,
            variant,
            k,
            external=external,
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            metric = compact_metric(
                result, None if external else split
            )
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "k": k,
                    "q": q,
                    **metric,
                }
            )
    return pd.DataFrame(rows)


def load_feature_values(data: dict, feature_csv: Path) -> np.ndarray:
    raw = load_user_day_frame(feature_csv)
    index = pd.MultiIndex.from_product(
        [data["users"], data["dates"]], names=["user_id", "date"]
    )
    aligned = raw.set_index(["user_id", "date"]).reindex(index)
    values = aligned[FEATURES].to_numpy(dtype=np.float64).reshape(
        len(data["users"]), len(data["dates"]), len(FEATURES)
    )
    for feature_index, feature in enumerate(FEATURES):
        if feature in LOG1P_FEATURES:
            finite = np.isfinite(values[:, :, feature_index])
            values[:, :, feature_index][finite] = np.log1p(
                values[:, :, feature_index][finite]
            )
    return values.astype(np.float32)


@lru_cache(maxsize=4)
def load_user_day_frame(feature_file: Path) -> pd.DataFrame:
    """Read each user-day table once per V3 process.

    V3 runs S2 and S4 in one process. Both stages need the same feature table,
    so a small process-local cache avoids converting the same large CSV twice.
    Parquet is preferred because it preserves the S1 numeric dtypes.
    """

    feature_file = Path(feature_file)
    parquet_file = feature_file.with_suffix(".parquet")
    if parquet_file.exists():
        return pd.read_parquet(parquet_file)
    return pd.read_csv(feature_file, parse_dates=["date"])


def augment(
    base_values: np.ndarray,
    extra_arrays: list[np.ndarray],
) -> np.ndarray:
    return np.concatenate(
        [base_values, *[array.astype(np.float32) for array in extra_arrays]],
        axis=-1,
    )


def variant_scores(
    base_values: np.ndarray,
    ts_quantile: np.ndarray,
    ts_dynamic: np.ndarray,
    personal_signal: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    seed: int,
) -> dict[str, np.ndarray]:
    variants = {}
    feature_sets = {
        "iforest_original23": [base_values],
        "iforest_tsbase24": [base_values, ts_quantile[..., None]],
        "iforest_tsdynamic24": [base_values, ts_dynamic[..., None]],
        "iforest_tsbase_personal25": [
            base_values,
            ts_quantile[..., None],
            personal_signal[..., None],
        ],
        "iforest_tsdynamic_personal25": [
            base_values,
            ts_dynamic[..., None],
            personal_signal[..., None],
        ],
    }
    for name, parts in feature_sets.items():
        values = parts[0] if len(parts) == 1 else augment(
            parts[0], parts[1:]
        )
        variants[name] = train_iforest(
            values, active_mask, train_end, seed
        )
    return variants


def main() -> None:
    global users, dates
    r42 = load_r42()
    r52 = load_r52()
    users = r42["users"]
    dates = r42["dates"]

    r42_features = load_feature_values(
        r42,
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
    )
    calibrated42 = np.load(
        ROOT / "processed_r42" / "r4.2_calibrated_method_scores.npz",
        allow_pickle=False,
    )
    r42_variants = variant_scores(
        r42_features,
        calibrated42["ts2vec_quantile"].astype(np.float32),
        calibrated42["ts2vec_quantile_dynamic"].astype(np.float32),
        r42["personal"],
        r42["active_mask"],
        r42["train_end"],
        SEED,
    )

    validation_rows = []
    validation_metrics = {}
    for name, aux_score in r42_variants.items():
        metrics = evaluate_variant(
            r42, r42["ts"], aux_score, name, external=False
        )
        validation = metrics[metrics["split"] == "validation"]
        validation_metrics[name] = metrics
        validation_rows.append(
            {
                "variant": name,
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
    selected_variant = str(
        validation_summary.iloc[0]["variant"]
    )
    validation_summary.to_csv(
        OUT / "feature_augmented_iforest_validation_summary.csv",
        index=False,
        encoding="utf-8",
    )
    pd.concat(
        validation_metrics.values(), ignore_index=True
    ).to_csv(
        OUT / "feature_augmented_iforest_validation_grid.csv",
        index=False,
        encoding="utf-8",
    )

    users = r52["users"]
    dates = r52["dates"]
    r52_features = load_feature_values(
        r52,
        ROOT / "processed_r52" / "r5.2_user_day_features.csv",
    )
    scores52 = np.load(
        ROOT / "processed_r52" / "r5.2_conditional_combination_scores.npz",
        allow_pickle=False,
    )
    r52_variants = variant_scores(
        r52_features,
        scores52["ts2vec_quantile"].astype(np.float32),
        scores52["ts2vec_dynamic"].astype(np.float32),
        scores52["zscore_signal"].astype(np.float32),
        r52["active_mask"],
        r52["test_start"] - 1,
        SEED,
    )

    final_rows = []
    final_rows.append(validation_metrics[selected_variant])
    selected_external = evaluate_variant(
        r52,
        r52["ts"],
        r52_variants[selected_variant],
        selected_variant,
        external=True,
    )
    final_rows.append(selected_external)
    final = pd.concat(final_rows, ignore_index=True)
    final.to_csv(
        OUT / "feature_augmented_iforest_final_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "selected_variant": selected_variant,
        "selection_split": "r4.2 validation",
        "q_relation": "q=min(max(1,round(0.4K)),floor(K/2))",
        "test_and_external_used_for_selection": False,
        "labels_used_for_model_fit": False,
        "validation_summary": str(
            OUT / "feature_augmented_iforest_validation_summary.csv"
        ),
        "final_metrics": str(
            OUT / "feature_augmented_iforest_final_metrics.csv"
        ),
    }
    (OUT / "feature_augmented_iforest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(validation_summary.to_string(index=False), flush=True)
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
