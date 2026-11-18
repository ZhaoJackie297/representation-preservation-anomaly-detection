from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
OUT.mkdir(parents=True, exist_ok=True)

K_GRID = [3, 5, 10, 20, 30, 50]
PRIMARY_K = 10
ALPHA = 0.4
COOLDOWN_DAYS = 7

sys.path.insert(0, str(ROOT))
from application_utility_analysis import PROFILES  # noqa: E402
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from production_protocol_r42 import build_alerts  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    daily_percentile,
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from train_r42_ts2vec import TS2Vec  # noqa: E402


ENCODE_ARGS = {
    "causal": True,
    "sliding_length": 1,
    "sliding_padding": 30,
    "batch_size": 32,
}


def load_model(model_path: Path, input_dims: int) -> TS2Vec:
    device = 0 if torch.cuda.is_available() else "cpu"
    model = TS2Vec(
        input_dims=input_dims,
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        batch_size=16,
    )
    load_device = (
        f"cuda:{device}" if isinstance(device, int) else "cpu"
    )
    state_dict = torch.load(
        model_path,
        map_location=load_device,
        weights_only=True,
    )
    model.net.load_state_dict(state_dict)
    model.net.eval()
    return model


def encode_representations(
    model: TS2Vec,
    values: np.ndarray,
    *,
    need_masked: bool,
) -> tuple[np.ndarray | None, np.ndarray]:
    masked = None
    if need_masked:
        print("  encoding masked representations", flush=True)
        masked = np.asarray(
            model.encode(
                values,
                mask="mask_last",
                **ENCODE_ARGS,
            ),
            dtype=np.float32,
        )
    print("  encoding unmasked representations", flush=True)
    plain = np.asarray(
        model.encode(values, **ENCODE_ARGS),
        dtype=np.float32,
    )
    return masked, plain


def latent_deviation(
    representations: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Current representation distance from each user's training profile.

    The center is estimated only from active training days for that user.
    Residual dimensions are scaled by pooled training residual RMS. This
    avoids per-user covariance estimation and an arbitrary shrinkage value.
    """
    users, time_steps, dimensions = representations.shape
    if dimensions < 1:
        raise ValueError("Representations must have at least one dimension.")

    train_mask = active_mask[:, : train_end + 1].copy()
    finite = np.isfinite(representations).all(axis=2)
    train_mask &= finite[:, : train_end + 1]
    global_rows = representations[:, : train_end + 1][train_mask]
    if len(global_rows) == 0:
        raise RuntimeError("No finite active training representations.")
    global_center = global_rows.mean(axis=0)

    centers = np.full(
        (users, dimensions), np.nan, dtype=np.float64
    )
    for user_index in range(users):
        training_indices = np.flatnonzero(train_mask[user_index])
        if len(training_indices):
            centers[user_index] = representations[
                user_index, training_indices
            ].mean(axis=0)
        else:
            centers[user_index] = global_center
    centers = np.where(np.isfinite(centers), centers, global_center)

    residuals = []
    for user_index in range(users):
        training_indices = np.flatnonzero(train_mask[user_index])
        if len(training_indices) == 0:
            continue
        residuals.append(
            representations[user_index, training_indices]
            - centers[user_index]
        )
    residual_rows = np.concatenate(residuals, axis=0)
    scale = np.sqrt(
        np.mean(np.square(residual_rows), axis=0)
    )
    scale = np.where(
        np.isfinite(scale) & (scale >= 1e-8),
        scale,
        1.0,
    )

    distances = np.full(
        active_mask.shape, np.nan, dtype=np.float32
    )
    for user_index in range(users):
        active_indices = np.flatnonzero(active_mask[user_index])
        if len(active_indices) == 0:
            continue
        standardized = (
            representations[user_index, active_indices]
            - centers[user_index]
        ) / scale
        valid = np.isfinite(standardized).all(axis=1)
        distances[
            user_index, active_indices[valid]
        ] = np.sqrt(
            np.mean(np.square(standardized[valid]), axis=1)
        )
    return distances, centers.astype(np.float32)


def build_ts_variants(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    train_end: int,
    masked_error: np.ndarray,
    latent_raw: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    quantiles = {
        "mask": per_user_quantile_scores(
            masked_error, active_mask, train_end
        ),
        "latent": per_user_quantile_scores(
            latent_raw, active_mask, train_end
        ),
    }
    quantiles["dual_mean"] = (
        quantiles["mask"] + quantiles["latent"]
    ) / 2.0
    quantiles["dual_max"] = np.maximum(
        quantiles["mask"], quantiles["latent"]
    )

    signals = {}
    dynamic_scores = {}
    for name, quantile in quantiles.items():
        dynamic = compute_risk(
            users,
            dates,
            quantile.astype(np.float32),
            active_mask,
        )
        dynamic_scores[name] = dynamic
        signals[name] = daily_percentile(dynamic, active_mask)
    return signals, {
        **quantiles,
        **{
            f"{name}_dynamic": dynamic
            for name, dynamic in dynamic_scores.items()
        },
    }


def evaluate_ts_signal(
    data: dict,
    ts_signal: np.ndarray,
    dual_view_auxiliary: np.ndarray,
    method: str,
    *,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        q = q_from_alpha(k, ALPHA)
        alerts = build_unique_quota_alerts(
            ts_signal,
            dual_view_auxiliary,
            data["active_mask"],
            k,
            q,
        )
        result = evaluate_alert_system(
            data,
            alerts,
            method,
            k,
            external=external,
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


def evaluate_complete_signal(
    data: dict,
    signal: np.ndarray,
    method: str,
    *,
    external: bool,
) -> pd.DataFrame:
    rows = []
    for k in K_GRID:
        alerts = build_alerts(
            signal,
            data["active_mask"],
            k,
            COOLDOWN_DAYS,
        )
        result = evaluate_alert_system(
            data,
            alerts,
            method,
            k,
            external=external,
        )
        splits = ["external_test"] if external else ["validation", "test"]
        for split in splits:
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "k": k,
                    "q": 0,
                    **compact_metric(
                        result, None if external else split
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_application_utilities(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    for profile_name, weights in PROFILES.items():
        metrics[profile_name] = sum(
            metrics[column] * weight
            for column, weight in weights.items()
        )
    return metrics


def summarize_rank_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    maxima = rankings.groupby(
        ["data_block", "split", "profile"]
    )["utility"].transform("max")
    rankings = rankings.copy()
    rankings["relative_utility"] = rankings["utility"] / maxima
    rankings["regret"] = 1.0 - rankings["relative_utility"]
    return (
        rankings.groupby("method", as_index=False)
        .agg(
            appearances=("rank", "size"),
            first_place=(
                "rank",
                lambda values: int((values == 1).sum()),
            ),
            top_three=(
                "rank",
                lambda values: int((values <= 3).sum()),
            ),
            mean_rank=("rank", "mean"),
            worst_rank=("rank", "max"),
            mean_utility=("utility", "mean"),
            mean_relative_utility=("relative_utility", "mean"),
            worst_relative_utility=("relative_utility", "min"),
            mean_regret=("regret", "mean"),
            worst_regret=("regret", "max"),
        )
        .sort_values(
            ["mean_rank", "mean_regret"],
            ascending=[True, True],
        )
    )


def build_k10_rankings(k10: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (data_block, split), group in k10.groupby(
        ["data_block", "split"]
    ):
        for profile_name in PROFILES:
            ordered = group.sort_values(
                [profile_name, "precision_at_k"],
                ascending=False,
            ).copy()
            ordered["rank"] = np.arange(1, len(ordered) + 1)
            for row in ordered.itertuples(index=False):
                rows.append(
                    {
                        "data_block": data_block,
                        "split": split,
                        "profile": profile_name,
                        "method": row.method,
                        "utility": float(getattr(row, profile_name)),
                        "rank": int(row.rank),
                        "event_recall_3d": float(row.event_recall_3d),
                        "scenario_macro_recall_3d": float(
                            row.scenario_macro_recall_3d
                        ),
                        "precision_at_k": float(row.precision_at_k),
                    }
                )
    return pd.DataFrame(rows)


def json_ready(frame: pd.DataFrame) -> list[dict[str, object]]:
    records = frame.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, np.generic):
                record[key] = value.item()
    return records


def run_block(
    block_name: str,
    data: dict,
    values: np.ndarray,
    model_path: Path,
    existing_signal: np.ndarray,
    existing_masked_error: np.ndarray | None,
    original_iforest: np.ndarray,
    augmented_iforest: np.ndarray,
    *,
    external: bool,
    saved_arrays: dict[str, np.ndarray],
    cached_latent: np.ndarray | None = None,
    cached_latent_centers: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"\nEvaluating {block_name}", flush=True)
    if cached_latent is None:
        model = load_model(model_path, values.shape[-1])
        masked, plain = encode_representations(
            model,
            values,
            need_masked=existing_masked_error is None,
        )
        if existing_masked_error is None:
            if masked is None:
                raise RuntimeError("Masked representation was not encoded.")
            masked_error = np.abs(masked - plain).sum(axis=-1)
        else:
            masked_error = existing_masked_error.astype(np.float32)
        latent_raw, latent_centers = latent_deviation(
            plain,
            data["active_mask"],
            int(data["train_end"])
            if "train_end" in data
            else int(data["test_start"]) - 1,
        )
        representation_dims = int(plain.shape[-1])
    else:
        print("  reusing cached latent representations", flush=True)
        if existing_masked_error is None:
            raise RuntimeError(
                "Cached latent reuse requires an existing masked score."
            )
        if cached_latent_centers is None:
            raise RuntimeError("Cached latent centers are missing.")
        masked_error = existing_masked_error.astype(np.float32)
        latent_raw = cached_latent.astype(np.float32)
        latent_centers = cached_latent_centers.astype(np.float32)
        representation_dims = int(latent_centers.shape[-1])
    signals, scores = build_ts_variants(
        data["users"],
        data["dates"],
        data["active_mask"],
        int(data["train_end"])
        if "train_end" in data
        else int(data["test_start"]) - 1,
        masked_error,
        latent_raw,
    )
    recomputed_mask_diff = float(
        np.nanmax(
            np.abs(signals["mask"] - existing_signal)
        )
    )

    auxiliary = np.maximum(original_iforest, augmented_iforest).astype(
        np.float32
    )
    frames = [
        evaluate_complete_signal(
            data,
            original_iforest,
            "iforest_complete",
            external=external,
        )
    ]
    for variant in ("mask", "latent", "dual_mean", "dual_max"):
        frames.append(
            evaluate_complete_signal(
                data,
                signals[variant],
                f"ts_{variant}_complete",
                external=external,
            )
        )
        frames.append(
            evaluate_ts_signal(
                data,
                signals[variant],
                auxiliary,
                f"ts_{variant}_dual_iforest",
                external=external,
            )
        )
    metrics = pd.concat(frames, ignore_index=True)
    metrics.insert(0, "data_block", block_name)
    metrics["external"] = int(external)

    saved_arrays[f"{block_name}_masked_error"] = masked_error.astype(
        np.float32
    )
    saved_arrays[f"{block_name}_latent_raw"] = latent_raw.astype(
        np.float32
    )
    saved_arrays[f"{block_name}_latent_centers"] = latent_centers
    saved_arrays[f"{block_name}_dual_view_iforest"] = auxiliary
    for name, signal in signals.items():
        saved_arrays[f"{block_name}_ts_{name}_signal"] = signal.astype(
            np.float32
        )
        saved_arrays[
            f"{block_name}_ts_{name}_quantile"
        ] = scores[name].astype(np.float32)

    diagnostics = {
        "block": block_name,
        "users": int(values.shape[0]),
        "time_steps": int(values.shape[1]),
        "representation_dims": representation_dims,
        "masked_signal_max_abs_diff_vs_existing": recomputed_mask_diff,
        "latent_raw_active_min": float(np.nanmin(latent_raw)),
        "latent_raw_active_median": float(np.nanmedian(latent_raw)),
        "latent_raw_active_max": float(np.nanmax(latent_raw)),
    }
    if cached_latent is None:
        del model, masked, plain
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, diagnostics


def main() -> None:
    np.random.seed(20261109)
    torch.manual_seed(20261109)

    print("Loading saved dual-view IForest scores", flush=True)
    view_scores = np.load(
        OUT / "iforest_view_ensemble_scores.npz",
        allow_pickle=False,
    )
    retrained_scores = np.load(
        ROOT
        / "processed_r42"
        / "innovation_sim"
        / "r4.2_quota_retrained_scores.npz",
        allow_pickle=False,
    )

    print("Loading CERT data", flush=True)
    r42 = load_r42()
    r52 = load_r52()
    r42_dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    r52_dataset = np.load(
        ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    r42_scores = np.load(
        ROOT / "processed_r42" / "r4.2_ts2vec_scores.npz",
        allow_pickle=False,
    )
    r52_scores = np.load(
        ROOT / "processed_r52" / "r5.2_ts2vec_scores.npz",
        allow_pickle=False,
    )

    retrained_data = dict(r42)
    retrained_data["ts"] = retrained_scores[
        "ts2vec_signal"
    ].astype(np.float32)

    saved_arrays: dict[str, np.ndarray] = {}
    all_metrics = []
    diagnostics = []
    cache: dict[str, np.ndarray] = {}
    cache_path = OUT / "ts2vec_dual_score_arrays.npz"
    if "--reuse-latent" in sys.argv and cache_path.exists():
        print("Reusing cached latent deviation arrays", flush=True)
        with np.load(cache_path, allow_pickle=False) as cached:
            cache = {key: cached[key].copy() for key in cached.files}

    tasks = [
        (
            "r42_original",
            r42,
            r42_dataset["values"].astype(np.float32),
            ROOT / "processed_r42" / "r4.2_ts2vec_model.pth",
            r42["ts"],
            r42_scores["raw_error"].astype(np.float32),
            view_scores["r42_original"],
            view_scores["r42_augmented"],
            False,
        ),
        (
            "r42_retrained",
            retrained_data,
            r42_dataset["values"].astype(np.float32),
            ROOT
            / "processed_r42"
            / "innovation_sim"
            / "r4.2_ts2vec_quota_retrain.pth",
            retrained_scores["ts2vec_signal"].astype(np.float32),
            cache.get("r42_retrained_masked_error"),
            view_scores["retrained_original"],
            view_scores["retrained_augmented"],
            False,
        ),
        (
            "r52_external",
            r52,
            r52_dataset["values"].astype(np.float32),
            ROOT / "processed_r52" / "r5.2_ts2vec_model.pth",
            r52["ts"],
            r52_scores["anomaly_score"].astype(np.float32),
            view_scores["r52_original"],
            view_scores["r52_augmented"],
            True,
        ),
    ]
    for (
        block_name,
        data,
        values,
        model_path,
        existing_signal,
        existing_error,
        original_iforest,
        augmented_iforest,
        external,
    ) in tasks:
        metrics, diagnostic = run_block(
            block_name,
            data,
            values,
            model_path,
            existing_signal,
            existing_error,
            original_iforest,
            augmented_iforest,
            external=external,
            saved_arrays=saved_arrays,
            cached_latent=(
                cache.get(f"{block_name}_latent_raw")
                if cache
                else None
            ),
            cached_latent_centers=(
                cache.get(f"{block_name}_latent_centers")
                if cache
                else None
            ),
        )
        all_metrics.append(metrics)
        diagnostics.append(diagnostic)

    metrics = add_application_utilities(
        pd.concat(all_metrics, ignore_index=True)
    )
    metrics.to_csv(
        OUT / "ts2vec_dual_score_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    k10 = metrics[metrics["k"] == PRIMARY_K].copy()
    k10.to_csv(
        OUT / "ts2vec_dual_score_k10_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    rankings = build_k10_rankings(k10)
    rankings.to_csv(
        OUT / "ts2vec_dual_score_k10_rankings.csv",
        index=False,
        encoding="utf-8",
    )
    stability = summarize_rank_stability(rankings)
    stability.to_csv(
        OUT / "ts2vec_dual_score_k10_rank_stability.csv",
        index=False,
        encoding="utf-8",
    )

    profile_orders = (
        rankings.sort_values(
            ["data_block", "split", "profile", "rank"]
        )
        .groupby(
            ["data_block", "split", "profile"],
            as_index=False,
        )
        .apply(
            lambda group: pd.Series(
                {"order": " > ".join(group["method"])}
            ),
            include_groups=False,
        )
    )
    profile_rankings_identical = bool(
        (
            profile_orders.groupby(["data_block", "split"])["order"]
            .nunique()
            == 1
        ).all()
    )

    np.savez_compressed(
        OUT / "ts2vec_dual_score_arrays.npz",
        **saved_arrays,
    )
    summary = {
        "experiment": "TS2Vec masked-difference and latent-deviation scores",
        "primary_k": PRIMARY_K,
        "k_grid": K_GRID,
        "q_relation": "q=min(max(1,round(0.4K)),floor(K/2))",
        "latent_distance": (
            "sqrt(mean(((z_t - user_training_center) / "
            "pooled_training_residual_rms)^2))"
        ),
        "latent_profile_scope": "active training days only",
        "score_combination_operators": ["mean", "max"],
        "operator_selection": (
            "Both fixed combinations are reported; no validation or test "
            "labels are used to choose a combination operator."
        ),
        "profiles": PROFILES,
        "profile_rankings_identical_at_k10": profile_rankings_identical,
        "labels_used_for_model_fit": False,
        "labels_used_for_hyperparameter_selection": False,
        "diagnostics": diagnostics,
        "k10_rank_stability": json_ready(stability),
        "outputs": {
            "metrics": str(OUT / "ts2vec_dual_score_metrics.csv"),
            "k10_metrics": str(
                OUT / "ts2vec_dual_score_k10_metrics.csv"
            ),
            "k10_rankings": str(
                OUT / "ts2vec_dual_score_k10_rankings.csv"
            ),
            "k10_rank_stability": str(
                OUT / "ts2vec_dual_score_k10_rank_stability.csv"
            ),
            "arrays": str(OUT / "ts2vec_dual_score_arrays.npz"),
        },
    }
    (OUT / "ts2vec_dual_score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nK=10 rank stability", flush=True)
    print(stability.to_string(index=False), flush=True)
    print("\nK=10 winners", flush=True)
    winners = rankings[rankings["rank"] == 1]
    print(
        winners.sort_values(
            ["data_block", "split", "profile"]
        ).to_string(index=False),
        flush=True,
    )
    print(
        "\nProfile rankings identical at K=10: "
        f"{profile_rankings_identical}",
        flush=True,
    )


if __name__ == "__main__":
    main()
