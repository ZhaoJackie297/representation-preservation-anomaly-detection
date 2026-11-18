from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e19_models"
SEEDS = [20261111, 20261112, 20261113]
BOUNDARY_QUANTILE = 0.95
TRAINING_DAY_STRIDE = 2

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from e19_user_manifold import (  # noqa: E402
    UserMetricProjector,
    fit_user_manifolds,
    project_with_model,
)
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
from quota_relationship_sim import load_r42, load_r52  # noqa: E402
from s2r3_hard_peer import train_hard_classifier  # noqa: E402
from s2r3_latent_synthetic import (  # noqa: E402
    build_signal,
    classifier_scores,
)
from v2_m_pipeline import add_utilities, evaluate_block  # noqa: E402


BLOCKS = {
    "r42_original": {
        "family": "r42",
        "plain": OUT / "e17_r42_M0_plain.npy",
        "masked": OUT / "e17_r42_M0_masked_quantile.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r42_original_channel_top2_quantile",
        ),
        "manifold_prefix": "r42",
        "metric_prefix": "r42",
        "feature_file": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        ),
        "external": False,
    },
    "r42_retrained": {
        "family": "r42",
        "plain": OUT / "e19_r42_retrained_plain.npy",
        "masked": OUT / "e19_r42_retrained_masked_quantile.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r42_retrained_channel_top2_quantile",
        ),
        "manifold_prefix": "r42_retrained",
        "metric_prefix": "r42_retrained",
        "feature_file": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        ),
        "external": False,
    },
    "r52_external": {
        "family": "r52",
        "plain": OUT / "e17_r52_M0_plain.npy",
        "masked": OUT / "e17_r52_M0_masked_quantile.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r52_external_channel_top2_quantile",
        ),
        "manifold_prefix": "r52",
        "metric_prefix": "r52",
        "feature_file": (
            ROOT / "processed_r52" / "r5.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        ),
        "external": True,
    },
}


def load_projection(
    config: dict,
    seed: int,
    input_dims: int,
    device: torch.device,
) -> UserMetricProjector:
    model = UserMetricProjector(input_dims, 24).to(device)
    model.load_state_dict(
        torch.load(
            MODEL_DIR
            / f"{config['metric_prefix']}_user_metric_{seed}.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    return model


def whitened_personal_space(
    projected: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users, time_steps, dimensions = projected.shape
    centers, inverse_covariances, _, _ = fit_user_manifolds(
        projected, active_mask, train_end
    )
    whiteners = np.empty_like(inverse_covariances)
    boundaries = np.empty(users, dtype=np.float32)
    for user in range(users):
        covariance = np.linalg.pinv(
            inverse_covariances[user], hermitian=True
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        whitener = (
            eigenvectors
            @ np.diag(1.0 / np.sqrt(eigenvalues))
            @ eigenvectors.T
        )
        whiteners[user] = whitener.astype(np.float32)
        training = np.flatnonzero(
            active_mask[user, : train_end + 1]
            & np.isfinite(
                projected[user, : train_end + 1]
            ).all(axis=-1)
        )
        centered = (
            projected[user, training] - centers[user]
        ).astype(np.float64) @ whitener
        radii = np.linalg.norm(centered, axis=1)
        boundaries[user] = np.quantile(
            radii, BOUNDARY_QUANTILE
        )
    return centers, whiteners, boundaries


def boundary_mixed_rows(
    projected: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    centers: np.ndarray,
    whiteners: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_mask = (
        active_mask[:, : train_end + 1]
        & np.isfinite(
            projected[:, : train_end + 1]
        ).all(axis=-1)
    )
    user_rows, date_rows = np.nonzero(train_mask)
    keep = (date_rows % TRAINING_DAY_STRIDE) == 0
    user_rows = user_rows[keep]
    date_rows = date_rows[keep]
    normal = projected[user_rows, date_rows].astype(np.float32)
    synthetic = np.empty_like(normal)
    alphas = np.ones(len(normal), dtype=np.float32)
    crossed = np.zeros(len(normal), dtype=bool)
    for date in np.unique(date_rows):
        positions = np.flatnonzero(date_rows == date)
        date_users = user_rows[positions]
        if len(positions) < 2:
            synthetic[positions] = normal[positions]
            continue
        for position in positions:
            user = int(user_rows[position])
            center = centers[user].astype(np.float64)
            whitener = whiteners[user].astype(np.float64)
            anchor = (
                normal[position].astype(np.float64) - center
            ) @ whitener
            candidate_positions = positions[date_users != user]
            candidate_vectors = (
                normal[candidate_positions].astype(np.float64)
                - center
            ) @ whitener
            target = max(
                float(boundaries[user]),
                float(np.linalg.norm(anchor) * 1.05),
            )
            best_alpha = None
            directions = candidate_vectors - anchor
            a_values = np.square(directions).sum(axis=1)
            b_values = 2.0 * (
                directions @ anchor
            )
            c_value = float(anchor @ anchor) - target * target
            discriminant = (
                b_values * b_values
                - 4.0 * a_values * c_value
            )
            valid = (
                (a_values >= 1e-12)
                & (discriminant >= 0)
            )
            roots = np.full(len(candidate_vectors), np.inf)
            roots[valid] = (
                -b_values[valid]
                + np.sqrt(discriminant[valid])
            ) / (2.0 * a_values[valid])
            valid &= (roots > 0.0) & (roots <= 1.0)
            if bool(valid.any()):
                valid_indices = np.flatnonzero(valid)
                best_index = valid_indices[
                    np.argmin(roots[valid_indices])
                ]
                best_alpha = float(roots[best_index])
                donor_row = candidate_positions[best_index]
            if best_alpha is not None:
                synthetic[position] = (
                    (1.0 - best_alpha) * normal[position]
                    + best_alpha * normal[donor_row]
                )
                alphas[position] = best_alpha
                crossed[position] = True
            else:
                farthest = candidate_positions[
                    int(
                        np.argmax(
                            np.linalg.norm(
                                candidate_vectors, axis=1
                            )
                        )
                    )
                ]
                synthetic[position] = normal[farthest]
    return normal, synthetic, {
        "rows": int(len(normal)),
        "boundary_crossed_fraction": float(crossed.mean()),
        "mean_alpha": float(alphas.mean()),
        "median_alpha": float(np.median(alphas)),
        "training_day_stride": TRAINING_DAY_STRIDE,
    }


def run_block(block_name: str) -> None:
    config = BLOCKS[block_name]
    data = load_r52() if config["family"] == "r52" else load_r42()
    dataset = np.load(config["sequence"], allow_pickle=False)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    plain = np.load(config["plain"], mmap_mode="r")
    masked_q = np.load(config["masked"])
    channel_store = np.load(config["channel"][0], allow_pickle=False)
    channel_q = channel_store[config["channel"][1]].astype(np.float32)
    manifold_q = np.mean(
        np.stack(
            [
                np.load(
                    OUT
                    / (
                        f"e19_{config['manifold_prefix']}_learned_"
                        f"manifold_quantile_{seed}.npy"
                    )
                )
                for seed in SEEDS
            ],
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    quantiles = []
    logs = {}
    diagnostics = {}
    for seed in SEEDS:
        print(f"{block_name}: boundary seed {seed}", flush=True)
        projection = load_projection(
            config, seed, plain.shape[-1], device
        )
        projected = project_with_model(projection, plain)
        centers, whiteners, boundaries = whitened_personal_space(
            projected, active_mask, train_end
        )
        normal, synthetic, diagnostic = boundary_mixed_rows(
            projected,
            active_mask,
            train_end,
            centers,
            whiteners,
            boundaries,
        )
        classifier, losses = train_hard_classifier(
            normal, synthetic, seed
        )
        raw = classifier_scores(classifier, projected)
        quantile = per_user_quantile_scores(
            raw, active_mask, train_end
        )
        quantiles.append(quantile)
        logs[str(seed)] = losses
        diagnostics[str(seed)] = diagnostic
        np.save(
            OUT / f"s2r3_boundary_{block_name}_raw_{seed}.npy",
            raw,
        )
        del projection, projected, classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    synthetic_q = np.mean(
        np.stack(quantiles, axis=0),
        axis=0,
    ).astype(np.float32)
    signal, dynamic = build_signal(
        data,
        masked_q,
        manifold_q,
        channel_q,
        synthetic_q,
    )
    original, augmented = fit_view_scores(
        data,
        config["feature_file"],
        dynamic,
        data["personal"],
        20261109,
    )
    auxiliary = np.maximum(original, augmented)
    metrics = add_utilities(
        evaluate_block(
            block_name,
            data,
            signal,
            auxiliary,
            "S2R3_boundary_calibrated",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s2r3_boundary_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / f"s2r3_boundary_{block_name}_components.npz",
        signal=signal,
        synthetic_quantile=synthetic_q,
        aligned_iforest=auxiliary,
    )
    summary = {
        "block": block_name,
        "method": "personal q95 boundary-calibrated cross-user mix",
        "boundary_quantile": BOUNDARY_QUANTILE,
        "seeds": SEEDS,
        "classifier_losses": logs,
        "diagnostics": diagnostics,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s2r3_boundary_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
                "diagnostics": diagnostics,
                "k10": metrics[
                    metrics["k"] == 10
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices=sorted(BLOCKS),
        default="r42_original",
    )
    args = parser.parse_args()
    run_block(args.block)
