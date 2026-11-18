from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e19_models"
SEEDS = [20261111, 20261112, 20261113]
MIX_ALPHA = 0.5
EPOCHS = 3
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from e19_user_manifold import UserMetricProjector, project_with_model  # noqa: E402
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
from quota_relationship_sim import load_r42, load_r52  # noqa: E402
from s2r3_latent_synthetic import (  # noqa: E402
    SyntheticAnomalyClassifier,
    build_signal,
    classifier_scores as score_all,
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


def nearest_peer_donors(
    projected: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_mask = (
        active_mask[:, : train_end + 1]
        & np.isfinite(
            projected[:, : train_end + 1]
        ).all(axis=-1)
    )
    users, dates = np.nonzero(train_mask)
    normal = projected[users, dates].astype(np.float32)
    donors = np.empty_like(normal)
    distances = np.empty(len(normal), dtype=np.float32)
    for date in np.unique(dates):
        rows = np.flatnonzero(dates == date)
        if len(rows) < 2:
            donors[rows] = normal[rows]
            distances[rows] = 0.0
            continue
        values = normal[rows]
        distances_matrix = np.square(
            values[:, None, :] - values[None, :, :]
        ).sum(axis=-1)
        np.fill_diagonal(distances_matrix, np.inf)
        nearest = distances_matrix.argmin(axis=1)
        donors[rows] = values[nearest]
        distances[rows] = np.sqrt(
            distances_matrix[np.arange(len(rows)), nearest]
        )
    return normal, donors, {
        "rows": int(len(normal)),
        "mean_nearest_distance": float(distances.mean()),
        "median_nearest_distance": float(np.median(distances)),
    }


def train_hard_classifier(
    normal: np.ndarray,
    synthetic: np.ndarray,
    seed: int,
) -> tuple[SyntheticAnomalyClassifier, list[float]]:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    normal_tensor = torch.from_numpy(normal).to(device)
    synthetic_tensor = torch.from_numpy(synthetic).to(device)
    classifier = SyntheticAnomalyClassifier(
        normal.shape[-1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    losses = []
    for _ in range(EPOCHS):
        permutation = torch.randperm(
            normal_tensor.shape[0], device=device
        )
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(permutation), BATCH_SIZE):
            batch_indices = permutation[start : start + BATCH_SIZE]
            normal_batch = normal_tensor[batch_indices]
            synthetic_batch = synthetic_tensor[batch_indices]
            logits = torch.cat(
                [
                    classifier(normal_batch),
                    classifier(synthetic_batch),
                ]
            )
            targets = torch.cat(
                [
                    torch.zeros(
                        len(normal_batch), device=device
                    ),
                    torch.ones(
                        len(synthetic_batch), device=device
                    ),
                ]
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, targets
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                classifier.parameters(), max_norm=1.0
            )
            optimizer.step()
            epoch_loss += float(loss.detach())
            steps += 1
        losses.append(epoch_loss / max(steps, 1))
    return classifier.eval(), losses


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
    seed_logs = {}
    diagnostics = {}
    for seed in SEEDS:
        print(f"{block_name}: hard-peer seed {seed}", flush=True)
        projection = load_projection(
            config, seed, plain.shape[-1], device
        )
        projected = project_with_model(projection, plain)
        normal, donors, diagnostic = nearest_peer_donors(
            projected, active_mask, train_end
        )
        synthetic = (
            MIX_ALPHA * normal
            + (1.0 - MIX_ALPHA) * donors
        )
        classifier, losses = train_hard_classifier(
            normal, synthetic, seed
        )
        raw = score_all(classifier, projected)
        quantile = per_user_quantile_scores(
            raw, active_mask, train_end
        )
        quantiles.append(quantile)
        seed_logs[str(seed)] = losses
        diagnostics[str(seed)] = diagnostic
        np.save(
            OUT
            / f"s2r3_hard_{block_name}_raw_{seed}.npy",
            raw,
        )
        del projection, projected, classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ensemble_q = np.mean(
        np.stack(quantiles, axis=0),
        axis=0,
    ).astype(np.float32)
    signal, dynamic = build_signal(
        data,
        masked_q,
        manifold_q,
        channel_q,
        ensemble_q,
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
            "S2R3_hard_peer",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s2r3_hard_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / f"s2r3_hard_{block_name}_components.npz",
        signal=signal,
        synthetic_quantile=ensemble_q,
        aligned_iforest=auxiliary,
    )
    summary = {
        "block": block_name,
        "method": "same-date nearest cross-user hard mix",
        "mix_alpha": MIX_ALPHA,
        "seeds": SEEDS,
        "classifier_losses": seed_logs,
        "peer_diagnostics": diagnostics,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s2r3_hard_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
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
