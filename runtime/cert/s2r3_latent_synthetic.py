from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e19_models"
SEEDS = [20261111, 20261112, 20261113]
BATCH_SIZE = 512
EPOCHS = 3
LEARNING_RATE = 1e-3
MIX_ALPHA = 0.5

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from e19_user_manifold import (  # noqa: E402
    UserMetricProjector,
    project_with_model,
)
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42,
    load_r52,
)
from risk_topk_r42 import compute_risk  # noqa: E402
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


class SyntheticAnomalyClassifier(nn.Module):
    def __init__(self, input_dims: int = 24) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def load_metric_model(
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


def train_synthetic_classifier(
    projected: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    seed: int,
) -> tuple[SyntheticAnomalyClassifier, list[float], dict]:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    train_mask = (
        active_mask[:, : train_end + 1]
        & np.isfinite(
            projected[:, : train_end + 1]
        ).all(axis=-1)
    )
    user_ids, time_ids = np.nonzero(train_mask)
    training = torch.from_numpy(
        projected[user_ids, time_ids].astype(np.float32)
    ).to(device)
    user_tensor = torch.from_numpy(
        user_ids.astype(np.int64)
    ).to(device)
    classifier = SyntheticAnomalyClassifier(
        training.shape[-1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    loss_log = []
    for epoch in range(EPOCHS):
        permutation = torch.randperm(
            training.shape[0], device=device
        )
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(permutation), BATCH_SIZE):
            batch_indices = permutation[start : start + BATCH_SIZE]
            normal = training[batch_indices]
            donor_indices = torch.randint(
                0,
                training.shape[0],
                (len(batch_indices),),
                device=device,
            )
            same_user = (
                user_tensor[donor_indices]
                == user_tensor[batch_indices]
            )
            if bool(same_user.any()):
                replacements = torch.randint(
                    0,
                    training.shape[0],
                    (int(same_user.sum()),),
                    device=device,
                )
                donor_indices[same_user] = replacements
            donor = training[donor_indices]
            synthetic = (
                MIX_ALPHA * normal
                + (1.0 - MIX_ALPHA) * donor
            )
            logits = torch.cat(
                [
                    classifier(normal),
                    classifier(synthetic),
                ],
                dim=0,
            )
            targets = torch.cat(
                [
                    torch.zeros(
                        len(normal), device=device
                    ),
                    torch.ones(
                        len(synthetic), device=device
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
        loss_log.append(epoch_loss / max(steps, 1))
        print(
            f"  synthetic classifier epoch {epoch + 1}: "
            f"{loss_log[-1]:.6f}",
            flush=True,
        )
    diagnostics = {
        "training_rows": int(training.shape[0]),
        "same_user_replacements": 0,
    }
    return classifier.eval(), loss_log, diagnostics


def classifier_scores(
    classifier: SyntheticAnomalyClassifier,
    projected: np.ndarray,
) -> np.ndarray:
    device = next(classifier.parameters()).device
    users, time_steps, dimensions = projected.shape
    output = np.full(
        (users, time_steps), np.nan, dtype=np.float32
    )
    with torch.no_grad():
        for start in range(0, users, 128):
            end = min(start + 128, users)
            block = torch.from_numpy(
                np.asarray(projected[start:end])
            ).to(device)
            logits = classifier(
                block.reshape(-1, dimensions)
            ).reshape(end - start, time_steps)
            output[start:end] = (
                torch.sigmoid(logits).cpu().numpy()
            )
    return output


def build_signal(
    data: dict,
    masked_q: np.ndarray,
    manifold_q: np.ndarray,
    channel_q: np.ndarray,
    synthetic_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = (
        masked_q + manifold_q + channel_q + synthetic_q
    ) / 4.0
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        scores,
        data["active_mask"],
    )
    return (
        daily_percentile(
            dynamic, data["active_mask"]
        ).astype(np.float32),
        dynamic,
    )


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
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    synthetic_quantiles = []
    manifold_quantiles = []
    loss_logs = {}
    for seed in SEEDS:
        print(
            f"{block_name}: synthetic seed {seed}",
            flush=True,
        )
        projection = load_metric_model(
            config, seed, plain.shape[-1], device
        )
        projected = project_with_model(projection, plain)
        classifier, losses, diagnostics = train_synthetic_classifier(
            projected, active_mask, train_end, seed
        )
        raw_score = classifier_scores(classifier, projected)
        quantile = per_user_quantile_scores(
            raw_score, active_mask, train_end
        )
        synthetic_quantiles.append(quantile)
        manifold_quantiles.append(
            np.load(
                OUT
                / (
                    f"e19_{config['manifold_prefix']}_learned_manifold_"
                    f"quantile_{seed}.npy"
                )
            )
        )
        loss_logs[str(seed)] = {
            "losses": losses,
            "diagnostics": diagnostics,
        }
        np.save(
            OUT
            / f"s2r3_{block_name}_synthetic_raw_{seed}.npy",
            raw_score,
        )
        torch.save(
            classifier.state_dict(),
            OUT / f"s2r3_{block_name}_classifier_{seed}.pth",
        )
        del projected, projection, classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    synthetic_ensemble_q = np.mean(
        np.stack(synthetic_quantiles, axis=0),
        axis=0,
    ).astype(np.float32)
    manifold_ensemble_q = np.mean(
        np.stack(manifold_quantiles, axis=0),
        axis=0,
    ).astype(np.float32)
    signal, dynamic = build_signal(
        data,
        masked_q,
        manifold_ensemble_q,
        channel_q,
        synthetic_ensemble_q,
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
            "S2R3_synthetic_augment",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s2r3_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / f"s2r3_{block_name}_components.npz",
        synthetic_signal=signal,
        synthetic_quantile=synthetic_ensemble_q,
        manifold_quantile=manifold_ensemble_q,
        aligned_iforest=auxiliary,
    )
    summary = {
        "block": block_name,
        "method": "latent cross-user synthetic anomaly classifier",
        "mix_alpha": MIX_ALPHA,
        "seeds": SEEDS,
        "classifier_training": loss_logs,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s2r3_{block_name}_summary.json").write_text(
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices=sorted(BLOCKS),
        default="r42_original",
    )
    args = parser.parse_args()
    run_block(args.block)


if __name__ == "__main__":
    main()
