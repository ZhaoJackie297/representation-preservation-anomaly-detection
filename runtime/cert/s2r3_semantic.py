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
WINDOW_DAYS = 31
WINDOW_STRIDE = 10
BLOCK_DAYS = 5
EPOCHS = 3
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

CHANNELS = {
    "logon": list(range(1, 8)),
    "device": [8, 9],
    "file": list(range(10, 14)),
    "email": list(range(14, 20)),
    "http": list(range(20, 24)),
}

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from e19_user_manifold import UserMetricProjector, project_with_model  # noqa: E402
from iforest_view_ensemble_sim import fit_view_scores  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    daily_percentile,
    load_r42,
    load_r52,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from ts2vec_dual_score_experiment import load_model  # noqa: E402
from v2_m_pipeline import add_utilities, evaluate_block  # noqa: E402


BLOCKS = {
    "r42_original": {
        "family": "r42",
        "model": ROOT / "processed_r42" / "r4.2_ts2vec_model.pth",
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
        "model": (
            ROOT
            / "processed_r42"
            / "innovation_sim"
            / "r4.2_ts2vec_quota_retrain.pth"
        ),
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
        "model": ROOT / "processed_r52" / "r5.2_ts2vec_model.pth",
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


class SyntheticClassifier(nn.Module):
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


def training_windows(
    values: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = []
    users = []
    ends = []
    for user in range(values.shape[0]):
        for end in range(
            WINDOW_DAYS - 1,
            train_end + 1,
            WINDOW_STRIDE,
        ):
            windows.append(
                values[
                    user,
                    end - WINDOW_DAYS + 1 : end + 1,
                ]
            )
            users.append(user)
            ends.append(end)
    return (
        np.stack(windows).astype(np.float32),
        np.asarray(users, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
    )


def temporal_shuffle(windows: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = windows.copy()
    history_days = WINDOW_DAYS - 1
    block_count = history_days // BLOCK_DAYS
    for index in range(len(output)):
        blocks = [
            output[
                index,
                start * BLOCK_DAYS : (start + 1) * BLOCK_DAYS,
            ].copy()
            for start in range(block_count)
        ]
        order = rng.permutation(block_count)
        shuffled = np.concatenate(
            [blocks[position] for position in order],
            axis=0,
        )
        output[index, : history_days] = shuffled
    return output


def channel_relation_scramble(
    windows: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = windows.copy()
    channel_items = list(CHANNELS.items())
    for index in range(len(output)):
        channel_name, indices = channel_items[
            int(rng.integers(0, len(channel_items)))
        ]
        del channel_name
        order = rng.permutation(WINDOW_DAYS - 1)
        channel_values = output[
            index, : WINDOW_DAYS - 1
        ][:, indices].copy()
        output[
            index, : WINDOW_DAYS - 1
        ][:, indices] = channel_values[order]
    return output


def encode_last(model, windows: np.ndarray) -> np.ndarray:
    representation = np.asarray(
        model.encode(
            windows,
            causal=True,
            sliding_length=1,
            sliding_padding=0,
            batch_size=128,
        ),
        dtype=np.float32,
    )
    return representation[:, -1]


def train_classifier(
    normal: np.ndarray,
    synthetic: np.ndarray,
    seed: int,
) -> tuple[SyntheticClassifier, list[float]]:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    normal_tensor = torch.from_numpy(normal).to(device)
    synthetic_tensor = torch.from_numpy(synthetic).to(device)
    classifier = SyntheticClassifier(normal.shape[-1]).to(device)
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


def score_all(
    classifier: SyntheticClassifier,
    projected: np.ndarray,
) -> np.ndarray:
    device = next(classifier.parameters()).device
    users, time_steps, dimensions = projected.shape
    scores = np.empty((users, time_steps), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, users, 128):
            end = min(start + 128, users)
            block = torch.from_numpy(
                np.asarray(projected[start:end])
            ).to(device)
            logits = classifier(
                block.reshape(-1, dimensions)
            ).reshape(end - start, time_steps)
            scores[start:end] = torch.sigmoid(logits).cpu().numpy()
    return scores


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
    values = dataset["values"].astype(np.float32)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    plain = np.load(config["plain"], mmap_mode="r")
    masked_q = np.load(config["masked"])
    channel_store = np.load(config["channel"][0], allow_pickle=False)
    channel_q = channel_store[config["channel"][1]].astype(np.float32)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    windows, _, _ = training_windows(values, train_end)
    print(
        f"{block_name}: encoding {len(windows)} normal windows",
        flush=True,
    )
    encoder = load_model(config["model"], values.shape[-1])
    normal_repr = encode_last(encoder, windows)
    temporal_repr = encode_last(
        encoder, temporal_shuffle(windows, 20261111)
    )
    print("  encoding channel-relation corruption", flush=True)
    relation_repr = encode_last(
        encoder,
        channel_relation_scramble(windows, 20261112),
    )
    del encoder, windows
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    methods = {}
    logs = {}
    for method, synthetic_repr in (
        ("temporal", temporal_repr),
        ("relation", relation_repr),
    ):
        quantiles = []
        method_logs = {}
        for seed in SEEDS:
            projection = UserMetricProjector(
                plain.shape[-1], 24
            ).to(device)
            projection.load_state_dict(
                torch.load(
                    MODEL_DIR
                    / (
                        f"{config['metric_prefix']}_user_metric_"
                        f"{seed}.pth"
                    ),
                    map_location=device,
                    weights_only=True,
                )
            )
            projection.eval()
            with torch.no_grad():
                normal_projected = projection(
                    torch.from_numpy(normal_repr).to(device)
                ).cpu().numpy()
                synthetic_projected = projection(
                    torch.from_numpy(synthetic_repr).to(device)
                ).cpu().numpy()
            classifier, losses = train_classifier(
                normal_projected,
                synthetic_projected,
                seed,
            )
            projected_all = project_with_model(projection, plain)
            raw = score_all(classifier, projected_all)
            quantile = per_user_quantile_scores(
                raw, active_mask, train_end
            )
            quantiles.append(quantile)
            method_logs[str(seed)] = losses
            del projection, classifier, projected_all
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        methods[method] = np.mean(
            np.stack(quantiles, axis=0),
            axis=0,
        ).astype(np.float32)
        logs[method] = method_logs

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
    metrics_frames = []
    signals = {}
    for method, synthetic_q in methods.items():
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
        metrics_frames.append(
            evaluate_block(
                block_name,
                data,
                signal,
                auxiliary,
                f"S2R3_semantic_{method}",
                external=bool(config["external"]),
            )
        )
        signals[f"{method}_signal"] = signal
        signals[f"{method}_quantile"] = synthetic_q
        signals[f"{method}_aligned_iforest"] = auxiliary
    metrics = add_utilities(
        pd.concat(metrics_frames, ignore_index=True)
    )
    metrics.to_csv(
        OUT / f"s2r3_semantic_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / f"s2r3_semantic_{block_name}_components.npz",
        **signals,
    )
    summary = {
        "block": block_name,
        "training_windows": int(len(normal_repr)),
        "methods": ["temporal", "relation"],
        "classifier_losses": logs,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s2r3_semantic_{block_name}_summary.json").write_text(
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
