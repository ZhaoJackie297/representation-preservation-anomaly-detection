from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
SEEDS = [20261111, 20261112, 20261113]
PREDICTOR_SEED = 20261111
LOOKBACK_ACTIVE_DAYS = 7
EPOCHS = 8
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIMS = 32
PRIMARY_K = 10

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
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
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r42_original_channel_top2_quantile",
        ),
        "manifold_prefix": "r42",
        "feature_file": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        ),
        "old_masked": OUT / "e17_r42_M0_masked_quantile.npy",
        "external": False,
    },
    "r42_retrained": {
        "family": "r42",
        "plain": OUT / "e19_r42_retrained_plain.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r42_retrained_channel_top2_quantile",
        ),
        "manifold_prefix": "r42_retrained",
        "feature_file": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        ),
        "old_masked": (
            OUT / "e19_r42_retrained_masked_quantile.npy"
        ),
        "external": False,
    },
    "r52_external": {
        "family": "r52",
        "plain": OUT / "e17_r52_M0_plain.npy",
        "channel": (
            OUT / "channel_evidence_arrays.npz",
            "r52_external_channel_top2_quantile",
        ),
        "manifold_prefix": "r52",
        "feature_file": (
            ROOT / "processed_r52" / "r5.2_user_day_features.csv"
        ),
        "sequence": (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        ),
        "old_masked": OUT / "e17_r52_M0_masked_quantile.npy",
        "external": True,
    },
}


class HistoryPredictor(nn.Module):
    def __init__(self, input_dims: int, target_calendar_dims: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dims,
            hidden_size=HIDDEN_DIMS,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIMS + target_calendar_dims, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

    def forward(
        self,
        history: torch.Tensor,
        target_calendar: torch.Tensor,
    ) -> torch.Tensor:
        _, hidden = self.gru(history)
        combined = torch.cat(
            [hidden[-1], target_calendar], dim=1
        )
        return self.head(combined)


def build_index_tables(
    active_mask: np.ndarray,
    *,
    target_end: int | None = None,
) -> dict[str, np.ndarray]:
    users = []
    targets = []
    histories = []
    gaps = []
    time_steps = active_mask.shape[1]

    for user_index in range(active_mask.shape[0]):
        active = np.flatnonzero(active_mask[user_index])
        for active_position in range(
            LOOKBACK_ACTIVE_DAYS, len(active)
        ):
            target_time = int(active[active_position])
            if target_end is not None and target_time > target_end:
                continue
            history = active[
                active_position - LOOKBACK_ACTIVE_DAYS :
                active_position
            ]
            previous_time = (
                int(active[active_position - LOOKBACK_ACTIVE_DAYS - 1])
                if active_position - LOOKBACK_ACTIVE_DAYS - 1 >= 0
                else int(history[0])
            )
            with_previous = np.concatenate(
                [np.asarray([previous_time]), history]
            )
            step_gaps = np.diff(with_previous).astype(np.float32)

            users.append(user_index)
            targets.append(user_index * time_steps + target_time)
            histories.append(
                user_index * time_steps + history.astype(np.int64)
            )
            gaps.append(step_gaps)

    return {
        "user": np.asarray(users, dtype=np.int32),
        "target": np.asarray(targets, dtype=np.int64),
        "history": np.asarray(histories, dtype=np.int64),
        "gap": np.asarray(gaps, dtype=np.float32),
    }


def assemble_inputs(
    normalized_plain: np.ndarray,
    calendar: np.ndarray,
    gap_log_batch: np.ndarray,
    history_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_features = np.concatenate(
        [
            normalized_plain[history_indices],
            calendar[history_indices],
            gap_log_batch[:, :, None],
        ],
        axis=-1,
    ).astype(np.float32)
    target_features = (
        normalized_plain[target_indices]
        .astype(np.float32)
    )
    target_calendar = calendar[target_indices].astype(np.float32)
    return (
        torch.from_numpy(history_features),
        torch.from_numpy(target_features),
        torch.from_numpy(target_calendar),
    )


def train_predictor(
    normalized_plain: np.ndarray,
    calendar: np.ndarray,
    gap_log: np.ndarray,
    train_table: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[HistoryPredictor, float]:
    torch.manual_seed(PREDICTOR_SEED)
    model = HistoryPredictor(
        input_dims=normalized_plain.shape[1] + 3 + 1,
        target_calendar_dims=3,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    count = len(train_table["target"])
    rng = np.random.default_rng(PREDICTOR_SEED)
    final_loss = float("nan")

    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(count)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, count, BATCH_SIZE):
            batch_index = order[start : start + BATCH_SIZE]
            history, target, target_calendar = assemble_inputs(
                normalized_plain,
                calendar,
                gap_log[batch_index],
                train_table["history"][batch_index],
                train_table["target"][batch_index],
            )
            history = history.to(device)
            target = F.normalize(
                target.to(device), dim=-1
            )
            target_calendar = target_calendar.to(device)

            optimizer.zero_grad()
            prediction = F.normalize(
                model(history, target_calendar), dim=-1
            )
            loss = (
                1.0
                - F.cosine_similarity(
                    prediction, target, dim=-1
                )
            ).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1

        final_loss = epoch_loss / max(batches, 1)
        print(
            f"Predictor epoch {epoch + 1}/{EPOCHS}: "
            f"cosine residual {final_loss:.6f}",
            flush=True,
        )
    return model, final_loss


def predict_residual(
    model: HistoryPredictor,
    normalized_plain: np.ndarray,
    calendar: np.ndarray,
    gap_log: np.ndarray,
    inference_table: dict[str, np.ndarray],
    active_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    raw = np.full(active_mask.shape, np.nan, dtype=np.float32)
    count = len(inference_table["target"])
    with torch.no_grad():
        for start in range(0, count, 2048):
            end = min(start + 2048, count)
            history, target, target_calendar = assemble_inputs(
                normalized_plain,
                calendar,
                gap_log[start:end],
                inference_table["history"][start:end],
                inference_table["target"][start:end],
            )
            target = F.normalize(
                target.to(device), dim=-1
            )
            prediction = F.normalize(
                model(
                    history.to(device),
                    target_calendar.to(device),
                ),
                dim=-1,
            )
            residual = (
                1.0
                - F.cosine_similarity(
                    prediction, target, dim=-1
                )
            ).cpu().numpy()
            flat_index = inference_table["target"][start:end]
            raw.reshape(-1)[flat_index] = residual
    return raw


def top_overlap(
    first: np.ndarray,
    second: np.ndarray,
    active_mask: np.ndarray,
    start_index: int,
    end_index: int,
) -> float:
    overlaps = []
    for time_index in range(start_index, end_index + 1):
        active = np.flatnonzero(active_mask[:, time_index])
        if len(active) == 0:
            continue
        first_order = active[
            np.lexsort(
                (
                    active,
                    -np.nan_to_num(first[active, time_index]),
                )
            )
        ][:PRIMARY_K]
        second_order = active[
            np.lexsort(
                (
                    active,
                    -np.nan_to_num(second[active, time_index]),
                )
            )
        ][:PRIMARY_K]
        overlaps.append(
            len(set(first_order.tolist()) & set(second_order.tolist()))
            / max(min(PRIMARY_K, len(active)), 1)
        )
    return float(np.mean(overlaps)) if overlaps else 0.0


def run_block(block_name: str) -> None:
    config = BLOCKS[block_name]
    data = (
        load_r52()
        if config["family"] == "r52"
        else load_r42()
    )
    dataset = np.load(config["sequence"], allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    train_end = int(dataset["train_end_index"])
    plain = np.load(
        config["plain"], mmap_mode="r"
    ).astype(np.float32)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)

    train_active = active_mask[:, : train_end + 1]
    train_vectors = plain[:, : train_end + 1][train_active]
    center = train_vectors.mean(axis=0)
    scale = train_vectors.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized_plain = (
        (plain - center) / scale
    ).astype(np.float32)

    calendar = np.nan_to_num(
        values[:, :, 24:27], nan=0.0, posinf=0.0, neginf=0.0
    )
    flat_calendar = calendar.reshape(-1, 3)

    train_table = build_index_tables(
        active_mask, target_end=train_end
    )
    inference_table = build_index_tables(active_mask)
    gap_train = np.log1p(train_table["gap"])
    gap_mean = float(gap_train.mean())
    gap_scale = float(gap_train.std())
    if gap_scale < 1e-8:
        gap_scale = 1.0
    train_gap_log = (
        gap_train - gap_mean
    ) / gap_scale
    inference_gap_log = (
        np.log1p(inference_table["gap"]) - gap_mean
    ) / gap_scale

    print(
        f"{block_name}: {len(train_table['target'])} train targets, "
        f"{len(inference_table['target'])} inference targets",
        flush=True,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, train_loss = train_predictor(
        normalized_plain.reshape(-1, plain.shape[-1]),
        flat_calendar,
        train_gap_log,
        train_table,
        device,
    )
    raw = predict_residual(
        model,
        normalized_plain.reshape(-1, plain.shape[-1]),
        flat_calendar,
        inference_gap_log,
        inference_table,
        active_mask,
        device,
    )

    calibrated = per_user_quantile_scores(
        raw, active_mask, train_end
    )
    # The first few active days have no seven-day history. They are not
    # compared by the predictive score and are treated as no predictive
    # evidence rather than being assigned an artificial increase.
    calibrated[active_mask & ~np.isfinite(calibrated)] = 0.0

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
    channel_store = np.load(
        config["channel"][0], allow_pickle=False
    )
    channel_q = channel_store[
        config["channel"][1]
    ].astype(np.float32)

    triple_q = (
        calibrated + manifold_q + channel_q
    ) / 3.0
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        triple_q,
        data["active_mask"],
    )
    signal = daily_percentile(
        dynamic, data["active_mask"]
    ).astype(np.float32)

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
            "S2R6_history_predictive",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s2r6_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    old_masked = np.load(
        config["old_masked"]
    ).astype(np.float32)
    valid = active_mask & np.isfinite(calibrated) & np.isfinite(
        old_masked
    )
    correlation = float(
        np.corrcoef(
            calibrated[valid], old_masked[valid]
        )[0, 1]
    )
    ranges = (
        {"external_test": (data["test_start"], len(data["dates"]) - 1)}
        if config["external"]
        else {
            "validation": (data["train_end"] + 1, data["val_end"]),
            "test": (data["val_end"] + 1, len(data["dates"]) - 1),
        }
    )
    overlap = {
        split: top_overlap(
            calibrated,
            old_masked,
            active_mask,
            start,
            end,
        )
        for split, (start, end) in ranges.items()
    }

    components = {
        "predictive_raw": raw,
        "predictive_quantile": calibrated,
        "predictive_signal": signal,
        "predictive_dynamic": dynamic,
        "predictive_triple_quantile": triple_q,
        "predictive_aligned_iforest": auxiliary,
        "train_target_user": train_table["user"],
        "train_target_index": train_table["target"],
    }
    np.savez_compressed(
        OUT / f"s2r6_{block_name}_components.npz",
        **components,
    )
    summary = {
        "block": block_name,
        "model": "frozen TS2Vec + shared one-layer GRU predictor",
        "predictor_seed": PREDICTOR_SEED,
        "lookback_active_days": LOOKBACK_ACTIVE_DAYS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "hidden_dims": HIDDEN_DIMS,
        "train_targets": len(train_table["target"]),
        "inference_targets": len(inference_table["target"]),
        "train_cosine_residual": train_loss,
        "fusion": "equal_weight_three_scores_replace_masked_last",
        "correlation_with_old_masked": correlation,
        "top10_overlap_with_old_masked": overlap,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s2r6_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
                "train_cosine_residual": train_loss,
                "correlation_with_old_masked": correlation,
                "top10_overlap_with_old_masked": overlap,
                "k10": metrics[
                    metrics["k"] == PRIMARY_K
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
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
