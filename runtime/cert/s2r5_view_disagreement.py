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
SEEDS = [20261111, 20261112, 20261113]
BLOCK_DAYS = 3
PADDING_DAYS = 30
PRIMARY_K = 10

# These groups are fixed before reading any application result. The activity
# indicator and calendar variables stay visible in every view so that the
# encoder still knows whether the day exists and what weekday it is.
VIEW_GROUPS = {
    "access_device": list(range(1, 10)),
    "file_email": list(range(10, 21)),
    "web": list(range(21, 25)),
}

sys.path.insert(0, str(ROOT))
from e17_multigranular_mask import (  # noqa: E402
    encode_causal_student,
    load_base_student,
)
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
        "model": ROOT / "processed_r42" / "r4.2_ts2vec_model.pth",
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
        "model": (
            ROOT
            / "processed_r42"
            / "innovation_sim"
            / "r4.2_ts2vec_quota_retrain.pth"
        ),
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
        "model": ROOT / "processed_r52" / "r5.2_ts2vec_model.pth",
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


def encode_causal_group_occlusion(
    student: torch.nn.Module,
    values: np.ndarray,
    feature_indices: list[int],
    *,
    device: torch.device,
) -> np.ndarray:
    """Encode each day with one semantic feature group removed recently."""
    student.eval()
    values_tensor = torch.from_numpy(values).float().to(device)
    users, time_steps, feature_dims = values_tensor.shape
    user_batch_size = 512
    user_outputs = []

    with torch.no_grad():
        for user_start in range(0, users, user_batch_size):
            user_end = min(user_start + user_batch_size, users)
            user_values = values_tensor[user_start:user_end]
            representations = []
            for time_index in range(time_steps):
                left = max(0, time_index - PADDING_DAYS)
                right = time_index + 1
                span = right - left
                window = torch.full(
                    (
                        user_end - user_start,
                        PADDING_DAYS + 1,
                        feature_dims,
                    ),
                    torch.nan,
                    device=device,
                    dtype=torch.float32,
                )
                window[:, -span:, :] = user_values[
                    :, left:right, :
                ]

                # Remove only the most recent three observations. Historical
                # values remain available as context, matching the causal
                # scoring setup used by the frozen TS2Vec pipeline.
                window[:, -BLOCK_DAYS:, feature_indices] = 0.0
                output = student(window, mask="all_true")
                representations.append(output[:, -1:, :])
            user_outputs.append(
                torch.cat(representations, dim=1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return np.concatenate(user_outputs, axis=0)


def cosine_disagreement(
    reference: np.ndarray,
    view: np.ndarray,
) -> np.ndarray:
    dot = np.sum(reference * view, axis=-1)
    reference_norm = np.linalg.norm(reference, axis=-1)
    view_norm = np.linalg.norm(view, axis=-1)
    denominator = np.maximum(
        reference_norm * view_norm, 1e-8
    )
    cosine = np.clip(dot / denominator, -1.0, 1.0)
    return (0.5 * (1.0 - cosine)).astype(np.float32)


def daily_top_overlap(
    first: np.ndarray,
    second: np.ndarray,
    active_mask: np.ndarray,
    start_index: int,
    end_index: int,
    k: int = PRIMARY_K,
) -> float:
    overlaps = []
    for time_index in range(start_index, end_index + 1):
        active = np.flatnonzero(active_mask[:, time_index])
        if len(active) == 0:
            continue
        first_order = active[
            np.lexsort(
                (active, -np.nan_to_num(first[active, time_index]))
            )
        ][:k]
        second_order = active[
            np.lexsort(
                (active, -np.nan_to_num(second[active, time_index]))
            )
        ][:k]
        overlaps.append(
            len(set(first_order.tolist()) & set(second_order.tolist()))
            / max(min(k, len(active)), 1)
        )
    return float(np.mean(overlaps)) if overlaps else 0.0


def build_disagreement_signal(
    student: torch.nn.Module,
    values: np.ndarray,
    data: dict,
    train_end: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    print("Encoding full causal representation", flush=True)
    full = encode_causal_student(
        student,
        values,
        mask=None,
        device=device,
        padding=PADDING_DAYS,
    )

    group_disagreements = []
    for group_name, indices in VIEW_GROUPS.items():
        print(f"Encoding leave-out view: {group_name}", flush=True)
        view = encode_causal_group_occlusion(
            student,
            values,
            indices,
            device=device,
        )
        group_disagreements.append(
            cosine_disagreement(full, view)
        )
        del view

    stacked = np.stack(group_disagreements, axis=0)
    raw = stacked.mean(axis=0).astype(np.float32)
    calibrated = per_user_quantile_scores(
        raw, data["active_mask"], train_end
    )
    return calibrated, {
        "full_representation": full,
        "access_device_disagreement": stacked[0],
        "file_email_disagreement": stacked[1],
        "web_disagreement": stacked[2],
        "view_disagreement_raw": raw,
        "view_disagreement_quantile": calibrated,
    }


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
    channel_q = channel_store[config["channel"][1]].astype(
        np.float32
    )

    student, device = load_base_student(
        config["model"], values.shape[-1]
    )
    view_q, arrays = build_disagreement_signal(
        student, values, data, train_end, device
    )

    # The new view-disagreement signal replaces the single masked-last signal.
    # User-manifold and channel-evidence branches remain unchanged.
    triple_q = (view_q + manifold_q + channel_q) / 3.0
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
            "S2R5_view_disagreement",
            external=bool(config["external"]),
        )
    )
    metrics.to_csv(
        OUT / f"s2r5_{block_name}_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    old_masked = np.load(config["old_masked"]).astype(np.float32)
    valid = data["active_mask"] & np.isfinite(view_q) & np.isfinite(
        old_masked
    )
    correlation = float(
        np.corrcoef(view_q[valid], old_masked[valid])[0, 1]
    )
    split_ranges = (
        {"external_test": (data["test_start"], len(data["dates"]) - 1)}
        if config["external"]
        else {
            "validation": (data["train_end"] + 1, data["val_end"]),
            "test": (data["val_end"] + 1, len(data["dates"]) - 1),
        }
    )
    overlap = {
        split: daily_top_overlap(
            view_q,
            old_masked,
            data["active_mask"],
            start,
            end,
        )
        for split, (start, end) in split_ranges.items()
    }

    arrays.update(
        {
            "view_disagreement_signal": signal,
            "view_disagreement_dynamic": dynamic,
            "view_disagreement_triple_quantile": triple_q,
            "view_disagreement_aligned_iforest": auxiliary,
        }
    )
    np.savez_compressed(
        OUT / f"s2r5_{block_name}_components.npz",
        **arrays,
    )
    summary = {
        "block": block_name,
        "primary_k": PRIMARY_K,
        "block_days": BLOCK_DAYS,
        "view_groups": VIEW_GROUPS,
        "fusion": "equal_weight_three_scores_replace_masked_last",
        "correlation_with_old_masked": correlation,
        "top10_overlap_with_old_masked": overlap,
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / f"s2r5_{block_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "block": block_name,
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
