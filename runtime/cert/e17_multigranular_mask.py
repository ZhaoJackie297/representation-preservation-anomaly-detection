from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e17_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 30
MASK_DAYS = 3
WINDOW_STRIDE = 5
EPOCHS = 2
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
SEED = 20261017
K = 10
Q_LIMIT = 4

CHANNELS = {
    "logon": list(range(1, 8)),
    "device": [8, 9],
    "file": list(range(10, 14)),
    "email": list(range(14, 20)),
    "http": list(range(20, 24)),
}

sys.path.insert(0, str(ROOT))
from deep_compare_per_user_r42 import per_user_quantile_scores  # noqa: E402
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    daily_percentile,
    evaluate_alert_system,
    load_r42,
    load_r52,
)
from risk_topk_r42 import compute_risk  # noqa: E402
from train_r42_ts2vec import TS2Vec  # noqa: E402
from ts2vec_dual_score_experiment import latent_deviation  # noqa: E402


ENCODE_ARGS = {
    "causal": True,
    "sliding_length": 1,
    "sliding_padding": 30,
    "batch_size": 32,
}


def load_base_student(
    model_file: Path,
    input_dims: int,
) -> tuple[torch.nn.Module, torch.device]:
    device = 0 if torch.cuda.is_available() else "cpu"
    wrapper = TS2Vec(
        input_dims=input_dims,
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        batch_size=16,
    )
    map_device = (
        f"cuda:{device}" if isinstance(device, int) else "cpu"
    )
    wrapper.net.load_state_dict(
        torch.load(
            model_file,
            map_location=map_device,
            weights_only=True,
        )
    )
    torch_device = torch.device(
        f"cuda:{device}" if isinstance(device, int) else "cpu"
    )
    return copy.deepcopy(wrapper.net.module), torch_device


def build_training_windows(
    values: np.ndarray,
    train_end: int,
) -> torch.Tensor:
    windows = []
    for user_index in range(values.shape[0]):
        for end_index in range(
            WINDOW_DAYS - 1,
            train_end + 1,
            WINDOW_STRIDE,
        ):
            window = values[
                user_index,
                end_index - WINDOW_DAYS + 1 : end_index + 1,
            ]
            windows.append(window)
    return torch.from_numpy(
        np.stack(windows).astype(np.float32)
    )


def build_mask_views(
    batch: torch.Tensor,
    include_time: bool,
    include_channel: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    batch_size, length, _ = batch.shape
    device = batch.device
    views = []

    if include_time:
        x_time = batch.clone()
        block_mask = torch.zeros(
            (batch_size, length), dtype=torch.bool, device=device
        )
        block_mask[:, -MASK_DAYS:] = True
        x_time[block_mask] = 0.0
        views.append((x_time, block_mask, "time"))

    channel_mask = torch.zeros(
        (batch_size, length, batch.shape[-1]),
        dtype=torch.bool,
        device=device,
    )
    channel_names = list(CHANNELS)
    selected_channels = torch.randint(
        0, len(channel_names), (batch_size,), device=device
    )
    for sample_index in range(batch_size):
        channel = channel_names[int(selected_channels[sample_index])]
        channel_mask[
            sample_index, -MASK_DAYS:, CHANNELS[channel]
        ] = True

    if include_channel:
        x_channel = batch.clone()
        x_channel[channel_mask] = 0.0
        channel_position_mask = channel_mask.any(dim=-1)
        views.append(
            (x_channel, channel_position_mask, "channel")
        )

    if include_time and include_channel:
        x_joint = batch.clone()
        joint_mask = channel_mask.clone()
        joint_mask[:, -MASK_DAYS:, :] = True
        x_joint[joint_mask] = 0.0
        views.append(
            (x_joint, joint_mask.any(dim=-1), "joint")
        )
    return views


def masked_consistency_loss(
    student,
    teacher_full: torch.Tensor,
    masked_view: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    student_masked = student(masked_view, mask="all_true")
    cosine = F.cosine_similarity(
        student_masked, teacher_full, dim=-1
    )
    selected = cosine[position_mask]
    if selected.numel() == 0:
        return torch.zeros((), device=cosine.device)
    return (1.0 - selected).mean()


def finetune_variant(
    base_student: torch.nn.Module,
    train_windows: torch.Tensor,
    *,
    variant: str,
    device,
) -> tuple[torch.nn.Module, list[float]]:
    student = copy.deepcopy(base_student).to(device)
    teacher = copy.deepcopy(base_student).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student.train()

    include_time = variant in {"M1_time", "M3_joint"}
    include_channel = variant in {"M2_channel", "M3_joint"}

    loader = DataLoader(
        TensorDataset(train_windows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    loss_log = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        iterations = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_full = teacher(batch, mask="all_true")
            student_full = student(batch, mask="all_true")
            full_loss = (
                1.0
                - F.cosine_similarity(
                    student_full, teacher_full, dim=-1
                )
            ).mean()
            loss = full_loss
            for masked_view, mask_positions, _ in build_mask_views(
                batch,
                include_time,
                include_channel,
            ):
                loss = loss + masked_consistency_loss(
                    student,
                    teacher_full,
                    masked_view,
                    mask_positions,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=1.0
            )
            optimizer.step()
            epoch_loss += float(loss.item())
            iterations += 1
        average_loss = epoch_loss / max(iterations, 1)
        loss_log.append(average_loss)
        print(
            f"  {variant} epoch {epoch + 1}: {average_loss:.6f}",
            flush=True,
        )
    student.eval()
    return student, loss_log


def encode_causal_student(
    student: torch.nn.Module,
    values: np.ndarray,
    *,
    mask: str | None,
    device: torch.device,
    padding: int = 30,
) -> np.ndarray:
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
            with torch.no_grad():
                for time_index in range(time_steps):
                    left = max(0, time_index - padding)
                    right = time_index + 1
                    span = right - left
                    window = torch.full(
                        (user_end - user_start, padding + 1, feature_dims),
                        torch.nan,
                        device=device,
                        dtype=torch.float32,
                    )
                    window[:, -span:, :] = user_values[
                        :, left:right, :
                    ]
                    if mask == "mask_last":
                        output = student(window, mask="mask_last")
                    else:
                        output = student(window, mask="all_true")
                    representations.append(output[:, -1:, :])
            user_outputs.append(
                torch.cat(representations, dim=1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return np.concatenate(user_outputs, axis=0)


def build_application_scores(
    student,
    values: np.ndarray,
    data: dict,
    train_end: int,
    device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    plain = encode_causal_student(
        student, values, mask=None, device=device
    )
    masked = encode_causal_student(
        student,
        values,
        mask="mask_last",
        device=device,
    )
    masked_raw = np.abs(masked - plain).sum(axis=-1)
    latent_raw, _ = latent_deviation(
        plain, data["active_mask"], train_end
    )
    masked_q = per_user_quantile_scores(
        masked_raw, data["active_mask"], train_end
    )
    latent_q = per_user_quantile_scores(
        latent_raw, data["active_mask"], train_end
    )

    channel_quantiles = []
    channel_raw = {}
    for channel, indices in CHANNELS.items():
        modified = values.copy()
        modified[:, :, indices] = 0.0
        representation = encode_causal_student(
            student,
            modified,
            mask=None,
            device=device,
        )
        raw = np.abs(representation - plain).sum(axis=-1)
        channel_raw[channel] = raw
        channel_quantiles.append(
            per_user_quantile_scores(
                raw, data["active_mask"], train_end
            )
        )
    stacked = np.stack(channel_quantiles, axis=0)
    ordered = np.sort(stacked, axis=0)
    channel_top2_q = ordered[-2:].mean(axis=0)
    triple_q = (masked_q + latent_q + channel_top2_q) / 3.0
    dynamic = compute_risk(
        data["users"],
        data["dates"],
        triple_q,
        data["active_mask"],
    )
    signal = daily_percentile(
        dynamic, data["active_mask"]
    ).astype(np.float32)
    return signal, {
        "plain": plain,
        "masked_raw": masked_raw,
        "latent_raw": latent_raw,
        "masked_quantile": masked_q,
        "latent_quantile": latent_q,
        "channel_top2_quantile": channel_top2_q,
        "triple_quantile": triple_q,
        "signal": signal,
    }


def evaluate_signal(
    data: dict,
    signal: np.ndarray,
    auxiliary,
    method: str,
    *,
    external: bool,
    use_iforest: bool,
) -> dict:
    if use_iforest:
        alerts = build_unique_quota_alerts(
            signal,
            auxiliary,
            data["active_mask"],
            K,
            Q_LIMIT,
        )
    else:
        alerts = np.zeros(signal.shape, dtype=np.uint8)
        last_alert = np.full(
            signal.shape[0], -10000, dtype=np.int32
        )
        for time_index in range(signal.shape[1]):
            available = data["active_mask"][:, time_index] & (
                time_index - last_alert >= 7
            )
            candidates = np.flatnonzero(available)
            order = candidates[
                np.lexsort(
                    (candidates, -signal[candidates, time_index])
                )
            ]
            selected = order[:K]
            alerts[selected, time_index] = 1
            last_alert[selected] = time_index
    result = evaluate_alert_system(
        data, alerts, method, K, external=external
    )
    splits = ["external_test"] if external else ["validation", "test"]
    metrics = {}
    for split in splits:
        metrics[split] = compact_metric(
            result, None if external else split
        )
    return metrics


def run_dataset(dataset_name: str) -> None:
    if dataset_name == "r42":
        dataset_file = (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        )
        model_file = ROOT / "processed_r42" / "r4.2_ts2vec_model.pth"
        data = load_r42()
        external = False
        auxiliary_all = np.load(
            OUT / "v2_iforest_alignment_arrays.npz",
            allow_pickle=False,
        )
        auxiliary = auxiliary_all[
            "r42_original_v2_aligned_dual_view_iforest"
        ]
    else:
        dataset_file = (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        )
        model_file = ROOT / "processed_r52" / "r5.2_ts2vec_model.pth"
        data = load_r52()
        external = True
        auxiliary_all = np.load(
            OUT / "v2_iforest_alignment_arrays.npz",
            allow_pickle=False,
        )
        auxiliary = auxiliary_all[
            "r52_external_v2_aligned_dual_view_iforest"
        ]

    dataset = np.load(dataset_file, allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    train_end = (
        int(dataset["train_end_index"])
        if "train_end_index" in dataset.files
        else data["test_start"] - 1
    )
    base_student, device = load_base_student(
        model_file, values.shape[-1]
    )
    train_windows = build_training_windows(values, train_end)
    print(
        f"{dataset_name}: {len(train_windows)} training windows",
        flush=True,
    )

    results = {}
    for variant in (
        "M0",
        "M1_time",
        "M2_channel",
        "M3_joint",
    ):
        if variant == "M0":
            student = copy.deepcopy(base_student)
            losses = []
        else:
            student, losses = finetune_variant(
                base_student,
                train_windows,
                variant=variant,
                device=device,
            )
        model_path = MODEL_DIR / f"{dataset_name}_{variant}.pth"
        torch.save(student.state_dict(), model_path)
        signal, arrays = build_application_scores(
            student, values, data, train_end, device
        )
        complete_metrics = evaluate_signal(
            data,
            signal,
            auxiliary,
            f"{variant}_complete",
            external=external,
            use_iforest=False,
        )
        iforest_metrics = evaluate_signal(
            data,
            signal,
            auxiliary,
            f"{variant}_iforest",
            external=external,
            use_iforest=True,
        )
        results[variant] = {
            "losses": losses,
            "model": str(model_path),
            "complete": complete_metrics,
            "iforest": iforest_metrics,
        }
        for array_name, array in arrays.items():
            np.save(
                OUT
                / f"e17_{dataset_name}_{variant}_{array_name}.npy",
                array,
            )
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (OUT / f"e17_{dataset_name}_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(results, ensure_ascii=False, indent=2),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["r42", "r52", "all"],
        default="r42",
    )
    args = parser.parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if args.dataset in {"r42", "all"}:
        run_dataset("r42")
    if args.dataset in {"r52", "all"}:
        run_dataset("r52")


if __name__ == "__main__":
    main()
