from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e18_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 30
MASK_DAYS = 3
WINDOW_STRIDE = 5
EPOCHS = 2
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
SEED = 20261017
NCE_TEMPERATURE = 0.1
NCE_WEIGHT = 0.1
RELATION_WEIGHT = 0.5

sys.path.insert(0, str(ROOT))
from e17_multigranular_mask import (  # noqa: E402
    CHANNELS,
    ENCODE_ARGS,
    K,
    OUT as E17_OUT,
    Q_LIMIT,
    build_application_scores,
    build_mask_views,
    build_training_windows,
    evaluate_signal,
    load_base_student,
)
from quota_relationship_sim import load_r42, load_r52  # noqa: E402


VARIANTS = {
    "C0": {
        "include_time": True,
        "include_channel": True,
        "nce": False,
        "relation": False,
    },
    "N1": {
        "include_time": True,
        "include_channel": True,
        "nce": True,
        "relation": False,
    },
    "R1": {
        "include_time": True,
        "include_channel": True,
        "nce": False,
        "relation": True,
    },
    "NR1": {
        "include_time": True,
        "include_channel": True,
        "nce": True,
        "relation": True,
    },
}


def symmetric_instance_nce(
    student_last: torch.Tensor,
    teacher_last: torch.Tensor,
    *,
    temperature: float = NCE_TEMPERATURE,
) -> torch.Tensor:
    """Match each masked student window to its own teacher window."""

    if student_last.shape[0] < 2:
        return torch.zeros((), device=student_last.device)
    student = F.normalize(student_last, dim=-1)
    teacher = F.normalize(teacher_last, dim=-1)
    logits = student @ teacher.T / temperature
    labels = torch.arange(
        student.shape[0], device=student.device
    )
    forward = F.cross_entropy(logits, labels)
    reverse = F.cross_entropy(logits.T, labels)
    return 0.5 * (forward + reverse)


def pairwise_relation_loss(
    student_last: torch.Tensor,
    teacher_last: torch.Tensor,
) -> torch.Tensor:
    """Preserve cross-window similarity geometry, excluding the diagonal."""

    if student_last.shape[0] < 2:
        return torch.zeros((), device=student_last.device)
    student = F.normalize(student_last, dim=-1)
    teacher = F.normalize(teacher_last, dim=-1)
    student_relation = student @ student.T
    teacher_relation = teacher @ teacher.T
    batch_size = student.shape[0]
    off_diagonal = ~torch.eye(
        batch_size, dtype=torch.bool, device=student.device
    )
    return F.mse_loss(
        student_relation[off_diagonal],
        teacher_relation[off_diagonal],
    )


def finetune_variant(
    base_student: torch.nn.Module,
    train_windows: torch.Tensor,
    *,
    variant: str,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    config = VARIANTS[variant]
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    student = copy.deepcopy(base_student).to(device)
    teacher = copy.deepcopy(base_student).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student.train()

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
    loss_log: list[dict[str, float]] = []

    for epoch in range(EPOCHS):
        totals = {
            "total": 0.0,
            "full_align": 0.0,
            "mask_align": 0.0,
            "nce": 0.0,
            "relation": 0.0,
        }
        iterations = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_full = teacher(batch, mask="all_true")
                teacher_last = teacher_full[:, -1]

            student_full = student(batch, mask="all_true")
            full_align = (
                1.0
                - F.cosine_similarity(
                    student_full, teacher_full, dim=-1
                )
            ).mean()
            loss = full_align
            mask_align_sum = torch.zeros((), device=device)
            nce_sum = torch.zeros((), device=device)
            relation_sum = torch.zeros((), device=device)
            view_count = 0

            for masked_view, position_mask, _ in build_mask_views(
                batch,
                config["include_time"],
                config["include_channel"],
            ):
                student_masked = student(
                    masked_view, mask="all_true"
                )
                cosine = F.cosine_similarity(
                    student_masked, teacher_full, dim=-1
                )
                selected = cosine[position_mask]
                if selected.numel() > 0:
                    mask_align = (1.0 - selected).mean()
                else:
                    mask_align = torch.zeros((), device=device)
                mask_align_sum = mask_align_sum + mask_align
                student_last = student_masked[:, -1]
                if config["nce"]:
                    nce_sum = nce_sum + symmetric_instance_nce(
                        student_last, teacher_last
                    )
                if config["relation"]:
                    relation_sum = (
                        relation_sum
                        + pairwise_relation_loss(
                            student_last, teacher_last
                        )
                    )
                view_count += 1

            if view_count:
                nce_mean = nce_sum / view_count
                relation_mean = relation_sum / view_count
            else:
                nce_mean = torch.zeros((), device=device)
                relation_mean = torch.zeros((), device=device)
            loss = (
                loss
                + mask_align_sum
                + NCE_WEIGHT * nce_mean
                + RELATION_WEIGHT * relation_mean
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=1.0
            )
            optimizer.step()

            totals["total"] += float(loss.item())
            totals["full_align"] += float(full_align.item())
            totals["mask_align"] += float(mask_align_sum.item())
            totals["nce"] += float(nce_mean.item())
            totals["relation"] += float(relation_mean.item())
            iterations += 1

        average = {
            key: value / max(iterations, 1)
            for key, value in totals.items()
        }
        average["epoch"] = float(epoch + 1)
        loss_log.append(average)
        print(
            f"  {variant} epoch {epoch + 1}: "
            f"total={average['total']:.6f}, "
            f"full={average['full_align']:.6f}, "
            f"mask={average['mask_align']:.6f}, "
            f"nce={average['nce']:.6f}, "
            f"relation={average['relation']:.6f}",
            flush=True,
        )

    student.eval()
    return student, loss_log


def run_dataset(dataset_name: str) -> None:
    if dataset_name == "r42":
        dataset_file = (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        )
        model_file = ROOT / "processed_r42" / "r4.2_ts2vec_model.pth"
        data = load_r42()
        external = False
        auxiliary_all = np.load(
            E17_OUT / "v2_iforest_alignment_arrays.npz",
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
            E17_OUT / "v2_iforest_alignment_arrays.npz",
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
    for variant in ("M0", "C0", "N1", "R1", "NR1"):
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
            student,
            values,
            data,
            train_end,
            device,
        )
        complete_metrics = evaluate_signal(
            data,
            signal,
            auxiliary,
            f"E18_{variant}_complete",
            external=external,
            use_iforest=False,
        )
        iforest_metrics = evaluate_signal(
            data,
            signal,
            auxiliary,
            f"E18_{variant}_iforest",
            external=external,
            use_iforest=True,
        )
        results[variant] = {
            "config": (
                {"base_ts2vec": True}
                if variant == "M0"
                else VARIANTS[variant]
            ),
            "losses": losses,
            "model": str(model_path),
            "complete": complete_metrics,
            "iforest": iforest_metrics,
        }
        for array_name, array in arrays.items():
            np.save(
                OUT
                / f"e18_{dataset_name}_{variant}_{array_name}.npy",
                array,
            )
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "protocol": {
            "window_days": WINDOW_DAYS,
            "mask_days": MASK_DAYS,
            "window_stride": WINDOW_STRIDE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "nce_temperature": NCE_TEMPERATURE,
            "nce_weight": NCE_WEIGHT,
            "relation_weight": RELATION_WEIGHT,
            "labels_used_for_training": False,
            "k": K,
            "q_limit": Q_LIMIT,
        },
        "results": results,
    }
    (OUT / f"e18_{dataset_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


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
