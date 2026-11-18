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
MODEL_DIR = OUT / "e18_models"
WINDOW_DAYS = 30
MASK_DAYS = 3
ANALYSIS_STRIDE = 7
MAX_RELATION_USERS = 256

sys.path.insert(0, str(ROOT))
from e17_consistency_analysis import (  # noqa: E402
    CHANNELS,
    load_student,
)
from e17_multigranular_mask import load_base_student  # noqa: E402


def masked_student_last(
    student,
    window: torch.Tensor,
    indices: list[int] | None,
) -> torch.Tensor:
    masked = window.clone()
    if indices is None:
        masked[:, -MASK_DAYS:, :] = 0.0
    else:
        masked[:, -MASK_DAYS:, indices] = 0.0
    with torch.no_grad():
        return student(masked, mask="all_true")[:, -1]


def pair_geometry(
    student_masked: torch.Tensor,
    teacher_full: torch.Tensor,
) -> dict[str, float]:
    if student_masked.shape[0] < 2:
        return {
            "positive_cosine": float("nan"),
            "hard_negative_gap": float("nan"),
            "nce_accuracy": float("nan"),
            "relation_correlation": float("nan"),
            "relation_mse": float("nan"),
        }

    student = F.normalize(student_masked, dim=-1)
    teacher = F.normalize(teacher_full, dim=-1)
    positive = (student * teacher).sum(dim=-1)
    logits = student @ teacher.T
    batch_size = student.shape[0]
    labels = torch.arange(batch_size, device=student.device)
    off_diagonal = ~torch.eye(
        batch_size, dtype=torch.bool, device=student.device
    )
    negative_logits = logits.masked_fill(~off_diagonal, -2.0)
    hardest_negative = negative_logits.max(dim=-1).values
    accuracy = (
        logits.argmax(dim=-1) == labels
    ).float().mean()

    student_relation = student @ student.T
    teacher_relation = teacher @ teacher.T
    student_off = student_relation[off_diagonal]
    teacher_off = teacher_relation[off_diagonal]
    centered_student = student_off - student_off.mean()
    centered_teacher = teacher_off - teacher_off.mean()
    correlation = (
        (centered_student * centered_teacher).sum()
        / (
            centered_student.square().sum().sqrt()
            * centered_teacher.square().sum().sqrt()
            + 1e-8
        )
    )
    relation_mse = F.mse_loss(student_off, teacher_off)
    return {
        "positive_cosine": float(positive.mean().item()),
        "hard_negative_gap": float(
            (positive - hardest_negative).mean().item()
        ),
        "nce_accuracy": float(accuracy.item()),
        "relation_correlation": float(correlation.item()),
        "relation_mse": float(relation_mse.item()),
    }


def run_dataset(dataset_name: str) -> None:
    if dataset_name == "r42":
        dataset_path = (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        )
        teacher_path = (
            ROOT / "processed_r42" / "r4.2_ts2vec_model.pth"
        )
    else:
        dataset_path = (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        )
        teacher_path = (
            ROOT / "processed_r52" / "r5.2_ts2vec_model.pth"
        )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    dataset = np.load(dataset_path, allow_pickle=False)
    values = torch.from_numpy(
        dataset["values"].astype(np.float32)
    ).to(device)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    users, time_steps, feature_dims = values.shape
    sample_indices = list(
        range(train_end + 1, time_steps, ANALYSIS_STRIDE)
    )
    variants = ["M0", "C0", "N1", "R1", "NR1"]
    teacher, _ = load_base_student(teacher_path, feature_dims)
    rows = []

    for variant in variants:
        print(f"Evaluating {variant}", flush=True)
        student = load_student(
            MODEL_DIR / f"{dataset_name}_{variant}.pth",
            feature_dims,
            device,
        )
        time_scores = []
        joint_scores = []
        channel_scores = {channel: [] for channel in CHANNELS}
        geometry_rows = []
        for time_index in sample_indices:
            start = time_index - WINDOW_DAYS + 1
            window = values[:, start : time_index + 1].clone()
            active = torch.from_numpy(
                active_mask[:, time_index]
            ).to(device)
            if not bool(active.any()):
                continue

            with torch.no_grad():
                full = student(window, mask="all_true")[:, -1]
                teacher_full_all = teacher(
                    window, mask="all_true"
                )[:, -1]
                time_masked = student(
                    torch.cat(
                        [
                            window[:, : -MASK_DAYS, :],
                            torch.zeros_like(
                                window[:, -MASK_DAYS:, :]
                            ),
                        ],
                        dim=1,
                    ),
                    mask="all_true",
                )[:, -1]
                time_scores.append(
                    float(
                        F.cosine_similarity(
                            full[active], time_masked[active], dim=-1
                        ).mean()
                    )
                )

            joint_rep = masked_student_last(
                student, window, None
            )
            joint_scores.append(
                float(
                    F.cosine_similarity(
                        full[active], joint_rep[active], dim=-1
                    ).mean()
                )
            )

            for channel, indices in CHANNELS.items():
                channel_rep = masked_student_last(
                    student, window, indices
                )
                channel_scores[channel].append(
                    float(
                        F.cosine_similarity(
                            full[active],
                            channel_rep[active],
                            dim=-1,
                        ).mean()
                    )
                )

            active_indices = torch.nonzero(
                active, as_tuple=False
            ).flatten()
            relation_indices = active_indices[
                :MAX_RELATION_USERS
            ]
            if relation_indices.numel() >= 2:
                geometry_rows.append(
                    pair_geometry(
                        joint_rep[relation_indices],
                        teacher_full_all[relation_indices],
                    )
                )

        row = {
            "variant": variant,
            "time_consistency": float(np.mean(time_scores)),
            "joint_consistency": float(np.mean(joint_scores)),
            "channel_consistency_mean": float(
                np.mean(
                    [
                        np.mean(scores)
                        for scores in channel_scores.values()
                    ]
                )
            ),
        }
        for metric in (
            "positive_cosine",
            "hard_negative_gap",
            "nce_accuracy",
            "relation_correlation",
            "relation_mse",
        ):
            row[metric] = float(
                np.nanmean(
                    [values_[metric] for values_ in geometry_rows]
                )
            )
        rows.append(row)
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    csv_path = OUT / f"e18_representation_metrics_{dataset_name}.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    summary = {
        "window_days": WINDOW_DAYS,
        "mask_days": MASK_DAYS,
        "analysis_stride_days": ANALYSIS_STRIDE,
        "max_relation_users": MAX_RELATION_USERS,
        "metrics": rows,
        "outputs": {"csv": str(csv_path)},
    }
    json_path = OUT / f"e18_representation_summary_{dataset_name}.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["r42", "r52", "all"],
        default="r42",
    )
    args = parser.parse_args()
    if args.dataset in {"r42", "all"}:
        run_dataset("r42")
    if args.dataset in {"r52", "all"}:
        run_dataset("r52")


if __name__ == "__main__":
    main()
