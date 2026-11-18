from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
MODEL_DIR = OUT / "e17_models"
WINDOW_DAYS = 30
MASK_DAYS = 3

CHANNELS = {
    "logon": list(range(1, 8)),
    "device": [8, 9],
    "file": list(range(10, 14)),
    "email": list(range(14, 20)),
    "http": list(range(20, 24)),
}

sys.path.insert(0, str(ROOT))
from train_r42_ts2vec import add_ts2vec_to_path  # noqa: E402

add_ts2vec_to_path()
from models import TSEncoder  # noqa: E402


def load_student(path: Path, input_dims: int, device) -> TSEncoder:
    student = TSEncoder(
        input_dims=input_dims,
        output_dims=64,
        hidden_dims=64,
        depth=5,
    ).to(device)
    student.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    student.eval()
    return student


def cosine_at_last(
    student: TSEncoder,
    window: torch.Tensor,
    active: torch.Tensor,
    masked_window: torch.Tensor,
) -> float:
    with torch.no_grad():
        full = student(window, mask="all_true")[:, -1]
        masked = student(masked_window, mask="all_true")[:, -1]
        cosine = F.cosine_similarity(full, masked, dim=-1)
        return float(cosine[active].mean().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["r42", "r52"], default="r42"
    )
    args = parser.parse_args()
    if args.dataset == "r42":
        dataset_path = (
            ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
        )
        model_prefix = "r42"
        output_suffix = "r42"
    else:
        dataset_path = (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        )
        model_prefix = "r52"
        output_suffix = "r52"
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
        range(train_end + 1, time_steps, 3)
    )
    variants = ["M0", "M1_time", "M2_channel", "M3_joint"]
    rows = []
    for variant in variants:
        print(f"Evaluating {variant}", flush=True)
        student = load_student(
            MODEL_DIR / f"{model_prefix}_{variant}.pth",
            feature_dims,
            device,
        )
        time_scores = []
        channel_scores = {channel: [] for channel in CHANNELS}
        joint_scores = []
        for time_index in sample_indices:
            start = time_index - WINDOW_DAYS + 1
            window = values[:, start : time_index + 1].clone()
            active = torch.from_numpy(
                active_mask[:, time_index]
            ).to(device)
            if not bool(active.any()):
                continue

            time_masked = window.clone()
            time_masked[:, -MASK_DAYS:, :] = 0.0
            time_scores.append(
                cosine_at_last(
                    student, window, active, time_masked
                )
            )

            for channel, indices in CHANNELS.items():
                masked = window.clone()
                masked[:, -MASK_DAYS:, indices] = 0.0
                channel_scores[channel].append(
                    cosine_at_last(
                        student, window, active, masked
                    )
                )

            joint = window.clone()
            joint[:, -MASK_DAYS:, :] = 0.0
            joint_scores.append(
                cosine_at_last(student, window, active, joint)
            )

        row = {
            "variant": variant,
            "time_consistency": float(np.mean(time_scores)),
            "joint_consistency": float(np.mean(joint_scores)),
            "channel_consistency_mean": float(
                np.mean(
                    [
                        np.mean(values)
                        for values in channel_scores.values()
                    ]
                )
            ),
        }
        for channel, scores in channel_scores.items():
            row[f"{channel}_consistency"] = float(
                np.mean(scores)
            )
        rows.append(row)
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = __import__("pandas").DataFrame(rows)
    frame.to_csv(
        OUT / f"e17_consistency_metrics_{output_suffix}.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "window_days": WINDOW_DAYS,
        "mask_days": MASK_DAYS,
        "sample_stride_days": 3,
        "metrics": rows,
        "outputs": {
            "csv": str(
                OUT / f"e17_consistency_metrics_{output_suffix}.csv"
            )
        },
    }
    (OUT / f"e17_consistency_summary_{output_suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
