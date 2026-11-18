from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "experiments" / "representation_preservation"
SOFTCLT_DIR = (
    ROOT.parents[1] / "references" / "softclt_repo" / "softclt_ts2vec"
)
sys.path.insert(0, str(SOFTCLT_DIR))
from soft_ts2vec import TS2Vec as SoftTS2Vec  # noqa: E402

sys.path.insert(0, str(ROOT))
from quota_relationship_sim import load_r42  # noqa: E402


VARIANTS = {
    "T0": "e2/runs/r42_T0_seed{seed}/model.pth",
    "T1": "e2/runs/r42_T1_seed{seed}/model.pth",
    "T2_tau1.5": "e2/runs/r42_T2_tau1.5_seed{seed}/model.pth",
    "T3_instance_soft": (
        "e2_instance/missing_safe_v2_20261117/"
        "T3_instance_soft/seed_{seed}/model.pth"
    ),
    "T4_full_soft": (
        "e2_instance/missing_safe_v2_20261117/"
        "T4_full_soft/seed_{seed}/model.pth"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20261116],
    )
    parser.add_argument(
        "--output-tag",
        default="pilot_20261117",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
    )
    return parser.parse_args()


def resolve_model_path(variant: str, seed: int) -> Path:
    return BASE / VARIANTS[variant].format(seed=seed)


def load_model(path: Path) -> SoftTS2Vec:
    device = 0 if torch.cuda.is_available() else "cpu"
    model = SoftTS2Vec(
        input_dims=27,
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        batch_size=16,
        lambda_=0.5,
        tau_temp=1.5,
        soft_instance=False,
        soft_temporal=False,
    )
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    model.net.load_state_dict(state)
    model.net.eval()
    return model


def normalized_dirichlet(
    representation: torch.Tensor,
    weights: torch.Tensor,
    *,
    axis: str,
) -> float:
    """Return tr(Z^T L_W Z) / tr(Z^T D_W Z)."""
    weights = weights.clone()
    weights.fill_diagonal_(0.0)
    degree = weights.sum(dim=1)
    if axis == "time":
        weighted_sum = torch.einsum(
            "ts,usd->utd",
            weights,
            representation,
        )
        mean = weighted_sum / degree[None, :, None].clamp_min(1e-12)
        numerator = (
            degree[None, :, None]
            * (representation - mean).pow(2)
        ).sum()
        denominator = (
            degree[None, :, None] * representation.pow(2)
        ).sum()
    elif axis == "user":
        users, times, dimensions = representation.shape
        numerator = representation.new_tensor(0.0)
        denominator = representation.new_tensor(0.0)
        for time_index in range(times):
            current = representation[:, time_index, :]
            mean = weights @ current
            mean /= degree[:, None].clamp_min(1e-12)
            numerator = numerator + (
                degree[:, None]
                * (current - mean).pow(2)
            ).sum()
            denominator = denominator + (
                degree[:, None] * current.pow(2)
            ).sum()
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return float((numerator / denominator.clamp_min(1e-12)).cpu())


def temporal_kernel(
    length: int,
    tau: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    index = torch.arange(length, dtype=torch.float32, device=device)
    distance = (index[:, None] - index[None, :]).abs()
    return 2.0 / (1.0 + torch.exp(tau * distance))


def pool_representation(
    representation: torch.Tensor,
) -> torch.Tensor:
    return F.max_pool1d(
        representation.transpose(1, 2),
        kernel_size=2,
    ).transpose(1, 2)


def day_level_scores(
    representation: torch.Tensor,
    instance_weights: torch.Tensor,
    temporal_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    users, times, _ = representation.shape
    instance_degree = instance_weights.sum(dim=1)
    temporal_degree = temporal_weights.sum(dim=1)

    instance_score = torch.empty(
        (users, times),
        dtype=torch.float32,
        device=representation.device,
    )
    for time_index in range(times):
        current = representation[:, time_index, :]
        mean = instance_weights @ current
        mean /= instance_degree[:, None].clamp_min(1e-12)
        instance_score[:, time_index] = (
            current - mean
        ).pow(2).sum(dim=1)

    weighted_sum = torch.einsum(
        "ts,usd->utd",
        temporal_weights,
        representation,
    )
    temporal_mean = weighted_sum / temporal_degree[
        None, :, None
    ].clamp_min(1e-12)
    temporal_score = (
        representation - temporal_mean
    ).pow(2).sum(dim=-1)
    return instance_score, temporal_score


def metric_auc(
    score: torch.Tensor,
    active: np.ndarray,
    labels: np.ndarray,
    start: int,
    end: int,
) -> dict:
    score_np = score.cpu().numpy()[:, start : end + 1]
    active_np = active[:, start : end + 1]
    label_np = labels[:, start : end + 1]
    valid = active_np & np.isfinite(score_np)
    y_true = label_np[valid].astype(int)
    y_score = score_np[valid]
    return {
        "positive_days": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "mean_normal": float(y_score[y_true == 0].mean()),
        "mean_anomaly": float(y_score[y_true == 1].mean()),
    }


def main() -> None:
    args = parse_args()
    out = BASE / "s2_graph_spectral" / args.output_tag
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
    dataset = np.load(dataset_path, allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    test_start = int(dataset["test_start_index"])
    test_end = int(dataset["test_end_index"])
    data = load_r42()
    labels = data["labels"]

    soft_path = (
        BASE
        / "e2_instance"
        / "missing_safe_v2_20261117"
        / "soft_instance_labels.npy"
    )
    instance_weights_np = np.load(soft_path)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    instance_weights = torch.from_numpy(
        instance_weights_np.astype(np.float32)
    ).to(device)

    rows = []
    for seed in args.seeds:
        for variant in VARIANTS:
            model_path = resolve_model_path(variant, seed)
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model = load_model(model_path)
            representation_np = np.asarray(
                model.encode(
                    values,
                    casual=True,
                    sliding_length=1,
                    sliding_padding=30,
                    batch_size=args.encode_batch_size,
                ),
                dtype=np.float32,
            )
            representation = torch.from_numpy(
                representation_np
            ).to(device)

            instance_energies = []
            temporal_energies = []
            pooled = representation[:, : train_end + 1, :]
            for depth in range(8):
                if pooled.shape[1] < 2:
                    break
                temporal = temporal_kernel(
                    pooled.shape[1],
                    tau=1.5 * (2**depth),
                    device=device,
                )
                instance_energies.append(
                    normalized_dirichlet(
                        pooled,
                        instance_weights,
                        axis="user",
                    )
                )
                temporal_energies.append(
                    normalized_dirichlet(
                        pooled,
                        temporal,
                        axis="time",
                    )
                )
                if pooled.shape[1] == 2:
                    break
                pooled = pool_representation(pooled)

            day_temporal = temporal_kernel(
                representation.shape[1],
                tau=1.5,
                device=device,
            )
            instance_score, temporal_score = day_level_scores(
                representation,
                instance_weights,
                day_temporal,
            )
            instance_test = metric_auc(
                instance_score,
                active,
                labels,
                test_start,
                test_end,
            )
            temporal_test = metric_auc(
                temporal_score,
                active,
                labels,
                test_start,
                test_end,
            )
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "instance_dirichlet_mean": float(
                        np.mean(instance_energies)
                    ),
                    "temporal_dirichlet_mean": float(
                        np.mean(temporal_energies)
                    ),
                    "instance_highpass_roc": instance_test["roc_auc"],
                    "instance_highpass_pr": instance_test["pr_auc"],
                    "temporal_highpass_roc": temporal_test["roc_auc"],
                    "temporal_highpass_pr": temporal_test["pr_auc"],
                    "temporal_normal_energy": temporal_test[
                        "mean_normal"
                    ],
                    "temporal_anomaly_energy": temporal_test[
                        "mean_anomaly"
                    ],
                    "positive_days": temporal_test["positive_days"],
                }
            )
            print(
                seed,
                variant,
                rows[-1],
                flush=True,
            )
            del model, representation, pooled
            del instance_score, temporal_score
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(
        out / "spectral_metrics_by_seed.csv",
        index=False,
    )
    summary = (
        frame.groupby("variant", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            instance_dirichlet_mean=(
                "instance_dirichlet_mean",
                "mean",
            ),
            temporal_dirichlet_mean=(
                "temporal_dirichlet_mean",
                "mean",
            ),
            instance_highpass_roc_mean=(
                "instance_highpass_roc",
                "mean",
            ),
            temporal_highpass_roc_mean=(
                "temporal_highpass_roc",
                "mean",
            ),
            instance_highpass_pr_mean=(
                "instance_highpass_pr",
                "mean",
            ),
            temporal_highpass_pr_mean=(
                "temporal_highpass_pr",
                "mean",
            ),
            temporal_normal_energy_mean=(
                "temporal_normal_energy",
                "mean",
            ),
            temporal_anomaly_energy_mean=(
                "temporal_anomaly_energy",
                "mean",
            ),
        )
    )
    summary.to_csv(
        out / "spectral_metrics_summary.csv",
        index=False,
    )
    (out / "audit.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "soft_instance_weights": str(soft_path),
                "seeds": args.seeds,
                "training_end_index": train_end,
                "validation_end_index": val_end,
                "test_range": [test_start, test_end],
                "labels_used_for_training": False,
                "labels_used_for_metric_selection": False,
                "energy_definition": (
                    "normalized graph Dirichlet energy "
                    "tr(Z^T L_W Z) / tr(Z^T D_W Z)"
                ),
                "temporal_tau_base": 1.5,
                "temporal_scales": 8,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
