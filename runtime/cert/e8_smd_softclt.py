from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    average_precision_score,
    pairwise_distances,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e8"
    / "smd"
)
SMD_BASELINE = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e7"
    / "smd"
    / "results_by_machine.csv"
)
SOFTCLT_DIR = (
    ROOT.parents[1]
    / "references"
    / "softclt_repo"
    / "softclt_ts2vec"
)
DATA = ROOT / "OmniAnomaly" / "ServerMachineDataset"
SEEDS = [20261116, 20261117, 20261118]
WINDOW = 512
TAU_TEMP = 1.5
ALPHA = 0.5
K_BANK = 500
TOP_K = 3
T2_WINDOW = 32
PAI_W_G = 0.6
PAI_W_Q = 0.4

sys.path.insert(0, str(SOFTCLT_DIR))
from soft_ts2vec import TS2Vec as SoftTS2Vec  # noqa: E402


def machine_names() -> list[str]:
    return sorted(
        path.stem for path in (DATA / "train").glob("*.txt")
    )


def load_machine(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(
        DATA / "train" / f"{name}.txt",
        header=None,
    ).to_numpy(dtype=np.float64)
    test = pd.read_csv(
        DATA / "test" / f"{name}.txt",
        header=None,
    ).to_numpy(dtype=np.float64)
    label = pd.read_csv(
        DATA / "test_label" / f"{name}.txt",
        header=None,
    ).iloc[:, 0].to_numpy(dtype=np.uint8)
    if len(test) != len(label):
        raise RuntimeError(
            f"{name}: test length {len(test)} != label {len(label)}"
        )
    return train, test, label


def contiguous_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[0, labels.astype(np.int8), 0]
    changes = np.diff(padded)
    return list(
        zip(
            np.flatnonzero(changes == 1).tolist(),
            np.flatnonzero(changes == -1).tolist(),
        )
    )


def threshold_metrics(
    label: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = score >= threshold
    tp = int((prediction & (label == 1)).sum())
    fp = int((prediction & (label == 0)).sum())
    fn = int((~prediction & (label == 1)).sum())
    tn = int((~prediction & (label == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0
    segments = contiguous_segments(label)
    detected = sum(
        bool(prediction[start:end].any())
        for start, end in segments
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "segment_recall": (
            detected / len(segments) if segments else 0.0
        ),
    }


def point_adjust_at_top_fraction(
    label: np.ndarray,
    score: np.ndarray,
    fraction: float,
) -> dict[str, float]:
    count = max(1, int(np.ceil(len(score) * fraction)))
    threshold = np.partition(score, -count)[-count]
    prediction = score >= threshold
    true_positive = int((prediction & (label == 1)).sum())
    precision = true_positive / int(prediction.sum())
    adjusted = prediction.copy()
    for start, end in contiguous_segments(label):
        if prediction[start:end].any():
            adjusted[start:end] = True
    adjusted_tp = int((adjusted & (label == 1)).sum())
    recall = adjusted_tp / int(label.sum())
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def causal_moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(
        [
            np.zeros((1, values.shape[1])),
            np.cumsum(values, axis=0),
        ],
        axis=0,
    )
    output = np.empty_like(values)
    for time_index in range(len(values)):
        start = max(0, time_index - window + 1)
        output[time_index] = (
            cumulative[time_index + 1] - cumulative[start]
        ) / (time_index - start + 1)
    return output


def amplitude_scores(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-6)
    train_g = np.abs(train - median) / scale
    test_g = np.abs(test - median) / scale
    train_q = (
        np.abs(causal_moving_mean(train, T2_WINDOW) - median)
        / scale
    )
    test_q = (
        np.abs(causal_moving_mean(test, T2_WINDOW) - median)
        / scale
    )
    return (
        np.stack(
            [train_g.mean(axis=1), train_q.mean(axis=1)],
            axis=1,
        ),
        np.stack(
            [test_g.mean(axis=1), test_q.mean(axis=1)],
            axis=1,
        ),
    )


def bank_distances(
    train_repr: np.ndarray,
    test_repr: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clusters = min(
        K_BANK,
        max(1, int(round(0.1 * len(train_repr)))),
    )
    clusters = min(clusters, len(train_repr) - 1)
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        init="k-means++",
        random_state=2024,
        batch_size=max(8192, clusters),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    ).fit(train_repr)
    bank = torch.from_numpy(
        kmeans.cluster_centers_.astype(np.float32)
    ).to(device)
    bank_norm = torch.nn.functional.normalize(bank, dim=1)

    def score(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        output_cos = []
        output_euc = []
        with torch.inference_mode():
            for start in range(0, len(values), 4096):
                query = torch.from_numpy(
                    values[start : start + 4096].astype(np.float32)
                ).to(device)
                query_norm = torch.nn.functional.normalize(query, dim=1)
                cosine = (
                    1.0 - query_norm @ bank_norm.T
                ).clamp_min_(0.0)
                euclidean = torch.cdist(query, bank)
                output_cos.append(
                    cosine.topk(
                        TOP_K,
                        dim=1,
                        largest=False,
                    ).values.mean(dim=1).cpu().numpy()
                )
                output_euc.append(
                    euclidean.topk(
                        TOP_K,
                        dim=1,
                        largest=False,
                    ).values.mean(dim=1).cpu().numpy()
                )
        return (
            np.concatenate(output_cos).astype(np.float32),
            np.concatenate(output_euc).astype(np.float32),
        )

    train_cos, train_euc = score(train_repr)
    test_cos, test_euc = score(test_repr)
    return train_cos, train_euc, test_cos, test_euc


def z_from_train(
    train_score: np.ndarray,
    test_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = float(train_score.mean())
    std = max(float(train_score.std()), 1e-8)
    return (
        ((train_score - mean) / std).astype(np.float32),
        ((test_score - mean) / std).astype(np.float32),
    )


def evaluate_machine(
    train: np.ndarray,
    test: np.ndarray,
    label: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float]:
    del train, test
    q99 = float(np.quantile(train_scores, 0.99))
    threshold = threshold_metrics(label, test_scores, q99)
    pa = point_adjust_at_top_fraction(label, test_scores, 0.05)
    return {
        "roc_auc": float(roc_auc_score(label, test_scores)),
        "pr_auc": float(average_precision_score(label, test_scores)),
        "recall_q99": threshold["recall"],
        "precision_q99": threshold["precision"],
        "f1_q99": threshold["f1"],
        "fpr_q99": threshold["fpr"],
        "segment_recall_q99": threshold["segment_recall"],
        "pa_f1_top5": pa["f1"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        default=SEEDS[0],
    )
    parser.add_argument("--machines", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def densify(
    values: np.ndarray,
    tau: float,
    alpha: float,
) -> np.ndarray:
    return (
        (2.0 * alpha) / (1.0 + np.exp(-tau * values))
        + (1.0 - alpha) * np.eye(values.shape[0])
    )


def make_windows(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(train) // WINDOW
    if count < 2:
        raise ValueError("At least two complete windows are required.")
    usable = train[: count * WINDOW].reshape(
        count,
        WINDOW,
        train.shape[1],
    )
    profiles = np.concatenate(
        [
            np.median(usable, axis=1),
            np.quantile(usable, 0.9, axis=1),
            np.std(usable, axis=1),
        ],
        axis=1,
    )
    mean = profiles.mean(axis=0)
    std = np.where(profiles.std(axis=0) < 1e-8, 1.0, profiles.std(axis=0))
    return usable.astype(np.float32), ((profiles - mean) / std).astype(
        np.float64
    )


def soft_instance_labels(profile: np.ndarray) -> tuple[np.ndarray, dict]:
    distances = pairwise_distances(profile, metric="euclidean")
    upper = distances[np.triu_indices(len(distances), k=1)]
    minimum = float(upper.min())
    maximum = float(distances.max())
    scale = maximum - minimum
    normalized = (
        (distances - minimum) / scale
        if scale > 1e-12
        else np.zeros_like(distances)
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    med = float(
        np.median(normalized[np.triu_indices(len(normalized), k=1)])
    )
    tau = 1.0 / max(med, 1e-6)
    labels = densify(-normalized, tau, ALPHA).astype(np.float32)
    return labels, {
        "distance": "window_profile_euclidean",
        "tau_inst": tau,
        "median_distance": med,
        "alpha": ALPHA,
        "window": WINDOW,
    }


def run_machine(name: str, seed: int, force: bool) -> dict:
    seed_root = OUT / f"seed_{seed}"
    output = seed_root / "machines" / f"{name}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        with np.load(output, allow_pickle=False) as data:
            return json.loads(str(data["metrics"]))

    start = time.time()
    train, test, label = load_machine(name)
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) < 1e-8, 1.0, train.std(axis=0))
    train_norm = ((train - mean) / std).astype(np.float32)
    test_norm = ((test - mean) / std).astype(np.float32)
    windows, profile = make_windows(train_norm)
    soft_labels, distance_metadata = soft_instance_labels(profile)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_id = 0 if torch.cuda.is_available() else "cpu"
    model = SoftTS2Vec(
        input_dims=train_norm.shape[1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device_id,
        batch_size=16,
        lr=0.001,
        lambda_=0.5,
        tau_temp=TAU_TEMP,
        max_train_length=None,
        soft_instance=True,
        soft_temporal=True,
    )
    losses = model.fit(
        windows,
        0,
        0,
        0,
        soft_labels,
        str(seed_root / "runs" / name),
        n_epochs=10,
        verbose=False,
    )
    encode_args = {
        "casual": True,
        "sliding_length": 1,
        "sliding_padding": 30,
        "batch_size": 32,
    }
    train_masked = np.asarray(
        model.encode(
            train_norm[None, :, :],
            mask="mask_last",
            **encode_args,
        )
    ).reshape(-1, 64)
    train_plain = np.asarray(
        model.encode(train_norm[None, :, :], **encode_args)
    ).reshape(-1, 64)
    test_masked = np.asarray(
        model.encode(
            test_norm[None, :, :],
            mask="mask_last",
            **encode_args,
        )
    ).reshape(-1, 64)
    test_plain = np.asarray(
        model.encode(test_norm[None, :, :], **encode_args)
    ).reshape(-1, 64)
    train_original = np.abs(train_masked - train_plain).sum(axis=1)
    test_original = np.abs(test_masked - test_plain).sum(axis=1)

    device = torch.device(
        f"cuda:{device_id}" if isinstance(device_id, int) else "cpu"
    )
    train_cos, train_euc, test_cos, test_euc = bank_distances(
        train_plain,
        test_plain,
        device=device,
    )
    train_amp, test_amp = amplitude_scores(train, test)
    train_original_z, test_original_z = z_from_train(
        train_original,
        test_original,
    )
    train_cos_z, test_cos_z = z_from_train(train_cos, test_cos)
    train_euc_z, test_euc_z = z_from_train(train_euc, test_euc)
    train_g_z, test_g_z = z_from_train(train_amp[:, 0], test_amp[:, 0])
    train_q_z, test_q_z = z_from_train(train_amp[:, 1], test_amp[:, 1])
    train_pai = (
        train_euc_z + PAI_W_G * train_g_z + PAI_W_Q * train_q_z
    )
    test_pai = (
        test_euc_z + PAI_W_G * test_g_z + PAI_W_Q * test_q_z
    )
    variants = {
        "original": (train_original, test_original),
        "cos": (train_cos, test_cos),
        "euc": (train_euc, test_euc),
        "pai": (train_pai, test_pai),
    }
    metrics = {
        "machine": name,
        "seed": seed,
        "train_windows": int(len(windows)),
        "distance_metadata": json.dumps(distance_metadata),
    }
    arrays = {
        "label": label,
        "loss_first": np.asarray(losses[0]),
        "loss_last": np.asarray(losses[-1]),
    }
    for variant, (train_score, test_score) in variants.items():
        metric = evaluate_machine(
            train,
            test,
            label,
            train_score,
            test_score,
        )
        for key, value in metric.items():
            metrics[f"{variant}_{key}"] = float(value)
        arrays[f"{variant}_train_score"] = train_score.astype(np.float32)
        arrays[f"{variant}_test_score"] = test_score.astype(np.float32)
    metrics["elapsed_seconds"] = time.time() - start
    np.savez_compressed(
        output,
        **arrays,
        metrics=json.dumps(metrics),
    )
    print(
        f"{name}: T4 PR={metrics['pai_pr_auc']:.4f}, "
        f"time={metrics['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return metrics


def macro_summary(frame: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    metrics = [
        "roc_auc",
        "pr_auc",
        "recall_q99",
        "precision_q99",
        "f1_q99",
        "fpr_q99",
        "segment_recall_q99",
        "pa_f1_top5",
    ]
    summary = {
        "machines": int(len(frame)),
        "macro_mean": {
            variant: {
                metric: float(frame[f"{variant}_{metric}"].mean())
                for metric in metrics
            }
            for variant in ("original", "cos", "euc", "pai")
        },
    }
    merged = frame.merge(
        baseline,
        on="machine",
        suffixes=("_t4", "_t0"),
        validate="one_to_one",
    )
    pairwise = {}
    pairs = [
        ("pai", "pai", "pai_joint_vs_t0_pai"),
        ("euc", "euc", "euc_joint_vs_t0_euc"),
        ("original", "original", "mask_joint_vs_t0"),
        ("pai", "euc", "pai_vs_euc_within_joint"),
    ]
    for left, right, label in pairs:
        for metric in ("roc_auc", "pr_auc", "pa_f1_top5"):
            left_key = f"{left}_{metric}_t4"
            right_key = (
                f"{right}_{metric}_t4"
                if label == "pai_vs_euc_within_joint"
                else f"{right}_{metric}_t0"
            )
            delta = merged[left_key] - merged[right_key]
            nonzero = delta.to_numpy()
            p_value = (
                float(wilcoxon(nonzero).pvalue)
                if len(nonzero) >= 2 and np.any(nonzero != 0)
                else None
            )
            pairwise[f"{label}_{metric}"] = {
                "mean_difference": float(delta.mean()),
                "positive_machines": int((delta > 0).sum()),
                "negative_machines": int((delta < 0).sum()),
                "wilcoxon_p": p_value,
            }
    summary["pairwise"] = pairwise
    return summary


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = args.machines or machine_names()
    rows = []
    for name in selected:
        rows.append(run_machine(name, args.seed, args.force))
    frame = pd.DataFrame(rows)
    seed_root = OUT / f"seed_{args.seed}"
    frame.to_csv(
        seed_root / "results_by_machine.csv",
        index=False,
        encoding="utf-8",
    )
    baseline = (
        pd.read_csv(SMD_BASELINE)
        if SMD_BASELINE.exists()
        else pd.DataFrame()
    )
    summary = macro_summary(frame, baseline) if len(baseline) else {
        "machines": int(len(frame)),
        "macro_mean": frame.mean(numeric_only=True).to_dict(),
    }
    summary["seed"] = args.seed
    summary["window"] = WINDOW
    (seed_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
