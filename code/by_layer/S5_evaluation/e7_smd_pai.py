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
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e7"
    / "smd"
)
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "OmniAnomaly" / "ServerMachineDataset"
SEED = 20261109
K_BANK = 500
TOP_K = 3
T2_WINDOW = 32
PAI_W_G = 0.6
PAI_W_Q = 0.4

sys.path.insert(0, str(ROOT))
from smd_pipeline import (  # noqa: E402
    contiguous_segments,
    load_machine,
    machine_names,
    point_adjust_at_top_fraction,
    threshold_metrics,
)
from train_r42_ts2vec import TS2Vec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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
        np.stack([train_g.mean(axis=1), train_q.mean(axis=1)], axis=1),
        np.stack([test_g.mean(axis=1), test_q.mean(axis=1)], axis=1),
    )


def bank_distances(
    train_repr: np.ndarray,
    test_repr: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clusters = min(K_BANK, max(1, int(round(0.1 * len(train_repr)))))
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
                query_norm = torch.nn.functional.normalize(
                    query, dim=1
                )
                cos_dist = (
                    1.0 - query_norm @ bank_norm.T
                ).clamp_min_(0.0)
                euc_dist = torch.cdist(query, bank)
                output_cos.append(
                    cos_dist.topk(
                        TOP_K,
                        dim=1,
                        largest=False,
                    ).values.mean(dim=1).cpu().numpy()
                )
                output_euc.append(
                    euc_dist.topk(
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


def run_machine(name: str, epochs: int, force: bool) -> dict:
    output = OUT / "machines" / f"{name}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        data = np.load(output, allow_pickle=False)
        return json.loads(str(data["metrics"]))
    start_time = time.time()
    train, test, label = load_machine(name)
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) < 1e-8, 1.0, train.std(axis=0))
    train_norm = ((train - mean) / std).astype(np.float32)
    test_norm = ((test - mean) / std).astype(np.float32)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device_id = 0 if torch.cuda.is_available() else "cpu"
    model = TS2Vec(
        input_dims=train.shape[1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device_id,
        batch_size=16,
        lr=0.001,
        max_train_length=1000,
    )
    losses = model.fit(
        train_norm[None, :, :],
        n_epochs=epochs,
        verbose=False,
    )
    encode_args = {
        "causal": True,
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
    train_g_z, test_g_z = z_from_train(
        train_amp[:, 0],
        test_amp[:, 0],
    )
    train_q_z, test_q_z = z_from_train(
        train_amp[:, 1],
        test_amp[:, 1],
    )
    train_pai = (
        train_euc_z + PAI_W_G * train_g_z + PAI_W_Q * train_q_z
    )
    test_pai = (
        test_euc_z + PAI_W_G * test_g_z + PAI_W_Q * test_q_z
    )
    scores = {
        "original": (train_original, test_original),
        "cos": (train_cos, test_cos),
        "euc": (train_euc, test_euc),
        "pai": (train_pai, test_pai),
    }
    metrics = {"machine": name}
    arrays = {
        "label": label,
        "loss_first": np.asarray(losses[0]),
        "loss_last": np.asarray(losses[-1]),
    }
    for variant, (train_score, test_score) in scores.items():
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
    metrics["elapsed_seconds"] = time.time() - start_time
    np.savez_compressed(output, **arrays, metrics=json.dumps(metrics))
    print(
        f"{name}: original PR={metrics['original_pr_auc']:.4f}, "
        f"PAI PR={metrics['pai_pr_auc']:.4f}, "
        f"time={metrics['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return metrics


def main() -> None:
    args = parse_args()
    selected = args.machines or machine_names()
    rows_path = OUT / "results_by_machine.csv"
    existing = {}
    if rows_path.exists():
        for row in pd.read_csv(rows_path).to_dict(orient="records"):
            existing[str(row["machine"])] = row
    for name in selected:
        try:
            result = run_machine(name, args.epochs, args.force)
            existing[name] = result
            pd.DataFrame(existing.values()).sort_values(
                "machine"
            ).to_csv(rows_path, index=False, encoding="utf-8")
        except Exception as error:
            print(f"FAILED {name}: {error}", flush=True)
            raise
    frame = pd.DataFrame(existing.values()).sort_values("machine")
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
    summary = {}
    for variant in ("original", "cos", "euc", "pai"):
        summary[variant] = {
            metric: float(frame[f"{variant}_{metric}"].mean())
            for metric in metrics
        }
    pairwise = {}
    for left, right in [
        ("pai", "original"),
        ("euc", "cos"),
        ("pai", "euc"),
    ]:
        for metric in ("pr_auc", "roc_auc", "pa_f1_top5"):
            delta = frame[f"{left}_{metric}"] - frame[f"{right}_{metric}"]
            pairwise[f"{left}_vs_{right}_{metric}"] = {
                "mean_difference": float(delta.mean()),
                "positive_machines": int((delta > 0).sum()),
                "wilcoxon_p": float(
                    wilcoxon(
                        frame[f"{left}_{metric}"],
                        frame[f"{right}_{metric}"],
                        alternative="two-sided",
                    ).pvalue
                ),
            }
    result = {
        "machines": len(frame),
        "macro_mean": summary,
        "pairwise": pairwise,
    }
    (OUT / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
