from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def machine_names(data_dir: Path) -> list[str]:
    return sorted(path.stem for path in (data_dir / "train").glob("*.txt"))


def load_machine(
    data_dir: Path,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(
        data_dir / "train" / f"{name}.txt",
        header=None,
    ).to_numpy(dtype=np.float64)
    test = pd.read_csv(
        data_dir / "test" / f"{name}.txt",
        header=None,
    ).to_numpy(dtype=np.float64)
    label = pd.read_csv(
        data_dir / "test_label" / f"{name}.txt",
        header=None,
    ).iloc[:, 0].to_numpy(dtype=np.uint8)
    if len(test) != len(label):
        raise ValueError(f"{name}: test and label lengths differ")
    return train, test, label


def normalize_from_train(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) < 1e-8, 1.0, train.std(axis=0))
    return (
        ((train - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
    )


def block_windows(values: np.ndarray, window: int) -> np.ndarray:
    starts = list(range(0, len(values), window))
    blocks = []
    for start in starts:
        block = values[start : start + window]
        if len(block) < window:
            padding = np.zeros(
                (window - len(block), values.shape[1]),
                dtype=values.dtype,
            )
            block = np.concatenate([block, padding], axis=0)
        blocks.append(block)
    return np.stack(blocks).astype(np.float32)


def trim_block_scores(
    scores: np.ndarray,
    length: int,
    window: int,
) -> np.ndarray:
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    return flat[:length]


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


def evaluate_machine(
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
