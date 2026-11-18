from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
DATA = ROOT / "OmniAnomaly" / "ServerMachineDataset"
OUT_DIR = WORK / "smd"
SCORE_DIR = OUT_DIR / "scores_v1"
AUDIT_JSON = OUT_DIR / "smd_audit.json"
AUDIT_CSV = OUT_DIR / "smd_audit.csv"
DETAIL_FILE = OUT_DIR / "smd_results_by_machine.csv"
SUMMARY_FILE = OUT_DIR / "smd_summary.json"

TS2VEC_SEED = 20261109
IFOREST_SEEDS = [20261109, 20261110, 20261111]
MAX_TRAIN_LENGTH = 1000

sys.path.insert(0, str(ROOT))
from train_r42_ts2vec import TS2Vec  # noqa: E402


def machine_names() -> list[str]:
    return sorted(
        path.stem
        for path in (DATA / "train").glob("*.txt")
    )


def load_machine(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(
        DATA / "train" / f"{name}.txt", header=None
    ).to_numpy(dtype=np.float64)
    test = pd.read_csv(
        DATA / "test" / f"{name}.txt", header=None
    ).to_numpy(dtype=np.float64)
    label = pd.read_csv(
        DATA / "test_label" / f"{name}.txt", header=None
    ).iloc[:, 0].to_numpy(dtype=np.uint8)
    if len(test) != len(label):
        raise RuntimeError(
            f"{name}: test length {len(test)} != label {len(label)}"
        )
    return train, test, label


def parse_interpretation(name: str) -> list[tuple[int, int, list[int]]]:
    path = DATA / "interpretation_label" / f"{name}.txt"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        span, dimensions = line.split(":", 1)
        start, end = [int(value) for value in span.split("-")]
        dims = [
            int(value)
            for value in dimensions.split(",")
            if value.strip()
        ]
        rows.append((start, end, dims))
    return rows


def contiguous_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[
        0,
        labels.astype(np.int8),
        0,
    ].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def audit() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in machine_names():
        train, test, label = load_machine(name)
        interpretations = parse_interpretation(name)
        rows.append(
            {
                "machine": name,
                "train_points": len(train),
                "test_points": len(test),
                "dimensions": train.shape[1],
                "anomaly_points": int(label.sum()),
                "anomaly_ratio": float(label.mean()),
                "anomaly_segments": len(contiguous_segments(label)),
                "interpretation_segments": len(interpretations),
                "interpretation_dimensions": len(
                    {
                        dim
                        for _, _, dims in interpretations
                        for dim in dims
                    }
                ),
                "missing_train": int(
                    np.isnan(train).sum()
                ),
                "missing_test": int(np.isnan(test).sum()),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT_CSV, index=False, encoding="utf-8")
    summary = {
        "source_dataset": str(DATA),
        "entities": len(frame),
        "train_points": int(frame["train_points"].sum()),
        "test_points": int(frame["test_points"].sum()),
        "anomaly_points": int(frame["anomaly_points"].sum()),
        "pooled_anomaly_ratio": float(
            frame["anomaly_points"].sum()
            / frame["test_points"].sum()
        ),
        "dimensions": sorted(
            frame["dimensions"].unique().tolist()
        ),
        "missing_values": int(
            frame["missing_train"].sum()
            + frame["missing_test"].sum()
        ),
        "audit_csv": str(AUDIT_CSV),
    }
    AUDIT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def robust_score(
    train_values: np.ndarray,
    all_values: np.ndarray,
) -> np.ndarray:
    median = np.median(train_values)
    mad = np.median(np.abs(train_values - median))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = np.std(train_values)
    if scale < 1e-8:
        scale = 1.0
    return np.clip(
        (all_values - median) / scale, 0.0, None
    )


def robust_zscore_point_score(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(train, axis=0)
    scale = np.where(scale < 1e-8, fallback, scale)
    scale = np.where(scale < 1e-8, 1.0, scale)
    train_z = np.abs((train - median) / scale)
    test_z = np.abs((test - median) / scale)
    return (
        np.sort(train_z, axis=1)[:, -3:].mean(axis=1),
        np.sort(test_z, axis=1)[:, -3:].mean(axis=1),
    )


def score_one(name: str, force: bool = False) -> Path:
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    output = SCORE_DIR / f"{name}.npz"
    if output.exists() and not force:
        print(f"Already scored: {name}", flush=True)
        return output

    import torch

    train, test, label = load_machine(name)
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    train_norm = ((train - mean) / std).astype(np.float32)
    test_norm = ((test - mean) / std).astype(np.float32)

    z_train, z_test = robust_zscore_point_score(train, test)

    forest_train = []
    forest_test = []
    for seed in IFOREST_SEEDS:
        model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(train_norm)
        forest_train.append(-model.score_samples(train_norm))
        forest_test.append(-model.score_samples(test_norm))
    forest_train_raw = np.mean(forest_train, axis=0)
    forest_test_raw = np.mean(forest_test, axis=0)
    forest_train = robust_score(
        forest_train_raw, forest_train_raw
    )
    forest_test = robust_score(
        forest_train_raw, forest_test_raw
    )

    np.random.seed(TS2VEC_SEED)
    torch.manual_seed(TS2VEC_SEED)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = TS2Vec(
        input_dims=train_norm.shape[1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        batch_size=16,
        lr=0.001,
        max_train_length=MAX_TRAIN_LENGTH,
    )
    loss_log = model.fit(
        train_norm[None, :, :],
        n_epochs=10,
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
        ),
        dtype=np.float32,
    )
    train_plain = np.asarray(
        model.encode(train_norm[None, :, :], **encode_args),
        dtype=np.float32,
    )
    test_masked = np.asarray(
        model.encode(
            test_norm[None, :, :],
            mask="mask_last",
            **encode_args,
        ),
        dtype=np.float32,
    )
    test_plain = np.asarray(
        model.encode(test_norm[None, :, :], **encode_args),
        dtype=np.float32,
    )
    ts_train = np.abs(
        train_masked - train_plain
    ).sum(axis=-1).reshape(-1)
    ts_test = np.abs(
        test_masked - test_plain
    ).sum(axis=-1).reshape(-1)
    ts_train_raw = ts_train.copy()
    ts_test_raw = ts_test.copy()
    ts_train = robust_score(ts_train_raw, ts_train_raw)
    ts_test = robust_score(ts_train_raw, ts_test_raw)

    np.savez_compressed(
        output,
        machine=name,
        label=label,
        zscore_train=z_train.astype(np.float32),
        zscore_test=z_test.astype(np.float32),
        iforest_train=forest_train.astype(np.float32),
        iforest_test=forest_test.astype(np.float32),
        ts2vec_train=ts_train.astype(np.float32),
        ts2vec_test=ts_test.astype(np.float32),
        ts2vec_loss_first=np.asarray(float(loss_log[0])),
        ts2vec_loss_last=np.asarray(float(loss_log[-1])),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Finished: {name}", flush=True)
    return output


def point_adjust_at_top_fraction(
    label: np.ndarray,
    score: np.ndarray,
    fraction: float,
) -> dict:
    count = max(1, int(np.ceil(len(score) * fraction)))
    threshold = np.partition(score, -count)[-count]
    prediction = score >= threshold
    true_positive = int((prediction & (label == 1)).sum())
    precision = (
        true_positive / int(prediction.sum())
        if prediction.any()
        else 0.0
    )
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
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def threshold_metrics(
    label: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict:
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
    detected_segments = sum(
        prediction[start:end].any()
        for start, end in segments
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "segment_recall": (
            detected_segments / len(segments)
            if segments
            else 0.0
        ),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def analyze() -> None:
    rows = []
    for name in machine_names():
        path = SCORE_DIR / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing score file for {name}"
            )
        with np.load(path, allow_pickle=False) as data:
            label = data["label"].astype(np.uint8)
            for method in ("zscore", "iforest", "ts2vec"):
                train_score = data[
                    f"{method}_train"
                ].astype(np.float64)
                test_score = data[
                    f"{method}_test"
                ].astype(np.float64)
                row = {
                    "machine": name,
                    "method": method,
                    "anomaly_ratio": float(label.mean()),
                    "roc_auc": float(
                        roc_auc_score(label, test_score)
                    ),
                    "pr_auc": float(
                        average_precision_score(
                            label, test_score
                        )
                    ),
                }
                for quantile in (0.95, 0.98, 0.99):
                    threshold = float(
                        np.quantile(train_score, quantile)
                    )
                    metrics = threshold_metrics(
                        label, test_score, threshold
                    )
                    suffix = f"q{int(quantile * 100)}"
                    for metric, value in metrics.items():
                        row[f"{metric}_{suffix}"] = value
                pa = point_adjust_at_top_fraction(
                    label, test_score, fraction=0.05
                )
                row["pa_precision_top5"] = pa["precision"]
                row["pa_recall_top5"] = pa["recall"]
                row["pa_f1_top5"] = pa["f1"]
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(DETAIL_FILE, index=False, encoding="utf-8")
    metric_columns = [
        "roc_auc",
        "pr_auc",
        "precision_q99",
        "recall_q99",
        "f1_q99",
        "fpr_q99",
        "segment_recall_q99",
        "pa_f1_top5",
    ]
    summary = {
        "protocol": {
            "entities": len(machine_names()),
            "train_test_split": "provided SMD split",
            "thresholds": "training-score quantiles 95%, 98%, 99%",
            "test_labels_used_for_selection": False,
            "pa_definition": (
                "Top 5% test scores; each true anomaly segment "
                "counts as adjusted-recall positive if any point "
                "inside it is predicted."
            ),
            "iforest_seeds": IFOREST_SEEDS,
            "ts2vec_seed": TS2VEC_SEED,
            "ts2vec_max_train_length": MAX_TRAIN_LENGTH,
        },
        "macro_mean": {},
        "pairwise_wilcoxon_iforest_vs_ts2vec": {},
        "detail_file": str(DETAIL_FILE),
    }
    for metric in metric_columns:
        summary["macro_mean"][metric] = {
            method: float(
                frame.loc[
                    frame["method"] == method, metric
                ].mean()
            )
            for method in ("zscore", "iforest", "ts2vec")
        }
        iforest = frame.loc[
            frame["method"] == "iforest", metric
        ].to_numpy()
        ts2vec = frame.loc[
            frame["method"] == "ts2vec", metric
        ].to_numpy()
        differences = iforest - ts2vec
        if np.allclose(differences, 0):
            statistic = None
            p_value = 1.0
        else:
            test = wilcoxon(
                iforest, ts2vec, alternative="two-sided"
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        summary["pairwise_wilcoxon_iforest_vs_ts2vec"][
            metric
        ] = {
            "mean_difference": float(differences.mean()),
            "statistic": statistic,
            "p_value": p_value,
        }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["audit", "score", "analyze"],
        required=True,
    )
    parser.add_argument("--machine")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "audit":
        audit()
    elif args.stage == "score":
        names = (
            [args.machine]
            if args.machine
            else machine_names()
        )
        for name in names:
            score_one(name, force=args.force)
    else:
        analyze()


if __name__ == "__main__":
    main()
