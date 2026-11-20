from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e2_instance"
)
OUT.mkdir(parents=True, exist_ok=True)
SOFTCLT_DIR = (
    ROOT.parents[1]
    / "references"
    / "softclt_repo"
    / "softclt_ts2vec"
)
sys.path.insert(0, str(SOFTCLT_DIR))
from soft_ts2vec import TS2Vec as SoftTS2Vec  # noqa: E402

sys.path.insert(0, str(ROOT))
from e2_softclt_training import (  # noqa: E402
    build_alerts,
    evaluate_scores,
)
from feature_contracts import (  # noqa: E402
    build_user_profiles,
    sanitize_user_day_features,
)
from quota_relationship_sim import load_r42  # noqa: E402


PADDING_DAYS = 30
TAU_TEMP = 1.5

RAW_FEATURES = [
    "logon_count",
    "logoff_count",
    "after_hours_logon_count",
    "other_pc_logon_count",
    "usb_connect_count",
    "usb_disconnect_count",
    "file_copy_count",
    "archive_copy_count",
    "executable_copy_count",
    "unique_file_extensions",
    "email_count",
    "email_recipient_count",
    "external_recipient_count",
    "email_attachment_count",
    "email_size_bytes",
    "after_hours_email_count",
    "http_count",
    "after_hours_http_count",
    "external_http_count",
    "unique_http_domains",
]
PROFILE_FEATURES = [
    *RAW_FEATURES,
    "first_logon_minute",
    "last_logoff_minute",
    "active_span_minutes",
]


def densify(
    values: np.ndarray,
    tau: float,
    alpha: float,
) -> np.ndarray:
    return (
        (2.0 * alpha) / (1.0 + np.exp(-tau * values))
        + (1.0 - alpha) * np.eye(values.shape[0])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20261116],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-tag",
        default="missing_safe_v2",
        help=(
            "Subdirectory under e2_instance for this run. The old "
            "results remain untouched."
        ),
    )
    return parser.parse_args()


def align_raw(
    users: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    frame = pd.read_csv(
        ROOT / "processed_r42" / "r4.2_user_day_features.csv",
        usecols=["user_id", "date", *PROFILE_FEATURES],
        parse_dates=["date"],
    )
    index = pd.MultiIndex.from_product(
        [users.tolist(), pd.to_datetime(dates)],
        names=["user_id", "date"],
    )
    aligned = frame.set_index(
        ["user_id", "date"]
    ).reindex(index)
    values = aligned[PROFILE_FEATURES].to_numpy(
        dtype=np.float64
    ).reshape(len(users), len(dates), len(PROFILE_FEATURES))
    return sanitize_user_day_features(
        values,
        count_dimensions=len(RAW_FEATURES),
    )


def soft_instance_labels(
    profiles: np.ndarray,
    *,
    alpha: float = 0.5,
) -> tuple[np.ndarray, dict]:
    distances = pairwise_distances(profiles, metric="euclidean")
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
    median_distance = float(
        np.median(normalized[np.triu_indices(len(normalized), k=1)])
    )
    tau_inst = 1.0 / max(median_distance, 1e-6)
    soft_labels = densify(-normalized, tau_inst, alpha)
    return soft_labels.astype(np.float32), {
        "distance": "user_profile_euclidean",
        "profile_dimensions": int(profiles.shape[1]),
        "profile_features": PROFILE_FEATURES,
        "tau_inst_from_median_distance": tau_inst,
        "median_pairwise_distance": median_distance,
        "alpha": alpha,
        "missing_policy": (
            "timing sentinel -1 -> NaN; per-user nanmedian/nanquantile; "
            "fully missing feature -> training population median"
        ),
    }


def train_and_evaluate(
    *,
    name: str,
    seed: int,
    soft_temporal: bool,
    epochs: int,
    batch_size: int,
    values: np.ndarray,
    train_values: np.ndarray,
    soft_labels: np.ndarray,
    data: dict,
    out_root: Path,
) -> pd.DataFrame:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = 0 if torch.cuda.is_available() else "cpu"
    run_dir = out_root / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model = SoftTS2Vec(
        input_dims=values.shape[-1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        lr=1e-3,
        batch_size=batch_size,
        lambda_=0.5,
        tau_temp=TAU_TEMP,
        max_train_length=None,
        soft_instance=True,
        soft_temporal=soft_temporal,
    )
    losses = model.fit(
        train_values,
        0,
        0,
        0,
        soft_labels,
        str(run_dir),
        n_epochs=epochs,
        verbose=True,
    )
    encode_args = {
        "casual": True,
        "sliding_length": 1,
        "sliding_padding": PADDING_DAYS,
        "batch_size": 256,
    }
    masked = np.asarray(
        model.encode(values, mask="mask_last", **encode_args),
        dtype=np.float32,
    )
    plain = np.asarray(
        model.encode(values, **encode_args),
        dtype=np.float32,
    )
    raw_score = np.abs(masked - plain).sum(axis=-1)
    np.save(run_dir / "masked_raw.npy", raw_score)
    torch.save(model.net.state_dict(), run_dir / "model.pth")
    metrics = evaluate_scores(
        data,
        raw_score,
        data["active_mask"],
        int(data["train_end"]),
        int(data["val_end"]),
        name,
    )
    metrics.to_csv(
        run_dir / "metrics.csv",
        index=False,
        encoding="utf-8",
    )
    metadata = {
        "name": name,
        "seed": seed,
        "soft_instance": True,
        "soft_temporal": soft_temporal,
        "tau_temp": TAU_TEMP if soft_temporal else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "losses": [float(value) for value in losses],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = parse_args()
    run_root = OUT / args.output_tag
    run_root.mkdir(parents=True, exist_ok=True)
    dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    train_values = values[:, : train_end + 1]
    data = load_r42()
    raw = align_raw(users, dates)
    profiles = build_user_profiles(raw, active, train_end)
    soft_labels, distance_meta = soft_instance_labels(profiles)
    np.save(run_root / "soft_instance_labels.npy", soft_labels)
    (run_root / "distance_metadata.json").write_text(
        json.dumps(distance_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    variants = [
        ("T3_instance_soft", False),
        ("T4_full_soft", True),
    ]
    frames = []
    for seed in args.seeds:
        for name, soft_temporal in variants:
            start_time = time.time()
            metrics = train_and_evaluate(
                name=name,
                seed=seed,
                soft_temporal=soft_temporal,
                epochs=args.epochs,
                batch_size=args.batch_size,
                values=values,
                train_values=train_values,
                soft_labels=soft_labels,
                data=data,
                out_root=run_root,
            )
            metrics.insert(0, "seed", seed)
            frames.append(metrics)
            print(
                f"{name} seed={seed} done in "
                f"{time.time() - start_time:.1f}s",
                flush=True,
            )
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(
        run_root / "metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        frame.groupby(["method", "split", "k"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            precision_at_k_mean=("precision_at_k", "mean"),
            incident_hits_mean=("incident_hits", "mean"),
        )
        .sort_values(["split", "k", "method"])
    )
    summary.to_csv(
        run_root / "metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
