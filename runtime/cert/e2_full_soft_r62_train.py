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
    / "r62"
    / "T4_full_soft"
)
SOFTCLT_DIR = (
    ROOT.parents[1]
    / "references"
    / "softclt_repo"
    / "softclt_ts2vec"
)
sys.path.insert(0, str(SOFTCLT_DIR))
from soft_ts2vec import TS2Vec as SoftTS2Vec  # noqa: E402


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
TAU_TEMP = 1.5
ALPHA = 0.5
SEEDS = [20261116, 20261117, 20261118]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        default=SEEDS[0],
    )
    parser.add_argument(
        "--max-train-length",
        type=int,
        default=0,
        help=(
            "Randomly crop each training sequence to this length; "
            "0 keeps the complete 222-day training window."
        ),
    )
    parser.add_argument(
        "--train-users",
        type=int,
        default=1000,
        help=(
            "Number of users in the deterministic training subset. "
            "Use 4000 for full-user training."
        ),
    )
    parser.add_argument(
        "--output-tag",
        default="missing_safe_v2",
        help="Run directory under the configured r6.2 output root.",
    )
    return parser.parse_args()


def densify(values: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    return (
        (2.0 * alpha) / (1.0 + np.exp(-tau * values))
        + (1.0 - alpha) * np.eye(values.shape[0])
    )


def align_raw(
    users: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    frame = pd.read_parquet(
        ROOT / "processed_r62" / "r6.2_user_day_features.parquet",
        columns=["user_id", "date", *PROFILE_FEATURES],
    )
    frame["date"] = pd.to_datetime(frame["date"])
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
    count_dimensions = len(RAW_FEATURES)
    counts = values[:, :, :count_dimensions]
    values[:, :, :count_dimensions] = np.where(
        np.isfinite(counts),
        np.log1p(np.maximum(counts, 0.0)),
        np.nan,
    )
    timing = values[:, :, count_dimensions:]
    values[:, :, count_dimensions:] = np.where(
        np.isfinite(timing) & (timing >= 0.0),
        timing,
        np.nan,
    )
    return values.astype(np.float32)


def profiles(
    raw: np.ndarray,
    active: np.ndarray,
    train_end: int,
) -> np.ndarray:
    rows = []
    for user_index in range(raw.shape[0]):
        mask = active[user_index, : train_end + 1]
        values = raw[user_index, : train_end + 1][mask]
        if len(values) == 0:
            values = raw[user_index, : train_end + 1]
        rows.append(
            np.concatenate(
                [
                    np.nanmedian(values, axis=0),
                    np.nanquantile(values, 0.9, axis=0),
                    np.asarray([mask.mean()]),
                ]
            )
        )
    profile = np.stack(rows)
    population_median = np.nanmedian(profile, axis=0)
    population_median = np.where(
        np.isfinite(population_median),
        population_median,
        0.0,
    )
    missing_rows, missing_columns = np.nonzero(~np.isfinite(profile))
    profile[missing_rows, missing_columns] = population_median[
        missing_columns
    ]
    mean = profile.mean(axis=0)
    std = np.where(
        profile.std(axis=0) < 1e-8,
        1.0,
        profile.std(axis=0),
    )
    return ((profile - mean) / std).astype(np.float64)


def main() -> None:
    args = parse_args()
    base_out_root = (
        OUT
        if args.train_users == 1000
        else OUT.parent / f"T4_full_soft_users{args.train_users}"
    )
    out_root = base_out_root / args.output_tag
    out = out_root / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    dataset = np.load(
        ROOT / "processed_r62" / "r6.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    rng = np.random.default_rng(2024)
    train_user_indices = np.sort(
        rng.choice(
            len(users),
            size=min(args.train_users, len(users)),
            replace=False,
        )
    )
    train_values = values[
        train_user_indices, : train_end + 1
    ]
    raw = align_raw(users, dates)
    profile = profiles(
        raw[train_user_indices],
        active[train_user_indices],
        train_end,
    )
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
    median_distance = float(
        np.median(normalized[np.triu_indices(len(normalized), k=1)])
    )
    tau_inst = 1.0 / max(median_distance, 1e-6)
    labels = densify(-normalized, tau_inst, ALPHA).astype(np.float32)
    metadata = {
        "dataset": "r6.2",
        "distance": "user_profile_euclidean",
        "profile_dimensions": int(profile.shape[1]),
        "profile_features": PROFILE_FEATURES,
        "tau_inst": tau_inst,
        "median_distance": median_distance,
        "alpha": ALPHA,
        "missing_policy": (
            "timing sentinel -1 -> NaN; per-user nanmedian/nanquantile; "
            "fully missing feature -> training population median"
        ),
        "train_users": int(len(train_user_indices)),
        "training_scope": (
            "full_user_population"
            if len(train_user_indices) == len(users)
            else f"fixed_{len(train_user_indices)}_user_subset"
        ),
        "max_train_length": (
            args.max_train_length
            if args.max_train_length > 0
            else None
        ),
        "train_user_indices": train_user_indices.tolist(),
    }
    (out / "distance_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = SoftTS2Vec(
        input_dims=values.shape[-1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        lr=1e-3,
        batch_size=16,
        lambda_=0.5,
        tau_temp=TAU_TEMP,
        max_train_length=(
            args.max_train_length
            if args.max_train_length > 0
            else None
        ),
        soft_instance=True,
        soft_temporal=True,
    )
    start_time = time.time()
    losses = model.fit(
        train_values,
        0,
        0,
        0,
        labels,
        str(out),
        n_epochs=10,
        verbose=True,
    )
    encode_args = {
        "casual": True,
        "sliding_length": 1,
        "sliding_padding": 30,
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
    torch.save(model.net.state_dict(), out / "model.pth")
    np.save(out / "masked.npy", masked)
    np.save(out / "plain.npy", plain)
    np.save(
        out / "masked_raw.npy",
        np.abs(masked - plain).sum(axis=-1),
    )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "tau_temp": TAU_TEMP,
                "batch_size": 16,
                "max_train_length": (
                    args.max_train_length
                    if args.max_train_length > 0
                    else None
                ),
                "losses": [float(value) for value in losses],
                "elapsed_seconds": time.time() - start_time,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"r6.2 T4 seed={args.seed} done in "
        f"{time.time() - start_time:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
