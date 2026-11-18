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
    / "r52"
    / "T4_full_soft"
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


SEEDS = [20261116, 20261117, 20261118]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-tag",
        default="missing_safe_v2",
        help="Run directory under e2_instance/r52/T4_full_soft.",
    )
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


def align_raw(
    users: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    frame = pd.read_csv(
        ROOT / "processed_r52" / "r5.2_user_day_features.csv",
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


def soft_labels(profile: np.ndarray) -> tuple[np.ndarray, dict]:
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
    return densify(-normalized, tau_inst, ALPHA).astype(
        np.float32
    ), {
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
    }


def main() -> None:
    args = parse_args()
    out_root = OUT / args.output_tag
    out_root.mkdir(parents=True, exist_ok=True)
    dataset = np.load(
        ROOT / "processed_r52" / "r5.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    train_values = values[:, : train_end + 1]
    raw = align_raw(users, dates)
    profile = profiles(raw, active, train_end)
    labels, metadata = soft_labels(profile)
    np.save(out_root / "soft_instance_labels.npy", labels)
    (out_root / "distance_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for seed in SEEDS:
        start_time = time.time()
        run_dir = out_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
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
            max_train_length=None,
            soft_instance=True,
            soft_temporal=True,
        )
        losses = model.fit(
            train_values,
            0,
            0,
            0,
            labels,
            str(run_dir),
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
        torch.save(model.net.state_dict(), run_dir / "model.pth")
        np.save(run_dir / "masked.npy", masked)
        np.save(run_dir / "plain.npy", plain)
        np.save(
            run_dir / "masked_raw.npy",
            np.abs(masked - plain).sum(axis=-1),
        )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "epochs": 10,
                    "lambda": 0.5,
                    "tau_temp": TAU_TEMP,
                    "soft_instance": True,
                    "soft_temporal": True,
                    "losses": [float(value) for value in losses],
                    "elapsed_seconds": time.time() - start_time,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"r5.2 T4 seed={seed} done in "
            f"{time.time() - start_time:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
