from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "processed_r52" / "r5.2_user_day_features.csv"
LABELS = ROOT / "processed_r52" / "r5.2_malicious_account_days.csv"
OUT_DIR = ROOT / "processed_r52"
ARRAY_FILE = OUT_DIR / "r5.2_sequence_dataset.npz"
META_FILE = OUT_DIR / "r5.2_sequence_metadata.json"

FEATURES = [
    "has_any_event",
    "logon_count",
    "logoff_count",
    "first_logon_minute",
    "last_logoff_minute",
    "active_span_minutes",
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
    "is_weekend",
    "weekday_sin",
    "weekday_cos",
]

LOG1P_FEATURES = [
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_input = INPUT.with_suffix(".parquet")
    frame = (
        pd.read_parquet(parquet_input)
        if parquet_input.exists()
        else pd.read_csv(INPUT, parse_dates=["date"])
    )
    labels = pd.read_csv(LABELS, parse_dates=["date"])
    users = sorted(frame["user_id"].unique())

    train_start = frame["date"].min().normalize()
    first_label_date = labels["date"].min().normalize()
    train_end = first_label_date - pd.Timedelta(days=1)
    test_start = first_label_date
    final_end = frame["date"].max().normalize()
    dates = pd.date_range(train_start, final_end, freq="D")
    if train_end < train_start:
        raise RuntimeError("No clean pre-incident training period.")
    if (labels["date"] <= train_end).any():
        raise RuntimeError("Malicious labels leaked into clean training.")

    base_features = FEATURES[:-2]
    value_columns = base_features + ["__exists"]
    frame["__exists"] = 1.0
    indexed = frame.set_index(["user_id", "date"])[value_columns]
    full_index = pd.MultiIndex.from_product(
        [users, dates], names=["user_id", "date"]
    )
    aligned = indexed.reindex(full_index)
    exists = aligned.pop("__exists").notna()
    exists_matrix = exists.to_numpy().reshape(len(users), len(dates))

    weekday = np.asarray(dates.weekday, dtype=np.float32)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)
    weekday_sin = np.broadcast_to(weekday_sin, (len(users), len(dates)))
    weekday_cos = np.broadcast_to(weekday_cos, (len(users), len(dates)))

    base_values = aligned.to_numpy(dtype=np.float32).reshape(
        len(users), len(dates), len(base_features)
    )
    values = np.empty((len(users), len(dates), len(FEATURES)), dtype=np.float32)
    values[:, :, : len(base_features)] = base_values
    values[:, :, -2] = np.where(exists_matrix, weekday_sin, np.nan)
    values[:, :, -1] = np.where(exists_matrix, weekday_cos, np.nan)

    for index, feature in enumerate(FEATURES):
        if feature in LOG1P_FEATURES:
            mask = ~np.isnan(values[:, :, index])
            values[:, :, index][mask] = np.log1p(values[:, :, index][mask])

    train_exclusive_end = int(
        (train_end - train_start).days + 1
    )
    train_end_index = train_exclusive_end - 1
    test_start_index = int((test_start - train_start).days)
    train = values[:, :train_exclusive_end, :]

    mean = np.nanmean(train, axis=(0, 1), keepdims=True)
    std = np.nanstd(train, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (values - mean) / std

    active_mask = exists_matrix.astype(np.uint8).reshape(
        len(users), len(dates), 1
    )
    np.savez_compressed(
        ARRAY_FILE,
        users=np.asarray(users, dtype="U32"),
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        features=np.asarray(FEATURES, dtype="U64"),
        values=normalized.astype(np.float32),
        active_mask=np.broadcast_to(
            active_mask, normalized.shape
        ).astype(np.uint8),
        train_end_index=np.asarray(train_end_index),
        test_start_index=np.asarray(test_start_index),
        test_end_index=np.asarray(len(dates) - 1),
        mean=mean.squeeze(0).squeeze(0),
        std=std.squeeze(0).squeeze(0),
    )

    metadata = {
        "array_file": str(ARRAY_FILE),
        "shape": list(normalized.shape),
        "users": len(users),
        "dates": len(dates),
        "features": FEATURES,
        "splits": {
            "train": {
                "start": str(train_start.date()),
                "end": str(train_end.date()),
                "days": train_exclusive_end,
                "indices": [0, train_end_index],
                "malicious_account_days": int(
                    (labels["date"] <= train_end).sum()
                ),
            },
            "external_test": {
                "start": str(test_start.date()),
                "end": str(final_end.date()),
                "days": len(dates) - test_start_index,
                "indices": [test_start_index, len(dates) - 1],
                "malicious_account_days": int(
                    (labels["date"] >= test_start).sum()
                ),
            },
        },
        "active_cells": {
            "train": int(active_mask[:, :train_exclusive_end, :].sum()),
            "external_test": int(
                active_mask[:, test_start_index:, :].sum()
            ),
        },
        "missing_cells": int(np.isnan(normalized).sum()),
        "log1p_features": LOG1P_FEATURES,
        "standardization": "training-period non-missing mean/std only",
        "labels_used_for_features": False,
        "labels_used_for_split": (
            "Only the first official malicious date was used to place a "
            "global clean cutoff before all incidents."
        ),
        "test_performance_used_for_split": False,
        "test_performance_used_for_model_selection": False,
        "external_evaluation_design": (
            "One-shot replication: all dates after the globally clean "
            "cutoff are external evaluation data; no validation split is "
            "used to choose models or hyperparameters."
        ),
    }
    META_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
