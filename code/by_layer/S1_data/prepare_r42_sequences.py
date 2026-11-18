from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "processed_r42" / "r4.2_user_day_features.csv"
OUT_DIR = ROOT / "processed_r42"
ARRAY_FILE = OUT_DIR / "r4.2_sequence_dataset.npz"
META_FILE = OUT_DIR / "r4.2_sequence_metadata.json"

TRAIN_START = pd.Timestamp("2010-01-02")
TRAIN_END = pd.Timestamp("2010-06-09")
VAL_START = pd.Timestamp("2010-06-10")
VAL_END = pd.Timestamp("2010-10-31")
TEST_START = pd.Timestamp("2010-11-01")
TEST_END = pd.Timestamp("2011-05-17")

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
    users = sorted(frame["user_id"].unique())
    dates = pd.date_range(TRAIN_START, TEST_END, freq="D")

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

    train_days = int((TRAIN_END - TRAIN_START).days + 1)
    val_days = int((VAL_END - VAL_START).days + 1)
    test_days = int((TEST_END - TEST_START).days + 1)
    train = values[:, :train_days, :]

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
        active_mask=np.broadcast_to(active_mask, normalized.shape).astype(np.uint8),
        train_end_index=np.asarray(train_days - 1),
        val_start_index=np.asarray(train_days),
        val_end_index=np.asarray(train_days + val_days - 1),
        test_start_index=np.asarray(train_days + val_days),
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
                "start": str(TRAIN_START.date()),
                "end": str(TRAIN_END.date()),
                "days": train_days,
                "indices": [0, train_days - 1],
            },
            "validation": {
                "start": str(VAL_START.date()),
                "end": str(VAL_END.date()),
                "days": val_days,
                "indices": [train_days, train_days + val_days - 1],
            },
            "test": {
                "start": str(TEST_START.date()),
                "end": str(TEST_END.date()),
                "days": test_days,
                "indices": [train_days + val_days, len(dates) - 1],
            },
        },
        "active_cells": {
            "train": int(active_mask[:, :train_days, :].sum()),
            "validation": int(
                active_mask[:, train_days:train_days + val_days, :].sum()
            ),
            "test": int(active_mask[:, train_days + val_days:, :].sum()),
        },
        "missing_cells": int(np.isnan(normalized).sum()),
        "log1p_features": LOG1P_FEATURES,
        "standardization": "training-period non-missing mean/std only",
        "labels_used_for_features": False,
        "labels_used_for_split": False,
        "split_reason": "training ends before the first official malicious event",
    }
    META_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
