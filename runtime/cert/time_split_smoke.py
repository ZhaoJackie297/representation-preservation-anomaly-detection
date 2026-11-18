from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "processed" / "r1_user_day.csv"
OUT_DIR = ROOT / "processed"
SPLIT_JSON = OUT_DIR / "r1_time_split.json"
SMOKE_JSON = OUT_DIR / "r1_ts2vec_smoke.json"
ARRAY_FILE = OUT_DIR / "r1_smoke_arrays.npz"
USERS_FILE = OUT_DIR / "r1_smoke_users.csv"

TRAIN_START = pd.Timestamp("2010-01-04")
TRAIN_END = pd.Timestamp("2010-10-31")
VAL_START = pd.Timestamp("2010-11-01")
VAL_END = pd.Timestamp("2011-02-06")
TEST_START = pd.Timestamp("2011-02-07")
TEST_END = pd.Timestamp("2011-05-14")

FEATURES = [
    "logon_count",
    "logoff_count",
    "first_logon_minute",
    "last_logoff_minute",
    "active_span_minutes",
    "after_hours_logon_count",
    "usb_connect_count",
    "usb_disconnect_count",
    "http_count",
    "unique_http_domains",
    "external_http_count",
    "after_hours_http_count",
    "is_weekend",
]


def add_ts2vec_to_path() -> None:
    configured = os.environ.get("TS2VEC_REPO")
    candidates = [
        Path(configured) if configured else None,
        Path(r"D:\Software\ts2vec\ts2vec-main"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "ts2vec.py").is_file():
            sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError("TS2Vec repository not found.")


add_ts2vec_to_path()
from ts2vec import TS2Vec  # noqa: E402


def complete_users(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> set[str]:
    expected_days = (end - start).days + 1
    segment = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    counts = segment.groupby("user_id")["date"].nunique()
    return set(counts[counts == expected_days].index)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT, parse_dates=["date"])
    frame = frame.sort_values(["user_id", "date"]).reset_index(drop=True)

    train_users = complete_users(frame, TRAIN_START, TRAIN_END)
    val_users = complete_users(frame, VAL_START, VAL_END)
    test_users = complete_users(frame, TEST_START, TEST_END)
    selected_users = sorted(train_users & val_users & test_users)
    if not selected_users:
        raise RuntimeError("No users fully cover all three time segments.")

    dates = pd.date_range(TRAIN_START, TEST_END, freq="D")
    subset = frame[frame["user_id"].isin(selected_users)].copy()
    keyed = subset.set_index(["user_id", "date"])[FEATURES]
    index = pd.MultiIndex.from_product([selected_users, dates], names=["user_id", "date"])
    keyed = keyed.reindex(index)
    missing_cells = int(keyed.isna().sum().sum())
    if missing_cells:
        raise RuntimeError(f"Selected users have {missing_cells} missing feature cells.")

    tensor = keyed.to_numpy(dtype=np.float32).reshape(
        len(selected_users), len(dates), len(FEATURES)
    )
    train_days = int((TRAIN_END - TRAIN_START).days + 1)
    val_days = int((VAL_END - VAL_START).days + 1)
    test_days = int((TEST_END - TEST_START).days + 1)
    if train_days + val_days + test_days != len(dates):
        raise RuntimeError("Time split does not cover the full date range.")

    train = tensor[:, :train_days, :]
    mean = train.mean(axis=(0, 1), keepdims=True)
    scale = train.std(axis=(0, 1), keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    tensor_normalized = (tensor - mean) / scale
    train_normalized = tensor_normalized[:, :train_days, :]

    split_summary = {
        "train": {
            "start": str(TRAIN_START.date()),
            "end": str(TRAIN_END.date()),
            "days": train_days,
        },
        "validation": {
            "start": str(VAL_START.date()),
            "end": str(VAL_END.date()),
            "days": val_days,
        },
        "test": {
            "start": str(TEST_START.date()),
            "end": str(TEST_END.date()),
            "days": test_days,
        },
        "users_full_train": len(train_users),
        "users_full_validation": len(val_users),
        "users_full_test": len(test_users),
        "users_full_all_splits": len(selected_users),
        "selected_users": selected_users,
        "feature_columns": FEATURES,
        "label_warning": "CERT r1 has no malicious labels; no detection metrics computed.",
    }
    SPLIT_JSON.write_text(json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"user_id": selected_users}).to_csv(USERS_FILE, index=False)

    np.random.seed(20261109)
    torch.manual_seed(20261109)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = TS2Vec(
        input_dims=len(FEATURES),
        output_dims=16,
        hidden_dims=32,
        depth=3,
        device=device,
        batch_size=min(6, len(selected_users)),
    )
    loss_log = model.fit(train_normalized, n_epochs=10, verbose=False)

    encode_args = {
        "causal": True,
        "sliding_length": 1,
        "sliding_padding": 14,
        "batch_size": len(selected_users),
    }
    masked = np.asarray(
        model.encode(tensor_normalized, mask="mask_last", **encode_args)
    )
    plain = np.asarray(model.encode(tensor_normalized, **encode_args))
    errors = np.abs(masked - plain).sum(axis=-1)

    cutoff = train_days - 1
    probe = tensor_normalized[[0]].copy()
    perturbed = probe.copy()
    perturbed[:, cutoff + 1 :, :] += 10.0
    probe_repr = np.asarray(
        model.encode(probe, mask="mask_last", **encode_args)
    )
    perturbed_repr = np.asarray(
        model.encode(perturbed, mask="mask_last", **encode_args)
    )
    causal_max_diff = float(
        np.max(np.abs(probe_repr[:, : cutoff + 1] - perturbed_repr[:, : cutoff + 1]))
    )
    repeat = np.asarray(model.encode(probe, mask="mask_last", **encode_args))
    repeat_max_diff = float(np.max(np.abs(probe_repr - repeat)))

    smoke = {
        "tensor_shape": list(tensor.shape),
        "train_shape": list(train.shape),
        "masked_repr_shape": list(masked.shape),
        "plain_repr_shape": list(plain.shape),
        "daily_error_shape": list(errors.shape),
        "loss_first": float(loss_log[0]),
        "loss_last": float(loss_log[-1]),
        "causal_cutoff_last_train_index": cutoff,
        "causal_future_perturbation_max_diff_before_cutoff": causal_max_diff,
        "repeat_encode_max_diff": repeat_max_diff,
        "selected_user_count": len(selected_users),
        "labels_used": False,
        "metrics_computed": False,
    }
    SMOKE_JSON.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        ARRAY_FILE,
        users=np.asarray(selected_users),
        dates=np.asarray(dates.astype(str)),
        features=np.asarray(FEATURES),
        values=tensor_normalized,
        masked_repr=masked,
        plain_repr=plain,
        daily_error=errors,
    )
    print(json.dumps(split_summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(smoke, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
