from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
LABELS = PROCESSED / "r4.2_malicious_account_days.csv"
MODEL_FILE = PROCESSED / "r4.2_ts2vec_model.pth"
SCORE_FILE = PROCESSED / "r4.2_ts2vec_daily_errors.csv"
SUMMARY_FILE = PROCESSED / "r4.2_ts2vec_training.json"
TRAINING_CHECKPOINT = PROCESSED / "r4.2_ts2vec_training_checkpoint.json"


def add_ts2vec_to_path() -> None:
    configured = os.environ.get("TS2VEC_REPO")
    candidates = [
        Path(configured) if configured else None,
        ROOT.parents[1] / "references" / "ts2vec_official",
        Path(r"D:\Software\ts2vec\ts2vec-main"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "ts2vec.py").is_file():
            sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError("TS2Vec repository not found.")


add_ts2vec_to_path()
from ts2vec import TS2Vec  # noqa: E402


def robust_scores(train_errors: np.ndarray, all_errors: np.ndarray) -> np.ndarray:
    median = float(np.median(train_errors))
    mad = float(np.median(np.abs(train_errors - median)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(np.std(train_errors))
    if scale < 1e-8:
        scale = 1.0
    return np.clip((all_errors - median) / scale, 0.0, None)


def main() -> None:
    np.random.seed(20261109)
    torch.manual_seed(20261109)

    dataset = np.load(DATASET, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    features = dataset["features"].astype(str)
    values = dataset["values"].astype(np.float32)
    active_mask = dataset["active_mask"].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_start = int(dataset["val_start_index"])
    val_end = int(dataset["val_end_index"])
    test_start = int(dataset["test_start_index"])
    test_end = int(dataset["test_end_index"])

    train_values = values[:, : train_end + 1, :]
    device = 0 if torch.cuda.is_available() else "cpu"
    model = TS2Vec(
        input_dims=values.shape[-1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        batch_size=16,
    )
    if "--reuse-model" in sys.argv and MODEL_FILE.exists():
        print("Loading existing TS2Vec model...", flush=True)
        load_device = f"cuda:{device}" if isinstance(device, int) else "cpu"
        state_dict = torch.load(
            MODEL_FILE, map_location=load_device, weights_only=True
        )
        model.net.load_state_dict(state_dict)
        if TRAINING_CHECKPOINT.exists():
            checkpoint = json.loads(
                TRAINING_CHECKPOINT.read_text(encoding="utf-8")
            )
            loss_first = float(checkpoint["loss_first"])
            loss_last = float(checkpoint["loss_last"])
        else:
            loss_first = None
            loss_last = None
    else:
        print("Training TS2Vec...", flush=True)
        loss_log = model.fit(train_values, n_epochs=10, verbose=False)
        loss_first = float(loss_log[0])
        loss_last = float(loss_log[-1])
        print(
            f"Training finished: {loss_first:.6f} -> {loss_last:.6f}",
            flush=True,
        )
        model.save(str(MODEL_FILE))
        TRAINING_CHECKPOINT.write_text(
            json.dumps(
                {
                    "loss_first": loss_first,
                    "loss_last": loss_last,
                    "epochs": 10,
                    "training_shape": list(train_values.shape),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    encode_args = {
        "causal": True,
        "sliding_length": 1,
        "sliding_padding": 30,
        "batch_size": 32,
    }
    print("Encoding masked representations...", flush=True)
    masked = np.asarray(
        model.encode(values, mask="mask_last", **encode_args),
        dtype=np.float32,
    )
    print(f"Masked representation shape: {masked.shape}", flush=True)
    print("Encoding unmasked representations...", flush=True)
    plain = np.asarray(model.encode(values, **encode_args), dtype=np.float32)
    print(f"Unmasked representation shape: {plain.shape}", flush=True)
    raw_error = np.abs(masked - plain).sum(axis=-1)

    train_mask = active_mask[:, : train_end + 1, 0] & np.isfinite(
        raw_error[:, : train_end + 1]
    )
    train_errors = raw_error[:, : train_end + 1][train_mask]
    if len(train_errors) == 0:
        raise RuntimeError("No active training errors.")
    scores = robust_scores(train_errors, raw_error)

    labels = pd.read_csv(LABELS, parse_dates=["date"])
    label_map = {
        (row.user_id, row.date.normalize()): dict(
            scenarios=row.scenarios,
            insider_ids=row.insider_ids,
        )
        for row in labels.itertuples(index=False)
    }

    rows = []
    for user_index, user in enumerate(users):
        for date_index, date in enumerate(dates):
            if not active_mask[user_index, date_index, 0]:
                continue
            split = (
                "train"
                if date_index <= train_end
                else "validation"
                if date_index <= val_end
                else "test"
            )
            label = label_map.get((user, date.normalize()), {})
            rows.append(
                {
                    "user_id": user,
                    "date": date.strftime("%Y-%m-%d"),
                    "split": split,
                    "raw_error": float(raw_error[user_index, date_index]),
                    "anomaly_score": float(scores[user_index, date_index]),
                    "is_malicious_account_day": int(bool(label)),
                    "scenarios": label.get("scenarios", ""),
                    "insider_ids": label.get("insider_ids", ""),
                }
            )
    score_frame = pd.DataFrame(rows)
    score_frame.to_csv(SCORE_FILE, index=False, encoding="utf-8")

    # A minimal future-information test: perturb all observations after a
    # cutoff and confirm that earlier causal representations do not change.
    probe_index = test_start
    probe = values[[0]].copy()
    perturbed = probe.copy()
    perturbed[:, probe_index + 1 :, :] += 5.0
    probe_repr = np.asarray(
        model.encode(probe, mask="mask_last", **encode_args)
    )
    perturbed_repr = np.asarray(
        model.encode(perturbed, mask="mask_last", **encode_args)
    )
    causal_diff = float(
        np.max(
            np.abs(
                probe_repr[:, : probe_index + 1]
                - perturbed_repr[:, : probe_index + 1]
            )
        )
    )

    summary = {
        "model_file": str(MODEL_FILE),
        "score_file": str(SCORE_FILE),
        "input_shape": list(values.shape),
        "training_shape": list(train_values.shape),
        "representation_shape": list(plain.shape),
        "loss_first": loss_first,
        "loss_last": loss_last,
        "active_training_days": int(train_mask.sum()),
        "active_scored_rows": int(len(score_frame)),
        "score_split_counts": score_frame["split"].value_counts().to_dict(),
        "label_rows_in_scores": int(score_frame["is_malicious_account_day"].sum()),
        "causal_test_cutoff_index": probe_index,
        "causal_future_perturbation_max_diff": causal_diff,
        "labels_used_in_training": False,
        "labels_used_in_calibration": False,
        "model_config": {
            "output_dims": 64,
            "hidden_dims": 64,
            "depth": 5,
            "batch_size": 16,
            "epochs": 10,
            "sliding_padding": 30,
        },
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        PROCESSED / "r4.2_ts2vec_scores.npz",
        users=users,
        dates=dates.astype(str),
        raw_error=raw_error.astype(np.float32),
        anomaly_score=scores.astype(np.float32),
        active_mask=active_mask[:, :, 0],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
