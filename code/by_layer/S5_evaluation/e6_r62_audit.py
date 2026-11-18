from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e6"
    / "r62"
)
OUT.mkdir(parents=True, exist_ok=True)
K_GRID = (5, 10, 20)

sys.path.insert(0, str(ROOT))
from e0_e1_representation_preservation import (  # noqa: E402
    RAW_FEATURES,
    bank_scores,
    global_train_z,
)
from e3_multichannel_amplitude import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    aggregate_components,
    build_alerts,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
)
from train_r42_ts2vec import TS2Vec  # noqa: E402


PROCESSED = ROOT / "processed_r62"
DATASET = PROCESSED / "r6.2_sequence_dataset.npz"
MODEL = PROCESSED / "r6.2_ts2vec_model.pth"
RAW = PROCESSED / "r6.2_user_day_features.parquet"
LABELS = PROCESSED / "r6.2_malicious_account_days.csv"
INCIDENTS = PROCESSED / "r6.2_incidents.csv"


def align_raw(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    frame = pd.read_parquet(
        RAW,
        columns=["user_id", "date", *RAW_FEATURES],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    index = pd.MultiIndex.from_product(
        [users.tolist(), dates],
        names=["user_id", "date"],
    )
    aligned = frame.set_index(
        ["user_id", "date"]
    ).reindex(index)
    values = aligned[RAW_FEATURES].fillna(0.0).to_numpy(
        dtype=np.float64
    ).reshape(len(users), len(dates), len(RAW_FEATURES))
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)


def label_matrix(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    labels = pd.read_csv(LABELS, parse_dates=["date"])
    user_index = {
        user: index for index, user in enumerate(users)
    }
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    matrix = np.zeros((len(users), len(dates)), dtype=np.uint8)
    for row in labels.itertuples(index=False):
        user_key = user_index.get(str(row.user_id))
        date_key = date_index.get(
            pd.Timestamp(row.date).strftime("%Y-%m-%d")
        )
        if user_key is not None and date_key is not None:
            matrix[user_key, date_key] = 1
    return matrix


def point_metrics(
    signal: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
    start: int,
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    score = signal[:, start:]
    label = labels[:, start:]
    mask = active[:, start:] & np.isfinite(score)
    return {
        "pr_auc": float(
            average_precision_score(label[mask], score[mask])
        ),
        "roc_auc": float(
            roc_auc_score(label[mask], score[mask])
        ),
        "positive_days": int(label[mask].sum()),
    }


def main() -> None:
    dataset = np.load(DATASET, allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    test_start = int(dataset["test_start_index"])
    labels = label_matrix(users, dates)
    incidents = pd.read_csv(
        INCIDENTS,
        parse_dates=[
            "start",
            "end",
            "first_observed_date",
            "last_observed_date",
        ],
    )
    data = {
        "users": users,
        "dates": dates,
        "active_mask": active,
        "labels": labels,
        "incidents": incidents,
        "test_start": test_start,
    }

    raw = align_raw(users, dates)
    g_a1, q_a1 = aggregate_components(
        raw,
        active,
        train_end,
        user_specific=True,
        aggregation="channel_mean",
    )
    g_a1_z = global_train_z(g_a1, active, train_end)
    q_a1_z = global_train_z(q_a1, active, train_end)

    cache_path = OUT / "encoded_scores.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        masked_raw = cached["masked_raw"]
        cos_score = cached["cos_score"]
        euc_score = cached["euc_score"]
        n_clusters = int(cached["n_clusters"])
        print("Loaded cached r6.2 encoded scores", flush=True)
    else:
        device_id = 0 if torch.cuda.is_available() else "cpu"
        model = TS2Vec(
            input_dims=values.shape[-1],
            output_dims=64,
            hidden_dims=64,
            depth=5,
            device=device_id,
            batch_size=16,
        )
        load_device = (
            f"cuda:{device_id}"
            if isinstance(device_id, int)
            else "cpu"
        )
        model.net.load_state_dict(
            torch.load(
                MODEL,
                map_location=load_device,
                weights_only=True,
            )
        )
        encode_args = {
            "causal": True,
            "sliding_length": 1,
            "sliding_padding": 30,
            "batch_size": 32,
        }
        print("Encoding r6.2 masked representation", flush=True)
        masked = np.asarray(
            model.encode(values, mask="mask_last", **encode_args),
            dtype=np.float32,
        )
        print("Encoding r6.2 plain representation", flush=True)
        plain = np.asarray(
            model.encode(values, **encode_args),
            dtype=np.float32,
        )
        masked_raw = np.abs(masked - plain).sum(axis=-1)
        device = torch.device(load_device)
        print("Building r6.2 representation bank", flush=True)
        cos_score, euc_score, n_clusters = bank_scores(
            plain,
            active,
            train_end,
            device=device,
        )
        np.savez_compressed(
            cache_path,
            masked_raw=masked_raw,
            cos_score=cos_score,
            euc_score=euc_score,
            n_clusters=np.asarray(n_clusters),
        )
    masked_z = global_train_z(masked_raw, active, train_end)
    cos_z = global_train_z(cos_score, active, train_end)
    euc_z = global_train_z(euc_score, active, train_end)
    pai = euc_z + PAI_W_G * g_a1_z + PAI_W_Q * q_a1_z
    variants = {
        "original_ts2vec_global": masked_z,
        "bank_cosine_global": cos_z,
        "bank_euclidean_global": euc_z,
        "pai_a1_channel_mean": pai,
    }
    rows = []
    point = point_metrics(
        variants["pai_a1_channel_mean"],
        labels,
        active,
        test_start,
    )
    for name, signal in variants.items():
        rank_signal = signal.astype(np.float32, copy=True)
        rank_signal[~np.isfinite(rank_signal)] = -np.inf
        for k in K_GRID:
            alerts = build_alerts(rank_signal, active, k)
            result = evaluate_alert_system(
                data,
                alerts,
                name,
                k,
                external=True,
            )
            rows.append(
                {
                    "method": name,
                    "k": k,
                    **compact_metric(result, None),
                    **point_metrics(
                        signal,
                        labels,
                        active,
                        test_start,
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "metrics.csv",
        index=False,
        encoding="utf-8",
    )
    np.savez_compressed(
        OUT / "scores.npz",
        **variants,
    )
    summary = {
        "dataset": "r6.2",
        "seed": "existing_frozen_model",
        "bank_clusters": n_clusters,
        "train_end": train_end,
        "test_start": test_start,
        "events": len(incidents),
        "metrics": frame.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
