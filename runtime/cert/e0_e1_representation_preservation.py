from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "representation_preservation" / "e0_e1"
OUT.mkdir(parents=True, exist_ok=True)

PADDING_DAYS = 30
BANK_RATIO = 0.10
BANK_MAX = 500
BANK_TOP_K = 3
BANK_SEED = 2024
T2_WINDOW = 32
PAI_W_G = 0.6
PAI_W_Q = 0.4
K_GRID = (5, 10, 20)

sys.path.insert(0, str(ROOT))
from quota_relationship_sim import load_r42, load_r52  # noqa: E402
from train_r42_ts2vec import TS2Vec  # noqa: E402
import e17_multigranular_mask as e17  # noqa: E402


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

CHANNEL_FEATURES = {
    "logon": [
        "logon_count",
        "logoff_count",
        "after_hours_logon_count",
        "other_pc_logon_count",
    ],
    "device": [
        "usb_connect_count",
        "usb_disconnect_count",
    ],
    "file": [
        "file_copy_count",
        "archive_copy_count",
        "executable_copy_count",
        "unique_file_extensions",
    ],
    "email": [
        "email_count",
        "email_recipient_count",
        "external_recipient_count",
        "email_attachment_count",
        "email_size_bytes",
        "after_hours_email_count",
    ],
    "http": [
        "http_count",
        "after_hours_http_count",
        "external_http_count",
        "unique_http_domains",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E0/E1 baseline and frozen PAI scoring diagnostic."
    )
    parser.add_argument(
        "--dataset",
        choices=["r42", "r52"],
        default="r42",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20261116, 20261117, 20261118],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def dataset_config(name: str) -> dict:
    if name == "r42":
        return {
            "processed": ROOT / "processed_r42",
            "dataset": (
                ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
            ),
            "raw_features": (
                ROOT / "processed_r42" / "r4.2_user_day_features.csv"
            ),
            "loader": load_r42,
            "external": False,
        }
    return {
        "processed": ROOT / "processed_r52",
        "dataset": (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        ),
        "raw_features": (
            ROOT / "processed_r52" / "r5.2_user_day_features.csv"
        ),
        "loader": load_r52,
        "external": True,
    }


def align_raw_features(
    raw_path: Path,
    users: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    frame = pd.read_csv(
        raw_path,
        usecols=["user_id", "date", *RAW_FEATURES],
        parse_dates=["date"],
    )
    index = pd.MultiIndex.from_product(
        [users.tolist(), pd.to_datetime(dates)],
        names=["user_id", "date"],
    )
    aligned = frame.set_index(
        ["user_id", "date"]
    ).reindex(index)
    missing = aligned[RAW_FEATURES].isna()
    if missing.any().any():
        # Missing rows correspond to unavailable user-day records. Their
        # activity mask is false in the sequence dataset.
        aligned[RAW_FEATURES] = aligned[RAW_FEATURES].fillna(0.0)
    values = aligned[RAW_FEATURES].to_numpy(
        dtype=np.float64
    ).reshape(len(users), len(dates), len(RAW_FEATURES))
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)


def group_indices() -> list[np.ndarray]:
    name_to_index = {
        name: index for index, name in enumerate(RAW_FEATURES)
    }
    return [
        np.asarray(
            [name_to_index[name] for name in names],
            dtype=np.int64,
        )
        for names in CHANNEL_FEATURES.values()
    ]


def user_robust_statistics(
    raw: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    users, _, channels = raw.shape
    train_mask = active_mask[:, : train_end + 1]
    train_values = raw[:, : train_end + 1]
    global_values = train_values[train_mask]
    if len(global_values) == 0:
        raise RuntimeError("No active raw training rows.")
    global_median = np.median(global_values, axis=0)
    global_mad = np.median(
        np.abs(global_values - global_median),
        axis=0,
    )
    medians = np.tile(global_median, (users, 1))
    scales = np.tile(
        np.maximum(1.4826 * global_mad, 1e-6),
        (users, 1),
    )
    for user_index in range(users):
        rows = train_values[user_index][train_mask[user_index]]
        if len(rows) < 2:
            continue
        med = np.median(rows, axis=0)
        mad = np.median(np.abs(rows - med), axis=0)
        medians[user_index] = med
        scales[user_index] = np.maximum(
            1.4826 * mad,
            1e-6,
        )
    return (
        medians.astype(np.float32),
        scales.astype(np.float32),
    )


def causal_moving_mean(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    users, time_steps, channels = values.shape
    cumulative = np.zeros(
        (users, time_steps + 1, channels),
        dtype=np.float64,
    )
    cumulative[:, 1:, :] = np.cumsum(values, axis=1)
    output = np.empty(values.shape, dtype=np.float32)
    for time_index in range(time_steps):
        start = max(0, time_index - window + 1)
        count = time_index - start + 1
        output[:, time_index, :] = (
            cumulative[:, time_index + 1, :]
            - cumulative[:, start, :]
        ) / count
    return output


def amplitude_components(
    raw: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    medians, scales = user_robust_statistics(
        raw,
        active_mask,
        train_end,
    )
    point_deviation = np.abs(raw - medians[:, None, :])
    point_deviation /= scales[:, None, :]
    local_mean = causal_moving_mean(raw, T2_WINDOW)
    local_deviation = np.abs(local_mean - medians[:, None, :])
    local_deviation /= scales[:, None, :]

    groups = group_indices()
    g_channels = np.stack(
        [
            point_deviation[:, :, indices].max(axis=2)
            for indices in groups
        ],
        axis=2,
    )
    q_channels = np.stack(
        [
            local_deviation[:, :, indices].max(axis=2)
            for indices in groups
        ],
        axis=2,
    )
    ordered_g = np.sort(g_channels, axis=2)
    ordered_q = np.sort(q_channels, axis=2)
    G = ordered_g[:, :, -2:].mean(axis=2)
    Q = ordered_q[:, :, -2:].mean(axis=2)
    G[~active_mask] = np.nan
    Q[~active_mask] = np.nan
    return G.astype(np.float32), Q.astype(np.float32)


def encode_ts2vec(
    model: TS2Vec,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    args = {
        "causal": True,
        "sliding_length": 1,
        "sliding_padding": PADDING_DAYS,
        "batch_size": 32,
    }
    masked = np.asarray(
        model.encode(values, mask="mask_last", **args),
        dtype=np.float32,
    )
    plain = np.asarray(
        model.encode(values, **args),
        dtype=np.float32,
    )
    return masked, plain


def bank_scores(
    plain: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_repr = plain[:, : train_end + 1][
        active_mask[:, : train_end + 1]
    ]
    n_train = len(train_repr)
    n_clusters = min(
        BANK_MAX,
        max(1, int(round(BANK_RATIO * n_train))),
    )
    n_clusters = min(n_clusters, len(train_repr) - 1)
    print(
        f"  bank: train_points={n_train}, clusters={n_clusters}",
        flush=True,
    )
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=BANK_SEED,
        batch_size=max(8192, n_clusters),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    )
    kmeans.fit(train_repr)
    bank = torch.from_numpy(
        kmeans.cluster_centers_.astype(np.float32)
    ).to(device)
    bank_norm = torch.nn.functional.normalize(bank, dim=1)

    query_users, time_steps = active_mask.shape
    cosine_score = np.full(
        (query_users, time_steps),
        np.nan,
        dtype=np.float32,
    )
    euclidean_score = np.full_like(cosine_score, np.nan)
    active_users, active_times = np.nonzero(active_mask)
    flat_queries = plain[active_users, active_times].astype(np.float32)
    chunk = 4096
    out_cos = []
    out_euc = []
    with torch.inference_mode():
        bank_cpu = bank.cpu()
        bank_norm_cpu = bank_norm.cpu()
        for start in range(0, len(flat_queries), chunk):
            query = torch.from_numpy(
                flat_queries[start : start + chunk]
            ).to(device)
            query_norm = torch.nn.functional.normalize(
                query,
                dim=1,
            )
            cosine_distance = (
                1.0 - query_norm @ bank_norm.T
            ).clamp_min_(0.0)
            cos_top = cosine_distance.topk(
                BANK_TOP_K,
                dim=1,
                largest=False,
            ).values.mean(dim=1)
            euclidean_distance = torch.cdist(query, bank)
            euc_top = euclidean_distance.topk(
                BANK_TOP_K,
                dim=1,
                largest=False,
            ).values.mean(dim=1)
            out_cos.append(cos_top.cpu().numpy())
            out_euc.append(euc_top.cpu().numpy())
            del cosine_distance, euclidean_distance
    cosine_values = np.concatenate(out_cos)
    euclidean_values = np.concatenate(out_euc)
    cosine_score[active_users, active_times] = cosine_values
    euclidean_score[active_users, active_times] = euclidean_values
    return cosine_score, euclidean_score, n_clusters


def global_train_z(
    scores: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    train_mask = active_mask[:, : train_end + 1] & np.isfinite(
        scores[:, : train_end + 1]
    )
    values = scores[:, : train_end + 1][train_mask]
    if len(values) == 0:
        raise RuntimeError("No finite global calibration scores.")
    mean = float(values.mean())
    std = max(float(values.std()), 1e-8)
    return ((scores - mean) / std).astype(np.float32)


def global_robust(
    scores: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    train_mask = active_mask[:, : train_end + 1] & np.isfinite(
        scores[:, : train_end + 1]
    )
    values = scores[:, : train_end + 1][train_mask]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = max(float(values.std()), 1e-8)
    return np.clip(
        (scores - median) / scale,
        0.0,
        None,
    ).astype(np.float32)


def user_tail_evidence(
    scores: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    output = np.full(scores.shape, np.nan, dtype=np.float32)
    for user_index in range(scores.shape[0]):
        train_mask = active_mask[
            user_index, : train_end + 1
        ] & np.isfinite(scores[user_index, : train_end + 1])
        train_values = scores[
            user_index, : train_end + 1
        ][train_mask]
        if len(train_values) == 0:
            continue
        ordered = np.sort(train_values)
        n_values = len(ordered)
        finite = active_mask[user_index] & np.isfinite(
            scores[user_index]
        )
        values = scores[user_index, finite]
        rank = np.searchsorted(ordered, values, side="right")
        tail = (n_values - rank + 1) / (n_values + 1)
        output[user_index, finite] = np.maximum(
            0.0,
            -np.log(tail),
        )
    return output


def zscore_aligned(*arrays: np.ndarray) -> list[np.ndarray]:
    result = []
    for array in arrays:
        finite = np.isfinite(array)
        mean = float(array[finite].mean())
        std = max(float(array[finite].std()), 1e-8)
        result.append((array - mean) / std)
    return result


def rankable(array: np.ndarray) -> np.ndarray:
    result = array.astype(np.float32, copy=True)
    result[~np.isfinite(result)] = -np.inf
    return result


def evaluate_variants(
    data: dict,
    *,
    external: bool,
    active_mask: np.ndarray,
    train_end: int,
    masked_raw: np.ndarray,
    cos_score: np.ndarray,
    euc_score: np.ndarray,
    G: np.ndarray,
    Q: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    robust = global_robust(masked_raw, active_mask, train_end)
    masked_user = user_tail_evidence(
        masked_raw,
        active_mask,
        train_end,
    )
    masked_global_z = global_train_z(
        masked_raw,
        active_mask,
        train_end,
    )

    cos_global_z = global_train_z(cos_score, active_mask, train_end)
    euc_global_z = global_train_z(euc_score, active_mask, train_end)
    g_global_z = global_train_z(G, active_mask, train_end)
    q_global_z = global_train_z(Q, active_mask, train_end)

    cos_user = user_tail_evidence(cos_score, active_mask, train_end)
    euc_user = user_tail_evidence(euc_score, active_mask, train_end)
    g_user = user_tail_evidence(G, active_mask, train_end)
    q_user = user_tail_evidence(Q, active_mask, train_end)
    cos_user_z = global_train_z(cos_user, active_mask, train_end)
    euc_user_z = global_train_z(euc_user, active_mask, train_end)
    g_user_z = global_train_z(g_user, active_mask, train_end)
    q_user_z = global_train_z(q_user, active_mask, train_end)

    variants = {
        "S0_mask_raw": masked_raw,
        "S0_mask_global_robust": robust,
        "S0_mask_global_z": masked_global_z,
        "S0_mask_user": masked_user,
        "S1_bank_cos_global": cos_global_z,
        "S1_bank_cos_user": cos_user_z,
        "S2_bank_euc_global": euc_global_z,
        "S2_bank_euc_user": euc_user_z,
        "S3_pai_mag_global": (
            euc_global_z + PAI_W_G * g_global_z
        ),
        "S3_pai_mag_user": (
            euc_user_z + PAI_W_G * g_user_z
        ),
        "S4_pai_t2_global": (
            euc_global_z + PAI_W_Q * q_global_z
        ),
        "S4_pai_t2_user": (
            euc_user_z + PAI_W_Q * q_user_z
        ),
        "S5_pai_full_global": (
            euc_global_z
            + PAI_W_G * g_global_z
            + PAI_W_Q * q_global_z
        ),
        "S5_pai_full_user": (
            euc_user_z
            + PAI_W_G * g_user_z
            + PAI_W_Q * q_user_z
        ),
        "S6_amp_only_global": (
            PAI_W_G * g_global_z
            + PAI_W_Q * q_global_z
        ),
        "S6_amp_only_user": (
            PAI_W_G * g_user_z
            + PAI_W_Q * q_user_z
        ),
    }

    rows = []
    for method, signal in variants.items():
        signal = rankable(signal)
        for k in K_GRID:
            e17.K = k
            metrics = e17.evaluate_signal(
                data,
                signal,
                None,
                method,
                external=external,
                use_iforest=False,
            )
            split = "external_test" if external else "test"
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "k": k,
                    **metrics[split],
                }
            )
            if not external:
                rows.append(
                    {
                        "method": method,
                        "split": "validation",
                        "k": k,
                        **metrics["validation"],
                    }
                )

    arrays = {
        "masked_raw": masked_raw,
        "cos_score": cos_score,
        "euc_score": euc_score,
        "G": G,
        "Q": Q,
    }
    diagnostics = {
        "corr_cos_euc": float(
            np.corrcoef(
                cos_score[np.isfinite(cos_score)],
                euc_score[np.isfinite(euc_score)],
            )[0, 1]
        )
    }
    return pd.DataFrame(rows), {"arrays": arrays, "diagnostics": diagnostics}


def main() -> None:
    args = parse_args()
    config = dataset_config(args.dataset)
    dataset = np.load(config["dataset"], allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    data = config["loader"]()
    raw = align_raw_features(
        config["raw_features"],
        users,
        dates,
    )
    G, Q = amplitude_components(
        raw,
        active_mask,
        train_end,
    )
    train_values = values[:, : train_end + 1, :]
    device = 0 if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(
        f"cuda:{device}" if isinstance(device, int) else "cpu"
    )

    combined_metrics = []
    for seed in args.seeds:
        start_time = time.time()
        seed_dir = OUT / args.dataset / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model_path = seed_dir / "ts2vec_model.pth"
        model = TS2Vec(
            input_dims=values.shape[-1],
            output_dims=64,
            hidden_dims=64,
            depth=5,
            device=device,
            batch_size=16,
        )
        if model_path.exists() and not args.force_train:
            print(f"Loading {model_path}", flush=True)
            model.load(str(model_path))
            losses = None
        else:
            print(f"Training T0 seed={seed}", flush=True)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            losses = model.fit(
                train_values,
                n_epochs=args.epochs,
                verbose=False,
            )
            model.save(str(model_path))
        masked, plain = encode_ts2vec(model, values)
        masked_raw = np.abs(masked - plain).sum(axis=-1)
        cos_score, euc_score, n_clusters = bank_scores(
            plain,
            active_mask,
            train_end,
            device=torch_device,
        )
        metrics, artifacts = evaluate_variants(
            data,
            external=bool(config["external"]),
            active_mask=active_mask,
            train_end=train_end,
            masked_raw=masked_raw,
            cos_score=cos_score,
            euc_score=euc_score,
            G=G,
            Q=Q,
        )
        metrics.insert(0, "seed", seed)
        combined_metrics.append(metrics)
        metrics.to_csv(
            seed_dir / "metrics.csv",
            index=False,
            encoding="utf-8",
        )
        np.savez_compressed(
            seed_dir / "scores.npz",
            **artifacts["arrays"],
        )
        summary = {
            "dataset": args.dataset,
            "seed": seed,
            "train_end": train_end,
            "epochs": args.epochs,
            "losses": losses,
            "bank_clusters": n_clusters,
            "bank_top_k": BANK_TOP_K,
            "t2_window": T2_WINDOW,
            "pai_weights": [1.0, PAI_W_G, PAI_W_Q],
            "diagnostics": artifacts["diagnostics"],
            "elapsed_seconds": time.time() - start_time,
        }
        (seed_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"seed={seed} done in {time.time() - start_time:.1f}s "
            f"bank_clusters={n_clusters}",
            flush=True,
        )
        del model, masked, plain
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_metrics = pd.concat(combined_metrics, ignore_index=True)
    all_metrics.to_csv(
        OUT / args.dataset / "seed_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    summary_rows = (
        all_metrics.groupby(["method", "split", "k"], as_index=False)
        .agg(
            event_recall_3d_mean=("event_recall_3d", "mean"),
            event_recall_3d_std=("event_recall_3d", "std"),
            precision_at_k_mean=("precision_at_k", "mean"),
            precision_at_k_std=("precision_at_k", "std"),
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_std=("incident_hits", "std"),
            false_alerts_per_day_mean=(
                "false_alerts_per_day",
                "mean",
            ),
        )
        .sort_values(["split", "k", "method"])
    )
    summary_rows.to_csv(
        OUT / args.dataset / "seed_metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    print(summary_rows.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
