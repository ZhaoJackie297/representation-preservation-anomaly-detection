"""Exp S6 Tier-2 paper hardening experiments.

Commands:
  python research/ccf-b/supplement/exp_s6_tier2.py s6a
  python research/ccf-b/supplement/exp_s6_tier2.py s6b
  python research/ccf-b/supplement/exp_s6_tier2.py s6c

All outputs are written below research/ccf-b/supplement_results/exp_s6_tier2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_DIR = REPO_ROOT / "runtime" / "cert"
PHASE0_DIR = REPO_ROOT / "research" / "ccf-b" / "phase0"
SUPPLEMENT_DIR = REPO_ROOT / "research" / "ccf-b" / "supplement"
sys.path.insert(0, str(CERT_DIR))
sys.path.insert(0, str(PHASE0_DIR))
sys.path.insert(0, str(SUPPLEMENT_DIR))

from coverage_soft_ts2vec import CoverageSoftTS2Vec  # noqa: E402
from e0_e1_representation_preservation import (  # noqa: E402
    RAW_FEATURES,
    global_train_z,
)
from e3_multichannel_amplitude import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    aggregate_components,
    build_alerts,
    point_metrics,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
)
from exp_s5a_norm_intervention import (  # noqa: E402
    build_random_batch_table,
    patched_contrastive_loss,
)


RESULT_ROOT = REPO_ROOT / "research" / "ccf-b" / "supplement_results"
OUT_ROOT = RESULT_ROOT / "exp_s6_tier2"
R62_ROOT = Path(r"E:\MarketRisker\work\cert\processed_r62")
CLUE_DATA = Path(
    r"E:\MarketRisker\work\cert\external\clue_lds"
    r"\processed_final\clue_user_day_dataset.npz"
)
R62_DATASET = R62_ROOT / "r6.2_sequence_dataset.npz"
R62_RAW = R62_ROOT / "r6.2_user_day_features.parquet"
R62_LABELS = R62_ROOT / "r6.2_malicious_account_days.csv"
R62_INCIDENTS = R62_ROOT / "r6.2_incidents.csv"
SEEDS_R62 = [20261116, 20261117, 20261118]
SEEDS_R52 = [20261116, 20261117, 20261118]
SEEDS_SMD = [20261201, 20261202, 20261203]
K_GRID = (5, 10, 20)
COOLDOWN_DAYS = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp S6 tier-2 hardening")
    parser.add_argument(
        "command",
        choices=["s6a", "s6b", "s6c", "s6c-summary"],
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_device() -> tuple[int | str, torch.device]:
    device_id = 0 if torch.cuda.is_available() else "cpu"
    return device_id, torch.device(
        f"cuda:{device_id}" if isinstance(device_id, int) else "cpu"
    )


def load_encoder(path: Path, input_dims: int) -> tuple[CoverageSoftTS2Vec, torch.device]:
    device_id, device = resolve_device()
    model = CoverageSoftTS2Vec(
        input_dims=input_dims,
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device_id,
        lr=1e-3,
        batch_size=16,
        lambda_=0.5,
        tau_temp=1.5,
        soft_instance=False,
        soft_temporal=False,
    )
    state = torch.load(path, map_location=device, weights_only=True)
    if any(key.startswith("module.") for key in state):
        model.net.load_state_dict(state)
    else:
        model._net.load_state_dict(state)
    model.net.eval()
    return model, device


def encode_pair(
    model: CoverageSoftTS2Vec,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    kwargs = {
        "causal": True,
        "sliding_length": 1,
        "sliding_padding": 30,
        "batch_size": 256,
    }
    plain = np.asarray(model.encode(values, **kwargs), dtype=np.float32)
    masked = np.asarray(
        model.encode(values, mask="mask_last", **kwargs),
        dtype=np.float32,
    )
    return plain, masked


def bank_scores(
    train_repr: np.ndarray,
    all_repr: np.ndarray,
    active: np.ndarray,
    train_end: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    train = train_repr[:, : train_end + 1][active[:, : train_end + 1]]
    clusters = min(500, max(1, int(round(0.1 * len(train)))))
    clusters = min(clusters, len(train) - 1)
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        init="k-means++",
        random_state=2024,
        batch_size=max(8192, clusters),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    ).fit(train)
    bank = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
    bank_norm = torch.nn.functional.normalize(bank, dim=1)
    users, times = np.nonzero(active)
    query_values = all_repr[users, times]
    cosine_parts = []
    euclidean_parts = []
    with torch.inference_mode():
        for start in range(0, len(query_values), 4096):
            query = torch.from_numpy(
                query_values[start : start + 4096].astype(np.float32)
            ).to(device)
            query_norm = torch.nn.functional.normalize(query, dim=1)
            cosine = (1.0 - query_norm @ bank_norm.T).clamp_min_(0.0)
            euclidean = torch.cdist(query, bank)
            cosine_parts.append(
                cosine.topk(3, dim=1, largest=False)
                .values.mean(dim=1)
                .cpu()
                .numpy()
            )
            euclidean_parts.append(
                euclidean.topk(3, dim=1, largest=False)
                .values.mean(dim=1)
                .cpu()
                .numpy()
            )
    cosine_score = np.full(active.shape, np.nan, dtype=np.float32)
    euclidean_score = np.full(active.shape, np.nan, dtype=np.float32)
    cosine_score[users, times] = np.concatenate(cosine_parts)
    euclidean_score[users, times] = np.concatenate(euclidean_parts)
    return cosine_score, euclidean_score, clusters


def load_r62_bundle() -> dict:
    with np.load(R62_DATASET, allow_pickle=False) as dataset:
        values = dataset["values"].astype(np.float32)
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
        test_start = int(dataset["test_start_index"])
    labels = np.zeros(active.shape, dtype=np.uint8)
    label_frame = pd.read_csv(R62_LABELS, parse_dates=["date"])
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)}
    for row in label_frame.itertuples(index=False):
        user = user_index.get(str(row.user_id))
        time = date_index.get(pd.Timestamp(row.date).strftime("%Y-%m-%d"))
        if user is not None and time is not None:
            labels[user, time] = 1
    incidents = pd.read_csv(
        R62_INCIDENTS,
        parse_dates=[
            "start",
            "end",
            "first_observed_date",
            "last_observed_date",
        ],
    )
    return {
        "values": values,
        "users": users,
        "dates": dates,
        "active": active,
        "labels": labels,
        "incidents": incidents,
        "train_end": train_end,
        "test_start": test_start,
    }


def align_r62_raw(users: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    frame = pd.read_parquet(
        R62_RAW,
        columns=["user_id", "date", *RAW_FEATURES],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    index = pd.MultiIndex.from_product(
        [users.tolist(), dates],
        names=["user_id", "date"],
    )
    aligned = frame.set_index(["user_id", "date"]).reindex(index)
    values = aligned[RAW_FEATURES].fillna(0.0).to_numpy(
        dtype=np.float64
    ).reshape(len(users), len(dates), len(RAW_FEATURES))
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)


def r62_data_dict(bundle: dict) -> dict:
    return {
        "users": bundle["users"],
        "dates": bundle["dates"],
        "active_mask": bundle["active"],
        "labels": bundle["labels"],
        "incidents": bundle["incidents"],
        "test_start": bundle["test_start"],
    }


def evaluate_scores(
    scores: dict[str, np.ndarray],
    data: dict,
    active: np.ndarray,
    test_start: int,
    seed: int,
) -> list[dict]:
    rows = []
    end = len(data["dates"]) - 1
    for name, signal in scores.items():
        point = point_metrics(signal, data, active, test_start, end)
        rank = signal.astype(np.float32, copy=True)
        rank[~np.isfinite(rank)] = -np.inf
        for k in K_GRID:
            alerts = build_alerts(rank, active, k)
            result = evaluate_alert_system(
                data,
                alerts,
                name,
                k,
                external=True,
            )
            rows.append(
                {
                    "seed": seed,
                    "method": name,
                    "split": "test",
                    "k": k,
                    **compact_metric(result, None),
                    **point,
                }
            )
    return rows


def summarize_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            event_recall_3d_std=("event_recall_3d", "std"),
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_std=("incident_hits", "std"),
        )
        .sort_values(group_columns)
    )


def run_s6a(force: bool) -> None:
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    by_seed_path = out_dir / "s6a_r62_pai_by_seed.csv"
    summary_path = out_dir / "s6a_r62_pai_summary.csv"
    if by_seed_path.exists() and summary_path.exists() and not force:
        print("[SKIP] S6a outputs already exist", flush=True)
        return

    bundle = load_r62_bundle()
    values = bundle["values"]
    active = bundle["active"]
    data = r62_data_dict(bundle)
    device_id, device = resolve_device()
    raw = align_r62_raw(bundle["users"], bundle["dates"])
    g_raw, q_raw = aggregate_components(
        raw,
        active,
        bundle["train_end"],
        user_specific=True,
        aggregation="channel_mean",
    )
    g_z = global_train_z(g_raw, active, bundle["train_end"])
    q_z = global_train_z(q_raw, active, bundle["train_end"])
    rows: list[dict] = []

    for arm in ("Hard-Rand-10ep", "SoftInst-Rand-10ep"):
        for seed in SEEDS_R62:
            checkpoint = (
                RESULT_ROOT
                / "exp_s5d_unify_clue_r62"
                / "r62_3seed"
                / arm
                / f"seed_{seed}"
                / "model.pth"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            start_time = time.time()
            model, _ = load_encoder(checkpoint, values.shape[-1])
            plain, masked = encode_pair(model, values)
            masked_raw = np.abs(masked - plain).sum(axis=-1)
            _, euclidean_raw, clusters = bank_scores(
                plain, plain, active, bundle["train_end"], device
            )
            euclidean_z = global_train_z(euclidean_raw, active, bundle["train_end"])
            pai_mean = (
                euclidean_z
                + 0.5 * g_z
                + 0.5 * q_z
            )
            pai_official = 0.6 * euclidean_z + 0.4 * g_z + 0.2 * q_z
            arm_rows = evaluate_scores(
                {
                    f"{arm} + PAI-Mean": pai_mean,
                    f"{arm} + PAI-Official": pai_official,
                },
                data,
                active,
                bundle["test_start"],
                seed,
            )
            for row in arm_rows:
                row["bank_clusters"] = clusters
                row["elapsed_sec"] = round(time.time() - start_time, 2)
            rows.extend(arm_rows)
            print(
                f"[DONE] S6a {arm} seed={seed} "
                f"time={time.time() - start_time:.1f}s",
                flush=True,
            )
            del model, plain, masked
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(by_seed_path, index=False, encoding="utf-8")
    summary = summarize_metrics(frame, ["method", "split", "k"])
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(summary[summary["k"] == 10].to_string(index=False), flush=True)


def training_modality_metrics(
    reprs: np.ndarray,
    input_values: np.ndarray,
    active: np.ndarray,
    train_end: int,
    device: torch.device,
) -> dict[str, float | int]:
    train_repr = reprs[:, : train_end + 1][active[:, : train_end + 1]]
    train_input = input_values[:, : train_end + 1][active[:, : train_end + 1]]
    norm = np.linalg.norm(train_repr, axis=-1)
    input_norm = np.linalg.norm(train_input, axis=-1)
    rho = float(spearmanr(norm, input_norm)[0])
    cv = float(norm.std() / max(norm.mean(), 1e-8))
    clusters = min(500, max(1, int(round(0.1 * len(train_repr)))))
    clusters = min(clusters, len(train_repr) - 1)
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        init="k-means++",
        random_state=2024,
        batch_size=max(8192, clusters),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    ).fit(train_repr)
    bank = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
    bank_norm = torch.nn.functional.normalize(bank, dim=1)
    bank_r = torch.linalg.norm(bank, dim=1)
    query = torch.from_numpy(train_repr.astype(np.float32)).to(device)
    query_r = torch.linalg.norm(query, dim=1)
    query_norm = torch.nn.functional.normalize(query, dim=1)
    with torch.inference_mode():
        distance = torch.cdist(query, bank)
        _, top_idx = distance.topk(3, dim=1, largest=False)
        nearest_r = bank_r[top_idx]
        cosine = (1.0 - query_norm @ bank_norm.T).clamp_min_(0.0)
        selected_cosine = torch.gather(cosine, 1, top_idx)
        radial_sq = (query_r.unsqueeze(1) - nearest_r) ** 2
        angular = 2.0 * query_r.unsqueeze(1) * nearest_r * selected_cosine
        share = radial_sq.mean(dim=1) / (
            (radial_sq + angular).mean(dim=1).clamp_min(1e-12)
        )
    predicted = (
        "bank_euclidean" if rho > 0.50 and cv > 0.50 else "bank_cosine"
    )
    return {
        "train_norm_mean": float(norm.mean()),
        "train_norm_cv": cv,
        "train_spearman_norm_input": rho,
        "train_radial_share": float(share.cpu().numpy().mean()),
        "train_active_cells": int(len(train_repr)),
        "predicted_metric": predicted,
        "bank_clusters": clusters,
    }


def expected_metric(dataset: str, arm: str) -> str:
    if dataset in ("r4.2", "r5.2"):
        return "bank_cosine"
    return "bank_euclidean"


def run_s6b(force: bool) -> None:
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    by_seed_path = out_dir / "s6b_zero_label_modality_by_seed.csv"
    summary_path = out_dir / "s6b_zero_label_modality_summary.csv"
    if by_seed_path.exists() and summary_path.exists() and not force:
        print("[SKIP] S6b outputs already exist", flush=True)
        return
    _, device = resolve_device()
    rows: list[dict] = []

    dataset_specs = [
        (
            "r4.2",
            CERT_DIR / "processed_r42" / "r4.2_sequence_dataset.npz",
            {
                "Hard-Rand-10ep": RESULT_ROOT
                / "exp_s5a_norm_intervention"
                / "Hard-Rand",
                "SoftInst-Rand-10ep": RESULT_ROOT
                / "exp_s5a_norm_intervention"
                / "SoftInst-Rand",
            },
            SEEDS_R62,
        ),
        (
            "r5.2",
            CERT_DIR / "processed_r52" / "r5.2_sequence_dataset.npz",
            {
                "Hard-Rand-10ep": RESULT_ROOT
                / "exp_s5b_symmetry_audit"
                / "r52_hard_full"
                / "Hard-Rand-10ep",
                "SoftInst-Rand-10ep": RESULT_ROOT
                / "exp_s1_r5.2"
                / "SoftInst-Rand-10ep",
            },
            SEEDS_R62,
        ),
        (
            "r6.2",
            R62_DATASET,
            {
                arm: RESULT_ROOT
                / "exp_s5d_unify_clue_r62"
                / "r62_3seed"
                / arm
                for arm in ("Hard-Rand-10ep", "SoftInst-Rand-10ep")
            },
            SEEDS_R62,
        ),
        (
            "CLUE-LDS",
            CLUE_DATA,
            {
                arm: RESULT_ROOT
                / "exp_s5d_unify_clue_r62"
                / "clue_10seed"
                / arm
                for arm in ("Hard-Rand-10ep", "SoftInst-Rand-10ep")
            },
            SEEDS_R62,
        ),
    ]

    for dataset, data_path, arm_dirs, seeds in dataset_specs:
        with np.load(data_path, allow_pickle=False) as dataset_file:
            values = dataset_file["values"].astype(np.float32)
            active = dataset_file["active_mask"].astype(bool)
            if active.ndim == 3:
                active = active[:, :, 0]
            train_end = int(dataset_file["train_end_index"])
        input_values = (
            values[:, :, :20]
            if dataset in ("r4.2", "r5.2", "r6.2")
            else values
        )
        for arm, arm_dir in arm_dirs.items():
            for seed in seeds:
                checkpoint = arm_dir / f"seed_{seed}" / "model.pth"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                start_time = time.time()
                model, _ = load_encoder(checkpoint, values.shape[-1])
                reprs, _ = encode_pair(model, values[:, : train_end + 1])
                metrics = training_modality_metrics(
                    reprs,
                    input_values[:, : train_end + 1],
                    active[:, : train_end + 1],
                    train_end,
                    device,
                )
                expected = expected_metric(dataset, arm)
                rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "seed": seed,
                        **metrics,
                        "expected_metric": expected,
                        "route_correct": metrics["predicted_metric"] == expected,
                        "metric_source": "training_active_rows_zero_label",
                        "zero_label_valid": True,
                        "elapsed_sec": round(time.time() - start_time, 2),
                    }
                )
                print(
                    f"[DONE] S6b {dataset} {arm} seed={seed} "
                    f"rho={metrics['train_spearman_norm_input']:.3f} "
                    f"cv={metrics['train_norm_cv']:.3f} "
                    f"route={metrics['predicted_metric']}",
                    flush=True,
                )
                del model, reprs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    smd_path = (
        RESULT_ROOT
        / "exp_s5c_norm_corr_3seed"
        / "smd_3seed"
        / "smd_norm_radial_all_machines_3seed.csv"
    )
    if not smd_path.is_file():
        raise FileNotFoundError(smd_path)
    smd = pd.read_csv(smd_path)
    for row in smd.itertuples(index=False):
        if row.variant not in ("T0_hard", "T3_inst_soft"):
            continue
        predicted = (
            "bank_euclidean"
            if row.spearman_norm_vs_xnorm > 0.50 and row.norm_cv_normal > 0.50
            else "bank_cosine"
        )
        rows.append(
            {
                "dataset": "SMD",
                "arm": "T0_hard" if row.variant == "T0_hard" else "T3_inst_soft",
                "seed": int(row.seed),
                "machine": row.machine,
                "train_norm_mean": row.norm_mean_normal,
                "train_norm_cv": row.norm_cv_normal,
                "train_spearman_norm_input": row.spearman_norm_vs_xnorm,
                "train_radial_share": row.radial_energy_share_normal,
                "train_active_cells": np.nan,
                "predicted_metric": predicted,
                "expected_metric": "bank_euclidean",
                "route_correct": predicted == "bank_euclidean",
                "metric_source": "test_normal_segment_existing_s5c_csv",
                "zero_label_valid": False,
                "elapsed_sec": np.nan,
                "bank_clusters": np.nan,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(by_seed_path, index=False, encoding="utf-8")
    summary = (
        frame.groupby(["dataset", "arm"], as_index=False)
        .agg(
            n_rows=("seed", "size"),
            n_seeds=("seed", "nunique"),
            train_norm_mean=("train_norm_mean", "mean"),
            train_norm_cv=("train_norm_cv", "mean"),
            train_spearman_norm_input=("train_spearman_norm_input", "mean"),
            train_radial_share=("train_radial_share", "mean"),
            radial_route_fraction=("predicted_metric", lambda values: float(np.mean(values == "bank_euclidean"))),
            route_accuracy=("route_correct", "mean"),
            zero_label_valid=("zero_label_valid", "first"),
            metric_source=("metric_source", "first"),
        )
        .sort_values(["dataset", "arm"])
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def test_geometry(
    reprs: np.ndarray,
    input_values: np.ndarray,
    active: np.ndarray,
    labels: np.ndarray,
    train_end: int,
    device: torch.device,
) -> dict[str, float]:
    active_test = active[:, train_end + 1 :]
    query = reprs[:, train_end + 1 :][active_test]
    norms = np.linalg.norm(query, axis=-1)
    centered = query.astype(np.float64) - query.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    probabilities = eigenvalues / eigenvalues.sum()
    effective_rank = float(
        np.exp(-np.sum(probabilities * np.log(probabilities)))
    )
    train_repr = reprs[:, : train_end + 1][active[:, : train_end + 1]]
    clusters = min(500, max(1, int(round(0.1 * len(train_repr)))))
    clusters = min(clusters, len(train_repr) - 1)
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        init="k-means++",
        random_state=2024,
        batch_size=max(8192, clusters),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    ).fit(train_repr)
    bank = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
    bank_r = torch.linalg.norm(bank, dim=1)
    bank_norm = torch.nn.functional.normalize(bank, dim=1)
    query_tensor = torch.from_numpy(query.astype(np.float32)).to(device)
    query_r = torch.linalg.norm(query_tensor, dim=1)
    query_norm = torch.nn.functional.normalize(query_tensor, dim=1)
    with torch.inference_mode():
        distance = torch.cdist(query_tensor, bank)
        _, top_idx = distance.topk(3, dim=1, largest=False)
        nearest_r = bank_r[top_idx]
        cosine = (1.0 - query_norm @ bank_norm.T).clamp_min_(0.0)
        selected_cosine = torch.gather(cosine, 1, top_idx)
        radial_sq = (query_r.unsqueeze(1) - nearest_r) ** 2
        angular = 2.0 * query_r.unsqueeze(1) * nearest_r * selected_cosine
        share = (
            radial_sq.mean(dim=1)
            / (radial_sq + angular).mean(dim=1).clamp_min(1e-12)
        ).cpu().numpy()
    y_test = labels[:, train_end + 1 :][active_test]
    return {
        "norm_mean": float(norms.mean()),
        "norm_cv": float(norms.std() / max(norms.mean(), 1e-8)),
        "effective_rank": effective_rank,
        "radial_energy_share_all": float(share.mean()),
        "radial_energy_share_pos": float(share[y_test == 1].mean()),
        "test_active_cells": int(len(query)),
        "positive_cells": int((y_test == 1).sum()),
    }


def _canonical_scorer(name: str) -> str:
    mapping = {
        "plain_l1": "Mask L1",
        "bank_cosine": "Cosine",
        "bank_euclidean": "Euclidean",
        "pai_a1": "PAI-Mean",
        "pai_official": "PAI-Official",
    }
    return mapping[name]


def existing_control_rows(
    geometry_lookup: dict[tuple[str, int], dict[str, float]],
) -> list[dict]:
    control_specs = [
        (
            "Hard-Rand-10ep",
            RESULT_ROOT
            / "exp_s5b_symmetry_audit"
            / "r52_hard_full"
            / "metrics_by_seed.csv",
            "Hard-Rand-10ep__",
        ),
        (
            "SoftInst-Rand-10ep",
            RESULT_ROOT
            / "exp_s1_r5.2"
            / "metrics_by_seed.csv",
            "SoftInst-Rand-10ep__",
        ),
    ]
    rows = []
    for arm, path, prefix in control_specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame = frame[
            frame["seed"].isin(SEEDS_R52)
            & (frame["split"] == "test")
            & frame["method"].str.startswith(prefix)
        ].copy()
        for row in frame.itertuples(index=False):
            suffix = row.method.removeprefix(prefix)
            rows.append(
                {
                    "arm": arm,
                    "scorer": _canonical_scorer(suffix),
                    "seed": int(row.seed),
                    "k": int(row.k),
                    "split": "test",
                    "pr_auc": row.pr_auc,
                    "roc_auc": row.roc_auc,
                    "incident_hits": row.incident_hits,
                    "incident_total": row.incident_total,
                    "event_recall_3d": row.event_recall_3d,
                    **geometry_lookup.get((arm, int(row.seed)), {}),
                    "source": "existing_control_archive",
                }
            )
    return rows


def write_s6c_summary(
    by_seed_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(by_seed_path)
    if "split" in frame:
        frame["split"] = frame["split"].fillna("test")
    summary = summarize_metrics(frame, ["arm", "scorer", "split", "k"])
    geometry_columns = [
        "norm_mean",
        "norm_cv",
        "effective_rank",
        "radial_energy_share_all",
        "radial_energy_share_pos",
    ]
    geometry_summary = (
        frame.groupby(["arm", "seed"], as_index=False)[geometry_columns]
        .first()
        .groupby("arm", as_index=False)[geometry_columns]
        .agg(["mean", "std"])
    )
    geometry_summary.columns = [
        "_".join(value for value in column if value)
        if isinstance(column, tuple)
        else column
        for column in geometry_summary.columns
    ]
    summary = summary.merge(geometry_summary, on="arm", how="left")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    return summary


def run_s6c(force: bool) -> None:
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    by_seed_path = out_dir / "s6c_r52_l2reg_by_seed.csv"
    summary_path = out_dir / "s6c_r52_l2reg_summary.csv"
    model_root = out_dir / "s6c_models"
    model_root.mkdir(parents=True, exist_ok=True)
    if by_seed_path.exists() and summary_path.exists() and not force:
        print("[SKIP] S6c outputs already exist", flush=True)
        return

    data_path = CERT_DIR / "processed_r52" / "r5.2_sequence_dataset.npz"
    raw_csv = CERT_DIR / "processed_r52" / "r5.2_user_day_features.csv"
    bundle = load_r52()
    with np.load(data_path, allow_pickle=False) as dataset:
        values = dataset["values"].astype(np.float32)
        users = dataset["users"].astype(str)
        dates = dataset["dates"].astype(str)
        active = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
    train_values = values[:, : train_end + 1]
    from e0_e1_representation_preservation import align_raw_features

    raw_20 = align_raw_features(raw_csv, users, dates)
    g_raw, q_raw = aggregate_components(
        raw_20,
        active,
        train_end,
        user_specific=True,
        aggregation="channel_mean",
    )
    g_z = global_train_z(g_raw, active, train_end)
    q_z = global_train_z(q_raw, active, train_end)
    data = bundle
    device_id, device = resolve_device()
    rows: list[dict] = []
    geometry_lookup: dict[tuple[str, int], dict[str, float]] = {}

    # Existing zero-retrain controls: scores are reused; representations are
    # loaded only to compute the same geometry diagnostics as the new arms.
    control_paths = {
        "Hard-Rand-10ep": (
            RESULT_ROOT
            / "exp_s5b_symmetry_audit"
            / "r52_hard_full"
            / "Hard-Rand-10ep"
        ),
        "SoftInst-Rand-10ep": (
            RESULT_ROOT / "exp_s1_r5.2" / "SoftInst-Rand-10ep"
        ),
    }
    for arm, arm_dir in control_paths.items():
        for seed in SEEDS_R52:
            checkpoint = arm_dir / f"seed_{seed}" / "model.pth"
            model, _ = load_encoder(checkpoint, values.shape[-1])
            plain, _ = encode_pair(model, values)
            geometry_lookup[(arm, seed)] = test_geometry(
                plain,
                values,
                active,
                bundle["labels"],
                train_end,
                device,
            )
            del model, plain
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rows.extend(existing_control_rows(geometry_lookup))

    n_steps = (values.shape[0] // 16) * 10
    for beta in (0.02, 0.20, 1.00):
        arm = f"Hard-L2Reg-b{beta:.2f}"
        for seed in SEEDS_R52:
            run_dir = model_root / arm / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            start_time = time.time()
            batches = build_random_batch_table(
                values.shape[0],
                16,
                n_steps,
                seed,
            )
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = CoverageSoftTS2Vec(
                input_dims=values.shape[-1],
                output_dims=64,
                hidden_dims=64,
                depth=5,
                device=device_id,
                lr=1e-3,
                batch_size=16,
                lambda_=0.5,
                tau_temp=1.5,
                soft_instance=False,
                soft_temporal=False,
            )
            hook_calls = [0]
            with patched_contrastive_loss("l2reg", beta, hook_calls):
                logs = model.fit_coverage_table(
                    train_data=train_values,
                    soft_labels=None,
                    batch_table=batches,
                    n_epochs=1,
                    verbose=False,
                )
            if hook_calls[0] != n_steps:
                raise RuntimeError(
                    f"{arm} seed={seed}: hook_calls={hook_calls[0]}, "
                    f"expected {n_steps}"
                )
            plain, masked = encode_pair(model, values)
            masked_raw = np.abs(masked - plain).sum(axis=-1)
            cosine_raw, euclidean_raw, clusters = bank_scores(
                plain,
                plain,
                active,
                train_end,
                device,
            )
            masked_z = global_train_z(masked_raw, active, train_end)
            cosine_z = global_train_z(cosine_raw, active, train_end)
            euclidean_z = global_train_z(euclidean_raw, active, train_end)
            pai_mean = euclidean_z + PAI_W_G * g_z + PAI_W_Q * q_z
            pai_official = 0.6 * euclidean_z + 0.4 * g_z + 0.2 * q_z
            geometry = test_geometry(
                plain,
                values,
                active,
                bundle["labels"],
                train_end,
                device,
            )
            geometry["bank_clusters"] = clusters
            score_rows = evaluate_scores(
                {
                    f"{arm}+Mask L1": masked_z,
                    f"{arm}+Cosine": cosine_z,
                    f"{arm}+Euclidean": euclidean_z,
                    f"{arm}+PAI-Mean": pai_mean,
                    f"{arm}+PAI-Official": pai_official,
                },
                data,
                active,
                bundle["test_start"],
                0,
            )
            for row in score_rows:
                row["arm"] = arm
                row["scorer"] = row["method"].split("+", 1)[1]
                row["seed"] = seed
                row.update(geometry)
                row["source"] = "new_s6c_training"
                row["hook_calls"] = hook_calls[0]
                row["n_steps"] = n_steps
                row["elapsed_sec"] = round(time.time() - start_time, 2)
                rows.append(row)
            torch.save(model.net.state_dict(), run_dir / "model.pth")
            (run_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "arm": arm,
                        "seed": seed,
                        "n_steps": n_steps,
                        "hook_calls": hook_calls[0],
                        "final_loss": float(logs[-1]["loss"]),
                        "elapsed_sec": round(time.time() - start_time, 2),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"[DONE] S6c {arm} seed={seed} steps={n_steps} "
                f"norm={geometry['norm_mean']:.3f} "
                f"time={time.time() - start_time:.1f}s",
                flush=True,
            )
            del model, plain, masked
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(by_seed_path, index=False, encoding="utf-8")
    summary = write_s6c_summary(by_seed_path, summary_path)
    print(
        summary[
            (summary["split"] == "test") & (summary["k"] == 10)
        ].to_string(index=False),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.command == "s6a":
        run_s6a(args.force)
    elif args.command == "s6b":
        run_s6b(args.force)
    elif args.command == "s6c":
        run_s6c(args.force)
    else:
        by_seed = OUT_ROOT / "s6c_r52_l2reg_by_seed.csv"
        summary_path = OUT_ROOT / "s6c_r52_l2reg_summary.csv"
        if not by_seed.is_file():
            raise FileNotFoundError(by_seed)
        summary = write_s6c_summary(by_seed, summary_path)
        print(
            summary[
                (summary["split"] == "test") & (summary["k"] == 10)
            ].to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
