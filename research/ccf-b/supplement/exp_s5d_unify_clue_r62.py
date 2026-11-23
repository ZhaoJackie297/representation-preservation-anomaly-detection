"""Exp S5d: unified Hard-Rand vs SoftInst-Rand on CLUE-LDS and CERT r6.2.

The handoff package does not include a Task D script.  This implementation uses
the same CoverageSoftTS2Vec encoder and random-batch 10-epoch budget as S5a/S5b.
It deliberately reports masked-L1 and representation-bank scores only, because
the 52-D CLUE and 27-D r6.2 schemas do not share the CERT 20-D PAI component
contract.
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    average_precision_score,
    pairwise_distances,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_DIR = REPO_ROOT / "runtime" / "cert"
PHASE0_DIR = REPO_ROOT / "research" / "ccf-b" / "phase0"
SOFTCLT_DIR = REPO_ROOT / "references" / "softclt_repo" / "softclt_ts2vec"
sys.path.insert(0, str(CERT_DIR))
sys.path.insert(0, str(PHASE0_DIR))
sys.path.insert(0, str(SOFTCLT_DIR))

from soft_ts2vec import TS2Vec as SoftTS2Vec  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
)


OUT_ROOT = (
    REPO_ROOT
    / "research"
    / "ccf-b"
    / "supplement_results"
    / "exp_s5d_unify_clue_r62"
)
CLUE_DATA = Path(
    r"E:\MarketRisker\work\cert\external\clue_lds"
    r"\processed_final\clue_user_day_dataset.npz"
)
R62_ROOT = Path(r"E:\MarketRisker\work\cert\processed_r62")
CLUE_SEEDS = list(range(20261116, 20261126))
R62_SEEDS = [20261116, 20261117, 20261118]
K_GRID = (5, 10, 20)
COOLDOWN_DAYS = 7


def global_train_z(
    scores: np.ndarray,
    active: np.ndarray,
    train_end: int,
) -> np.ndarray:
    train_mask = active[:, : train_end + 1] & np.isfinite(
        scores[:, : train_end + 1]
    )
    values = scores[:, : train_end + 1][train_mask]
    if len(values) == 0:
        raise RuntimeError("No finite global calibration scores.")
    mean = float(values.mean())
    std = max(float(values.std()), 1e-8)
    return ((scores - mean) / std).astype(np.float32)


def bank_scores(
    train_repr: np.ndarray,
    all_repr: np.ndarray,
    active: np.ndarray,
    train_end: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    train = train_repr[:, : train_end + 1][
        active[:, : train_end + 1]
    ]
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
    bank = torch.from_numpy(
        kmeans.cluster_centers_.astype(np.float32)
    ).to(device)
    bank_norm = torch.nn.functional.normalize(bank, dim=1)
    active_users, active_times = np.nonzero(active)
    queries = all_repr[active_users, active_times]
    cosine_values = []
    euclidean_values = []
    with torch.inference_mode():
        for start in range(0, len(queries), 4096):
            query = torch.from_numpy(
                queries[start : start + 4096].astype(np.float32)
            ).to(device)
            query_norm = torch.nn.functional.normalize(query, dim=1)
            cosine = (1.0 - query_norm @ bank_norm.T).clamp_min_(0.0)
            euclidean = torch.cdist(query, bank)
            cosine_values.append(
                cosine.topk(3, dim=1, largest=False)
                .values.mean(dim=1)
                .cpu()
                .numpy()
            )
            euclidean_values.append(
                euclidean.topk(3, dim=1, largest=False)
                .values.mean(dim=1)
                .cpu()
                .numpy()
            )
    cosine_score = np.full(active.shape, np.nan, dtype=np.float32)
    euclidean_score = np.full(active.shape, np.nan, dtype=np.float32)
    cosine_score[active_users, active_times] = np.concatenate(cosine_values)
    euclidean_score[active_users, active_times] = np.concatenate(euclidean_values)
    return cosine_score, euclidean_score


def build_alerts(
    score: np.ndarray,
    active: np.ndarray,
    k: int,
    cooldown_days: int = COOLDOWN_DAYS,
) -> np.ndarray:
    alerts = np.zeros(score.shape, dtype=np.uint8)
    last_alert = np.full(score.shape[0], -10000, dtype=np.int64)
    for time in range(score.shape[1]):
        available = active[:, time] & (time - last_alert >= cooldown_days)
        candidates = np.flatnonzero(available & np.isfinite(score[:, time]))
        if len(candidates) == 0:
            continue
        ranked = candidates[
            np.lexsort((candidates, -score[candidates, time]))
        ][:k]
        alerts[ranked, time] = 1
        last_alert[ranked] = time
    return alerts


def point_metrics(
    signal: np.ndarray,
    data: dict,
    active: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float | int]:
    split_score = signal[:, start : end + 1]
    split_labels = data["labels"][:, start : end + 1]
    split_active = active[:, start : end + 1]
    valid = split_active & np.isfinite(split_score)
    y_true = split_labels[valid].astype(np.int8)
    y_score = split_score[valid]
    if len(np.unique(y_true)) < 2:
        return {"pr_auc": 0.0, "roc_auc": 0.0, "positive_days": int(y_true.sum())}
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "positive_days": int(y_true.sum()),
    }


def densify(values: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    return (
        (2.0 * alpha) / (1.0 + np.exp(-tau * values))
        + (1.0 - alpha) * np.eye(values.shape[0])
    )


def soft_instance_labels(
    profile: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, dict]:
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
    median = float(
        np.median(normalized[np.triu_indices(len(normalized), k=1)])
    )
    tau = 1.0 / max(median, 1e-6)
    return densify(-normalized, tau, alpha).astype(np.float32), {
        "alpha": alpha,
        "median_normalized_distance": median,
        "tau": tau,
    }


def fixed_length_chunks(
    values: np.ndarray,
    soft_labels: np.ndarray,
    chunk_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    sections = values.shape[1] // chunk_length
    if sections < 2:
        return values, soft_labels
    usable = values[:, : sections * chunk_length]
    chunks = np.concatenate(
        np.split(usable, sections, axis=1),
        axis=0,
    )
    expanded_labels = np.tile(
        soft_labels,
        (sections, sections),
    )
    return chunks, expanded_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp S5d: unify Hard-Rand and SoftInst-Rand on CLUE/r6.2"
    )
    parser.add_argument("--dataset", choices=["clue", "r62"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def training_profiles(
    values: np.ndarray,
    active: np.ndarray,
    train_end: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for user in range(values.shape[0]):
        mask = active[user, : train_end + 1]
        user_values = values[user, : train_end + 1][mask]
        if len(user_values) == 0:
            user_values = values[user, : train_end + 1]
        rows.append(
            np.concatenate(
                [
                    np.nanmedian(user_values, axis=0),
                    np.nanquantile(user_values, 0.9, axis=0),
                    np.asarray([mask.mean()], dtype=np.float64),
                ]
            )
        )
    profile = np.stack(rows)
    population_median = np.nanmedian(profile, axis=0)
    population_median = np.where(np.isfinite(population_median), population_median, 0)
    missing_rows, missing_columns = np.nonzero(~np.isfinite(profile))
    profile[missing_rows, missing_columns] = population_median[missing_columns]
    mean = profile.mean(axis=0)
    std = np.where(profile.std(axis=0) < 1e-8, 1.0, profile.std(axis=0))
    return ((profile - mean) / std).astype(np.float64)


def load_clue() -> dict:
    with np.load(CLUE_DATA, allow_pickle=False) as dataset:
        values = dataset["values"].astype(np.float32)
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active = dataset["active_mask"].astype(bool)
        event_id = dataset["event_id"].astype(np.int16)
        train_end = int(dataset["train_end_index"])
        test_start = int(dataset["eval_start_index"])
        test_end = int(dataset["eval_end_index"])
    labels = (event_id >= 0).astype(np.uint8)
    incidents = []
    for incident_id in sorted(int(value) for value in np.unique(event_id[event_id >= 0])):
        user_rows, time_rows = np.nonzero(event_id == incident_id)
        if len(user_rows) == 0:
            continue
        event_users = sorted(set(users[user_rows].tolist()))
        ordered_times = np.sort(time_rows)
        incidents.append(
            {
                "incident_id": str(incident_id),
                "scenario": "account_switch",
                "observed_accounts": "|".join(event_users),
                "first_observed_date": dates[ordered_times[0]],
                "last_observed_date": dates[ordered_times[-1]],
            }
        )
    incident_frame = pd.DataFrame(incidents)
    return {
        "values": values,
        "users": users,
        "dates": dates,
        "active": active,
        "labels": labels,
        "incidents": incident_frame,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_indices": np.arange(len(users)),
        "data": {
            "users": users,
            "dates": dates,
            "active_mask": active,
            "labels": labels,
            "incidents": incident_frame,
            "test_start": test_start,
        },
    }


def r62_label_matrix(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    labels = np.zeros((len(users), len(dates)), dtype=np.uint8)
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)}
    frame = pd.read_csv(
        R62_ROOT / "r6.2_malicious_account_days.csv",
        parse_dates=["date"],
    )
    for row in frame.itertuples(index=False):
        user = user_index.get(row.user_id)
        time = date_index.get(row.date.strftime("%Y-%m-%d"))
        if user is not None and time is not None:
            labels[user, time] = 1
    return labels


def load_r62() -> dict:
    with np.load(
        R62_ROOT / "r6.2_sequence_dataset.npz",
        allow_pickle=False,
    ) as dataset:
        values = dataset["values"].astype(np.float32)
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
        test_start = int(dataset["test_start_index"])
        test_end = int(dataset["test_end_index"])
    rng = np.random.default_rng(2024)
    train_indices = np.sort(
        rng.choice(
            len(users),
            size=min(1000, len(users)),
            replace=False,
        )
    )
    labels = r62_label_matrix(users, dates)
    incidents = pd.read_csv(
        R62_ROOT / "r6.2_incidents.csv",
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
        "test_end": test_end,
        "train_indices": train_indices,
        "data": {
            "users": users,
            "dates": dates,
            "active_mask": active,
            "labels": labels,
            "incidents": incidents,
            "test_start": test_start,
        },
    }


def evaluate_signal(
    method: str,
    signal: np.ndarray,
    bundle: dict,
    seed: int,
) -> list[dict]:
    active = bundle["active"]
    data = bundle["data"]
    test_start = bundle["test_start"]
    test_end = bundle["test_end"]
    rank = signal.astype(np.float32, copy=True)
    rank[~np.isfinite(rank)] = -np.inf
    point = point_metrics(signal, data, active, test_start, test_end)
    rows = []
    for k in K_GRID:
        alerts = build_alerts(rank, active, k)
        result = evaluate_alert_system(
            data,
            alerts,
            method,
            k,
            external=True,
        )
        rows.append(
            {
                "seed": seed,
                "dataset": "r6.2" if "insider_id" in data["incidents"] else "clue",
                "method": method,
                "split": "test",
                "k": k,
                **compact_metric(result, None),
                **point,
            }
        )
    return rows


def run_dataset(
    dataset_name: str,
    seeds: list[int],
    batch_size: int,
    epochs: int,
    force: bool,
) -> None:
    bundle = load_clue() if dataset_name == "clue" else load_r62()
    out_dir = OUT_ROOT / ("clue_10seed" if dataset_name == "clue" else "r62_3seed")
    out_dir.mkdir(parents=True, exist_ok=True)
    values = bundle["values"]
    active = bundle["active"]
    train_indices = bundle["train_indices"]
    train_values = values[train_indices, : bundle["train_end"] + 1]
    train_active = active[train_indices, : bundle["train_end"] + 1]
    profiles = training_profiles(train_values, train_active, bundle["train_end"])
    soft_labels, distance_meta = soft_instance_labels(profiles, alpha=0.5)
    chunk_length = 256
    fit_values, fit_soft_labels = fixed_length_chunks(
        train_values,
        soft_labels,
        chunk_length,
    )
    nominal_steps = (len(fit_values) // batch_size) * epochs
    device_id = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{device_id}" if isinstance(device_id, int) else "cpu")
    all_rows: list[dict] = []

    for arm, use_soft in (("Hard-Rand-10ep", False), ("SoftInst-Rand-10ep", True)):
        for seed in seeds:
            run_dir = out_dir / arm / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_csv = run_dir / "metrics.csv"
            if metrics_csv.exists() and not force:
                all_rows.extend(pd.read_csv(metrics_csv).to_dict(orient="records"))
                print(f"[SKIP] {dataset_name} {arm} seed={seed}", flush=True)
                continue

            start_time = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = SoftTS2Vec(
                input_dims=values.shape[-1],
                output_dims=64,
                hidden_dims=64,
                depth=5,
                device=device_id,
                lr=1e-3,
                batch_size=batch_size,
                lambda_=0.5,
                tau_temp=1.5,
                max_train_length=None,
                soft_instance=use_soft,
                soft_temporal=False,
            )
            logs = model.fit(
                fit_values,
                0,
                0,
                0,
                fit_soft_labels if use_soft else None,
                str(run_dir / "native_training"),
                n_epochs=epochs,
                verbose=False,
            )
            encode_args = {
                "casual": True,
                "sliding_length": 1,
                "sliding_padding": 30,
                "batch_size": 256,
            }
            plain = np.asarray(model.encode(values, **encode_args), dtype=np.float32)
            masked = np.asarray(
                model.encode(values, mask="mask_last", **encode_args),
                dtype=np.float32,
            )
            torch.save(model.net.state_dict(), run_dir / "model.pth")
            masked_raw = np.abs(masked - plain).sum(axis=-1)
            masked_z = global_train_z(masked_raw, active, bundle["train_end"])
            cosine_raw, euclidean_raw = bank_scores(
                plain,
                plain,
                active,
                bundle["train_end"],
                device,
            )
            cosine_z = global_train_z(cosine_raw, active, bundle["train_end"])
            euclidean_z = global_train_z(euclidean_raw, active, bundle["train_end"])
            seed_rows = []
            for scorer, signal in (
                ("plain_l1", masked_z),
                ("bank_cosine", cosine_z),
                ("bank_euclidean", euclidean_z),
            ):
                seed_rows.extend(
                    evaluate_signal(f"{arm}__{scorer}", signal, bundle, seed)
                )
            pd.DataFrame(seed_rows).to_csv(
                metrics_csv,
                index=False,
                encoding="utf-8",
            )
            (run_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "arm": arm,
                        "seed": seed,
                        "train_users": int(len(train_indices)),
                        "epochs": epochs,
                        "nominal_steps": nominal_steps,
                        "model_impl": "upstream SoftTS2Vec",
                        "training_chunk_length": chunk_length,
                        "training_instances": int(len(fit_values)),
                        "final_loss": float(logs[-1]) if logs else None,
                        "elapsed_sec": round(time.time() - start_time, 2),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            all_rows.extend(seed_rows)
            print(
                f"[DONE] {dataset_name} {arm} seed={seed} "
                f"epochs={epochs} nominal_steps={nominal_steps} "
                f"time={time.time() - start_time:.1f}s",
                flush=True,
            )
            del model, plain, masked
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    frame = pd.DataFrame(all_rows)
    frame.to_csv(out_dir / "metrics_by_seed.csv", index=False, encoding="utf-8")
    summary = (
        frame.groupby(["dataset", "method", "split", "k"], as_index=False)
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
        .sort_values(["split", "k", "method"])
    )
    summary.to_csv(out_dir / "method_summary.csv", index=False, encoding="utf-8")
    (OUT_ROOT / f"{dataset_name}_manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "seeds": seeds,
                "train_users": int(len(train_indices)),
                "evaluation_users": int(len(bundle["users"])),
                "train_end": int(bundle["train_end"]),
                "test_start": int(bundle["test_start"]),
                "test_end": int(bundle["test_end"]),
                "batch_size": batch_size,
                "epochs": epochs,
                "nominal_steps": nominal_steps,
                "model_impl": "upstream SoftTS2Vec",
                "training_chunk_length": chunk_length,
                "training_instances": int(len(fit_values)),
                "arms": ["Hard-Rand-10ep", "SoftInst-Rand-10ep"],
                "scorers": ["plain_l1", "bank_cosine", "bank_euclidean"],
                "distance_metadata": distance_meta,
                "labels_used_in_training": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary[(summary["k"] == 10)].to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.dataset == "clue":
        seeds = args.seeds or CLUE_SEEDS
    else:
        seeds = args.seeds or R62_SEEDS
    run_dataset(
        args.dataset,
        seeds,
        args.batch_size,
        args.epochs,
        args.force,
    )


if __name__ == "__main__":
    main()
