from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e2"
)
OUT.mkdir(parents=True, exist_ok=True)
SOFTCLT_DIR = (
    ROOT.parents[1]
    / "references"
    / "softclt_repo"
    / "softclt_ts2vec"
)
sys.path.insert(0, str(SOFTCLT_DIR))

sys.path.insert(0, str(ROOT))
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
)


PADDING_DAYS = 30
COOLDOWN_DAYS = 7
K_GRID = (5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2 TS2Vec temporal SoftCLT training comparison."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["T0", "T1", "T2"],
        default=["T0", "T1", "T2"],
    )
    parser.add_argument(
        "--tau-temps",
        nargs="+",
        type=float,
        default=[1.5, 8.0],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20261116],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Run identifier for immutable aggregate outputs. Defaults to a "
            "UTC timestamp."
        ),
    )
    return parser.parse_args()


def per_user_evidence(
    scores: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    output = np.full(scores.shape, np.nan, dtype=np.float32)
    for user_index in range(scores.shape[0]):
        train_values = scores[
            user_index, : train_end + 1
        ][
            active_mask[user_index, : train_end + 1]
            & np.isfinite(scores[user_index, : train_end + 1])
        ]
        if len(train_values) == 0:
            continue
        ordered = np.sort(train_values)
        n_values = len(ordered)
        finite = active_mask[user_index] & np.isfinite(
            scores[user_index]
        )
        values = scores[user_index, finite]
        ranks = np.searchsorted(ordered, values, side="right")
        tails = (n_values - ranks + 1) / (n_values + 1)
        output[user_index, finite] = np.maximum(
            0.0,
            -np.log(tails),
        )
    return output


def global_z(
    scores: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    mask = active_mask[:, : train_end + 1] & np.isfinite(
        scores[:, : train_end + 1]
    )
    values = scores[:, : train_end + 1][mask]
    return ((scores - values.mean()) / max(values.std(), 1e-8)).astype(
        np.float32
    )


def build_alerts(
    signal: np.ndarray,
    active_mask: np.ndarray,
    k: int,
) -> np.ndarray:
    alerts = np.zeros(signal.shape, dtype=np.uint8)
    last_alert = np.full(signal.shape[0], -10000, dtype=np.int32)
    for time_index in range(signal.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= COOLDOWN_DAYS
        )
        candidates = np.flatnonzero(available)
        if len(candidates) == 0:
            continue
        order = candidates[
            np.lexsort(
                (candidates, -signal[candidates, time_index])
            )
        ][:k]
        alerts[order, time_index] = 1
        last_alert[order] = time_index
    return alerts


def point_metrics(
    signal: np.ndarray,
    labels: np.ndarray,
    active_mask: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    score = signal[:, start : end + 1]
    label = labels[:, start : end + 1]
    active = active_mask[:, start : end + 1]
    valid = active & np.isfinite(score)
    y_true = label[valid]
    y_score = score[valid]
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "positive_days": int(y_true.sum()),
    }


def evaluate_scores(
    data: dict,
    masked_raw: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
    val_end: int,
    method: str,
) -> pd.DataFrame:
    variants = {
        f"{method}_global_z": global_z(
            masked_raw,
            active_mask,
            train_end,
        ),
        f"{method}_user_tail": per_user_evidence(
            masked_raw,
            active_mask,
            train_end,
        ),
    }
    rows = []
    test_start = val_end + 1
    test_end = len(data["dates"]) - 1
    for score_name, score in variants.items():
        rank_score = score.astype(np.float32, copy=True)
        rank_score[~np.isfinite(rank_score)] = -np.inf
        for split, start, end in [
            ("validation", train_end + 1, val_end),
            ("test", test_start, test_end),
        ]:
            point = point_metrics(
                score,
                data["labels"],
                active_mask,
                start,
                end,
            )
            for k in K_GRID:
                alerts = build_alerts(rank_score, active_mask, k)
                result = evaluate_alert_system(
                    data,
                    alerts,
                    score_name,
                    k,
                    external=False,
                )
                rows.append(
                    {
                        "method": score_name,
                        "split": split,
                        "k": k,
                        **compact_metric(result, split),
                        **point,
                    }
                )
    return pd.DataFrame(rows)


def train_variant(
    variant: str,
    tau_temp: float | None,
    seed: int,
    train_values: np.ndarray,
    values: np.ndarray,
    batch_size: int,
    epochs: int,
) -> tuple[np.ndarray, dict]:
    from soft_ts2vec import TS2Vec as SoftTS2Vec

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = 0 if torch.cuda.is_available() else "cpu"
    run_name = (
        f"r42_{variant}"
        + (f"_tau{tau_temp:g}" if tau_temp is not None else "")
        + f"_seed{seed}"
    )
    run_dir = OUT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if variant == "T0":
        lambda_ = 0.5
        soft_temporal = False
    elif variant == "T1":
        lambda_ = 0.0
        soft_temporal = False
    else:
        lambda_ = 0.0
        soft_temporal = True
    model = SoftTS2Vec(
        input_dims=values.shape[-1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        lr=1e-3,
        batch_size=batch_size,
        lambda_=lambda_,
        tau_temp=tau_temp if tau_temp is not None else 2.0,
        max_train_length=None,
        soft_instance=False,
        soft_temporal=soft_temporal,
    )
    print(
        f"Training {variant} seed={seed} tau={tau_temp}",
        flush=True,
    )
    losses = model.fit(
        train_values,
        0,
        0,
        0,
        None,
        str(run_dir),
        n_epochs=epochs,
        verbose=True,
    )
    encode_args = {
        "casual": True,
        "sliding_length": 1,
        "sliding_padding": PADDING_DAYS,
        "batch_size": 32,
    }
    masked = np.asarray(
        model.encode(values, mask="mask_last", **encode_args),
        dtype=np.float32,
    )
    plain = np.asarray(
        model.encode(values, **encode_args),
        dtype=np.float32,
    )
    raw = np.abs(masked - plain).sum(axis=-1)
    torch.save(model.net.state_dict(), run_dir / "model.pth")
    metadata = {
        "variant": variant,
        "tau_temp": tau_temp,
        "seed": seed,
        "lambda": lambda_,
        "soft_temporal": soft_temporal,
        "soft_instance": False,
        "epochs": epochs,
        "batch_size": batch_size,
        "losses": [float(value) for value in losses],
    }
    (run_dir / "training.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return raw, metadata


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    aggregate_dir = OUT / "aggregates" / run_tag
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    values = dataset["values"].astype(np.float32)
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    train_values = values[:, : train_end + 1, :]
    data = load_r42()
    all_results = []
    summaries = []
    for seed in args.seeds:
        runs = []
        if "T0" in args.variants:
            runs.append(("T0", None))
        if "T1" in args.variants:
            runs.append(("T1", None))
        if "T2" in args.variants:
            runs.extend(
                ("T2", float(tau)) for tau in args.tau_temps
            )
        for variant, tau in runs:
            start_time = time.time()
            raw, metadata = train_variant(
                variant,
                tau,
                seed,
                train_values,
                values,
                args.batch_size,
                args.epochs,
            )
            method = (
                f"{variant}"
                + (f"_tau{tau:g}" if tau is not None else "")
            )
            metrics = evaluate_scores(
                data,
                raw,
                active_mask,
                train_end,
                val_end,
                method,
            )
            metrics.insert(0, "seed", seed)
            all_results.append(metrics)
            result_dir = OUT / "results" / method / f"seed_{seed}"
            result_dir.mkdir(parents=True, exist_ok=True)
            np.save(result_dir / "masked_raw.npy", raw)
            metrics.to_csv(
                result_dir / "metrics.csv",
                index=False,
                encoding="utf-8",
            )
            summary = {
                **metadata,
                "elapsed_seconds": time.time() - start_time,
            }
            (result_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summaries.append(summary)
            print(
                f"{method} seed={seed} done in "
                f"{time.time() - start_time:.1f}s",
                flush=True,
            )
    if all_results:
        frame = pd.concat(all_results, ignore_index=True)
        frame.to_csv(
            aggregate_dir / "metrics_by_seed.csv",
            index=False,
            encoding="utf-8",
        )
        summary = (
            frame.groupby(["method", "split", "k"], as_index=False)
            .agg(
                event_recall_3d_mean=("event_recall_3d", "mean"),
                event_recall_3d_std=("event_recall_3d", "std"),
                precision_at_k_mean=("precision_at_k", "mean"),
                precision_at_k_std=("precision_at_k", "std"),
                pr_auc_mean=("pr_auc", "mean"),
                pr_auc_std=("pr_auc", "std"),
                incident_hits_mean=("incident_hits", "mean"),
            )
            .sort_values(["split", "k", "method"])
        )
        summary.to_csv(
            aggregate_dir / "metric_summary.csv",
            index=False,
            encoding="utf-8",
        )
        (aggregate_dir / "run_summaries.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
