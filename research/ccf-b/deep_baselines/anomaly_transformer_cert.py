from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from anomaly_transformer_smd import association_terms, my_kl_loss
from protocol_utils import record_run_protocol


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def block_windows(values: np.ndarray, window: int) -> np.ndarray:
    starts = list(range(0, len(values), window))
    blocks = []
    for start in starts:
        block = values[start : start + window]
        if len(block) < window:
            padding = np.zeros(
                (window - len(block), values.shape[1]),
                dtype=values.dtype,
            )
            block = np.concatenate([block, padding], axis=0)
        blocks.append(block)
    return np.stack(blocks).astype(np.float32)


def label_matrix(
    labels_path: Path,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    frame = pd.read_csv(labels_path, parse_dates=["date"])
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    matrix = np.zeros((len(users), len(dates)), dtype=np.uint8)
    for row in frame.itertuples(index=False):
        user = user_index.get(str(row.user_id))
        day = date_index.get(pd.Timestamp(row.date).strftime("%Y-%m-%d"))
        if user is not None and day is not None:
            matrix[user, day] = 1
    return matrix


def score_segment(
    model: nn.Module,
    values: np.ndarray,
    *,
    start: int,
    end: int,
    window: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    segment = values[:, start : end + 1]
    all_windows = []
    for user in range(len(segment)):
        all_windows.append(block_windows(segment[user], window))
    windows = np.concatenate(all_windows, axis=0)
    criterion = nn.MSELoss(reduction="none")
    scores = []
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(
                windows[batch_start : batch_start + batch_size]
            ).to(device)
            reconstruction, series, prior, _ = model(batch)
            point_loss = criterion(batch, reconstruction).mean(dim=-1)
            series_loss, prior_loss = association_terms(
                series,
                prior,
                temperature=50,
            )
            metric = torch.softmax(-series_loss - prior_loss, dim=-1)
            scores.append(
                (metric[:, None] * point_loss).cpu().numpy().reshape(-1)
            )
    flat = np.concatenate(scores, axis=0)
    blocks_per_user = (segment.shape[1] + window - 1) // window
    output = np.empty(segment.shape[:2], dtype=np.float32)
    for user in range(len(segment)):
        block_scores = flat[
            user * blocks_per_user * window : (user + 1) * blocks_per_user * window
        ]
        output[user] = block_scores[: segment.shape[1]]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261201])
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--user-limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Anomaly Transformer official code requires CUDA")
    device = torch.device(args.device)
    sys.path.insert(0, str(args.repo.resolve()))
    sys.path.insert(0, str(args.runtime_dir.resolve()))
    from model.AnomalyTransformer import AnomalyTransformer
    from e6_r62_audit import point_metrics
    from production_protocol_r42 import build_alerts, evaluate

    dataset = np.load(args.dataset, allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    test_start = int(dataset["test_start_index"])
    test_end = int(dataset["test_end_index"])
    if args.user_limit is not None:
        values = values[: args.user_limit]
        users = users[: args.user_limit]
        active = active[: args.user_limit]
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    labels = label_matrix(args.labels, users, dates)
    incidents = pd.read_csv(args.incidents)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "repo": str(args.repo),
        "dataset": str(args.dataset),
        "seeds": args.seeds,
        "window": args.window,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "layers": args.layers,
        "user_limit": args.user_limit,
        "adaptation": "one model over all user-day windows; block scoring",
    }
    record_run_protocol(
        args.output_dir / "protocol.json",
        protocol,
        args.seeds,
    )

    train_windows = []
    for user in range(len(values)):
        train_windows.append(
            block_windows(values[user, : train_end + 1], args.window)
        )
    train_windows = np.concatenate(train_windows, axis=0)

    summaries = []
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        score_path = seed_dir / "scores.npz"
        summary_path = seed_dir / "summary.json"
        if score_path.exists() and summary_path.exists() and not args.force:
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue

        start_time = time.time()
        set_seed(seed)
        model = AnomalyTransformer(
            win_size=args.window,
            enc_in=values.shape[-1],
            c_out=values.shape[-1],
            d_model=args.d_model,
            n_heads=8,
            e_layers=args.layers,
            d_ff=args.d_model * 2,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        rng = np.random.default_rng(seed)
        for _ in range(args.epochs):
            model.train()
            order = rng.permutation(len(train_windows))
            for batch_start in range(0, len(order), args.batch_size):
                indices = order[batch_start : batch_start + args.batch_size]
                batch = torch.from_numpy(train_windows[indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, series, prior, _ = model(batch)
                series_loss, prior_loss = association_terms(series, prior)
                reconstruction_loss = criterion(reconstruction, batch)
                loss_series = reconstruction_loss - 3.0 * series_loss.mean()
                loss_prior = reconstruction_loss + 3.0 * prior_loss.mean()
                loss_series.backward(retain_graph=True)
                loss_prior.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

        scores = np.empty_like(values[:, :, 0], dtype=np.float32)
        scores[:, : train_end + 1] = score_segment(
            model,
            values,
            start=0,
            end=train_end,
            window=args.window,
            batch_size=args.batch_size,
            device=device,
        )
        scores[:, test_start : test_end + 1] = score_segment(
            model,
            values,
            start=test_start,
            end=test_end,
            window=args.window,
            batch_size=args.batch_size,
            device=device,
        )
        if val_end >= test_start - 1:
            scores[:, train_end + 1 : val_end + 1] = -np.inf

        point = point_metrics(scores, labels, active, test_start)
        data = {
            "users": users,
            "dates": dates,
            "active_mask": active,
            "labels": labels,
            "incidents": incidents,
            "test_start": test_start,
        }
        event = {}
        if args.user_limit is None:
            for k in (5, 10, 20):
                alerts = build_alerts(scores, active, k, 7)
                event[str(k)] = evaluate(
                    users,
                    dates,
                    active,
                    alerts,
                    labels,
                    incidents,
                    train_end,
                    val_end,
                    "anomaly_transformer",
                    k,
                )
        summary = {
            "seed": seed,
            "point": point,
            "event": event,
            "elapsed_seconds": time.time() - start_time,
        }
        np.savez_compressed(
            score_path,
            scores=scores,
            labels=labels,
            active_mask=active,
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        summaries.append(summary)

    (args.output_dir / "all_seeds_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
