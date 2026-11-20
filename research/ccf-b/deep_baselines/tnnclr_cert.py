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
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score

from anomaly_transformer_cert import block_windows, label_matrix
from protocol_utils import record_run_protocol


def point_metrics(
    signal: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
    start: int,
) -> dict[str, float]:
    scores = signal[:, start:]
    label = labels[:, start:]
    mask = active[:, start:] & np.isfinite(scores)
    return {
        "pr_auc": float(average_precision_score(label[mask], scores[mask])),
        "roc_auc": float(roc_auc_score(label[mask], scores[mask])),
        "positive_days": int(label[mask].sum()),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TNCLRModel(torch.nn.Module):
    def __init__(
        self,
        *,
        encoder,
        projection_head,
        prediction_head,
        augmenter,
        embed_dim: int,
        queue_size: int = 1000,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.prediction_head = prediction_head
        self.augmenter = augmenter
        self.embed_dim = embed_dim
        self.queue_size = queue_size
        self.temperature = temperature
        self.register_buffer(
            "feature_queue",
            F.normalize(torch.randn(queue_size, embed_dim), dim=1),
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_queue(self, features: torch.Tensor) -> None:
        features = F.normalize(features, dim=1)
        batch_size = len(features)
        pointer = int(self.queue_ptr)
        if pointer + batch_size <= self.queue_size:
            self.feature_queue[pointer : pointer + batch_size] = features
        else:
            remaining = self.queue_size - pointer
            self.feature_queue[pointer:] = features[:remaining]
            self.feature_queue[: batch_size - remaining] = features[remaining:]
        self.queue_ptr[0] = (pointer + batch_size) % self.queue_size

    def nearest_neighbors(self, projections: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(projections, dim=1)
        queue = F.normalize(self.feature_queue, dim=1)
        indices = (normalized @ queue.T).argmax(dim=1)
        return self.feature_queue[indices]

    def contrastive_loss(
        self,
        neighbors: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        similarities = (
            F.normalize(predictions, dim=1)
            @ F.normalize(neighbors, dim=1).T
            / self.temperature
        )
        labels = torch.arange(len(similarities), device=similarities.device)
        return F.cross_entropy(similarities, labels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = self.augmenter(values)
        second = self.augmenter(values)
        first_projection = self.projection_head(self.encoder(first))
        second_projection = self.projection_head(self.encoder(second))
        first_prediction = self.prediction_head(first_projection)
        second_prediction = self.prediction_head(second_projection)
        first_neighbors = self.nearest_neighbors(first_projection)
        second_neighbors = self.nearest_neighbors(second_projection)
        self.update_queue(first_projection.detach())
        return (
            self.contrastive_loss(first_neighbors, second_prediction)
            + self.contrastive_loss(second_neighbors, first_prediction)
        ) / 2.0

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)


def windows_for_segment(
    values: np.ndarray,
    *,
    start: int,
    end: int,
    window: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    segment = values[:, start : end + 1]
    windows = []
    locations = []
    for user in range(len(segment)):
        blocks = block_windows(segment[user], window)
        windows.extend(blocks)
        for block_index in range(len(blocks)):
            block_start = start + block_index * window
            locations.append((user, block_start, min(block_start + window, end + 1)))
    return np.asarray(windows, dtype=np.float32), locations


def encode_windows(
    model: TNCLRModel,
    windows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(
                windows[start : start + batch_size]
            ).to(device)
            outputs.append(model.encode(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def window_bank_scores(
    train_repr: np.ndarray,
    target_repr: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    clusters = min(500, max(2, int(round(0.1 * len(train_repr)))))
    model = MiniBatchKMeans(
        n_clusters=clusters,
        init="k-means++",
        random_state=2024,
        batch_size=max(8192, clusters),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    ).fit(train_repr)
    bank = torch.from_numpy(model.cluster_centers_.astype(np.float32)).to(device)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(target_repr), 4096):
            query = torch.from_numpy(
                target_repr[start : start + 4096].astype(np.float32)
            ).to(device)
            scores.append(
                torch.cdist(query, bank)
                .topk(3, dim=1, largest=False)
                .values.mean(dim=1)
                .cpu()
                .numpy()
            )
    return np.concatenate(scores).astype(np.float32)


def scores_to_days(
    length: int,
    window_scores: np.ndarray,
    locations: list[tuple[int, int, int]],
    *,
    users: int,
    active: np.ndarray,
) -> np.ndarray:
    output = np.zeros((users, length), dtype=np.float32)
    counts = np.zeros((users, length), dtype=np.int16)
    for score, (user, start, end) in zip(window_scores, locations):
        output[user, start:end] += score
        counts[user, start:end] += 1
    seen = counts > 0
    output[seen] /= counts[seen]
    output[~seen & active] = np.nan
    output[~seen & ~active] = -np.inf
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
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--user-limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    sys.path.insert(0, str(args.repo.resolve()))
    sys.path.insert(0, str(args.runtime_dir.resolve()))
    from models.encoder import Encoder, ProjectionHead, PredictionHead
    from utils.augmentations import get_augmenter
    from production_protocol_r42 import build_alerts, evaluate

    dataset = np.load(args.dataset, allow_pickle=False)
    values = np.nan_to_num(
        dataset["values"].astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
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
    labels = label_matrix(args.labels, users, dates)
    incidents = pd.read_csv(args.incidents)

    train_windows, _ = windows_for_segment(
        values,
        start=0,
        end=train_end,
        window=args.window,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_run_protocol(
        args.output_dir / "protocol.json",
        {
            "repo": str(args.repo),
            "window": args.window,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "embed_dim": args.embed_dim,
            "layers": args.layers,
            "heads": args.heads,
            "user_limit": args.user_limit,
            "adaptation": "t-NNCLR encoder plus fixed representation bank",
        },
        args.seeds,
    )

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
        encoder = Encoder(
            input_shape=(args.window, values.shape[-1]),
            segment_size=4,
            embed_dim=args.embed_dim,
            num_heads=args.heads,
            num_layers=args.layers,
            mlp_ratio=4.0,
            dropout=0.1,
        )
        model = TNCLRModel(
            encoder=encoder,
            projection_head=ProjectionHead(args.embed_dim, args.embed_dim),
            prediction_head=PredictionHead(args.embed_dim, args.embed_dim * 2),
            augmenter=get_augmenter("contrastive_augmenter"),
            embed_dim=args.embed_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.05,
        )
        rng = np.random.default_rng(seed)
        for _ in range(args.epochs):
            model.train()
            order = rng.permutation(len(train_windows))
            for batch_start in range(0, len(order), args.batch_size):
                indices = order[batch_start : batch_start + args.batch_size]
                if len(indices) < 2:
                    continue
                batch = torch.from_numpy(train_windows[indices]).to(device)
                loss = model(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        train_repr = encode_windows(
            model,
            train_windows,
            batch_size=256,
            device=device,
        )
        full_windows, full_locations = windows_for_segment(
            values,
            start=0,
            end=test_end,
            window=args.window,
        )
        full_repr = encode_windows(
            model,
            full_windows,
            batch_size=256,
            device=device,
        )
        window_scores = window_bank_scores(
            train_repr,
            full_repr,
            device=device,
        )
        scores = scores_to_days(
            test_end + 1,
            window_scores,
            full_locations,
            users=len(users),
            active=active,
        )
        scores[:, : train_end + 1] = np.nan
        scores[:, train_end + 1 : test_start] = -np.inf
        point = point_metrics(scores, labels, active, test_start)
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
                    "tnnclr",
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
