from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch

PHASE0 = Path(__file__).resolve().parent
if str(PHASE0) not in sys.path:
    sys.path.insert(0, str(PHASE0))

from common_coverage_sampler import build_common_coverage_table  # noqa: E402
from coverage_soft_ts2vec import CoverageSoftTS2Vec  # noqa: E402


VARIANTS = (
    "A0-official",
    "A0-matched",
    "A0-A3batch",
    "A1-R0",
    "A1-Cov",
    "A1-Cov-A5a",
    "A1-Cov-uniform",
)


def random_batch_table(
    *,
    user_count: int,
    batch_size: int,
    batches: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    if user_count < batch_size:
        raise ValueError("user_count must be at least batch_size")
    rng = np.random.default_rng(seed)
    return tuple(
        tuple(sorted(int(value) for value in rng.choice(
            user_count, size=batch_size, replace=False
        )))
        for _ in range(batches)
    )


def densify(
    values: np.ndarray,
    tau: float,
    alpha: float,
) -> np.ndarray:
    exponent = -tau * values
    exponent = np.clip(exponent, -700.0, 700.0)
    sigmoid = np.where(
        exponent >= 0,
        1.0 / (1.0 + np.exp(-exponent)),
        np.exp(exponent) / (1.0 + np.exp(exponent)),
    )
    return (
        2.0 * alpha * sigmoid
        + (1.0 - alpha) * np.eye(values.shape[0])
    )


def profile_soft_labels(
    profile_features: np.ndarray,
    *,
    alpha: float = 0.5,
) -> tuple[np.ndarray, dict[str, object]]:
    profiles = np.asarray(profile_features, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError("profile features must be user x feature")
    squared_norm = np.sum(profiles * profiles, axis=1)
    squared_distance = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * profiles @ profiles.T
    )
    distances = np.sqrt(np.clip(squared_distance, 0.0, None))
    upper = distances[np.triu_indices(len(distances), k=1)]
    minimum = float(upper.min())
    maximum = float(upper.max())
    scale = maximum - minimum
    normalized = (
        (distances - minimum) / scale
        if scale > 1e-12
        else np.zeros_like(distances)
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    median_distance = float(
        np.median(normalized[np.triu_indices(len(normalized), k=1)])
    )
    tau_inst = 1.0 / max(median_distance, 1e-6)
    weights = densify(-normalized, tau_inst, alpha).astype(np.float32)
    return weights, {
        "distance": "user_profile_euclidean",
        "profile_dimensions": int(profiles.shape[1]),
        "alpha": alpha,
        "minimum_distance": minimum,
        "maximum_distance": maximum,
        "median_normalized_distance": median_distance,
        "tau_inst": tau_inst,
        "missing_policy": "profile features are already training-period imputed and standardized",
    }


def load_graph_edges(graph_file: Path) -> tuple[int, dict[str, set[tuple[int, int]]]]:
    with np.load(graph_file) as data:
        prior = data["prior"]
        if prior.ndim != 2 or prior.shape[0] != prior.shape[1]:
            raise ValueError("graph file prior must be square")
        targets = {
            name: {
                tuple(sorted((int(edge[0]), int(edge[1]))))
                for edge in data[name]
            }
            for name in ("A3", "A5a", "A5b")
        }
    return int(prior.shape[0]), targets


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ordered_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and ordered_scores[end] == ordered_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def evaluate_scores(
    *,
    scores: np.ndarray,
    event_id: np.ndarray,
    active_mask: np.ndarray,
    eval_start: int,
    eval_end: int,
    k: int = 10,
    cooldown_days: int = 7,
) -> dict[str, object]:
    scores = np.asarray(scores, dtype=np.float64)
    event_id = np.asarray(event_id)
    active_mask = np.asarray(active_mask, dtype=bool)
    if scores.shape != event_id.shape or scores.shape != active_mask.shape:
        raise ValueError("scores, event_id, and active_mask must share shape")

    eval_slice = np.s_[:, eval_start : eval_end + 1]
    eval_active = active_mask[eval_slice]
    eval_labels = (event_id[eval_slice] >= 0) & eval_active
    eval_scores = scores[eval_slice]
    finite = eval_active & np.isfinite(eval_scores)
    point_scores = eval_scores[finite]
    point_labels = eval_labels[finite]

    event_count = int(event_id.max()) + 1 if event_id.max() >= 0 else 0
    hit_events: set[int] = set()
    last_hit: dict[int, int] = {}
    for day in range(eval_start, eval_end + 1):
        candidates = np.flatnonzero(active_mask[:, day] & np.isfinite(scores[:, day]))
        if len(candidates) == 0:
            continue
        order = candidates[
            np.argsort(-scores[candidates, day], kind="mergesort")
        ][:k]
        for user in order:
            current = int(event_id[user, day])
            if current < 0:
                continue
            previous = last_hit.get(current)
            if previous is not None and day - previous < cooldown_days:
                continue
            hit_events.add(current)
            last_hit[current] = day

    return {
        "pr_auc": average_precision(point_labels, point_scores),
        "roc_auc": roc_auc(point_labels, point_scores),
        f"event_recall_at_k{k}": (
            len(hit_events) / event_count if event_count else 0.0
        ),
        "events_hit": len(hit_events),
        "events_total": event_count,
        "eval_active_user_days": int(eval_active.sum()),
        "labeled_user_days": int(point_labels.sum()),
    }


def robust_scores(
    train_errors: np.ndarray,
    all_errors: np.ndarray,
) -> np.ndarray:
    median = float(np.median(train_errors))
    mad = float(np.median(np.abs(train_errors - median)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(np.std(train_errors))
    if scale < 1e-8:
        scale = 1.0
    return np.clip((all_errors - median) / scale, 0.0, None)


def choose_batch_table(
    *,
    variant: str,
    user_count: int,
    batch_size: int,
    seed: int,
    base_steps: int,
    coverage,
) -> tuple[tuple[int, ...], ...]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if variant == "A0-official":
        return random_batch_table(
            user_count=user_count,
            batch_size=batch_size,
            batches=base_steps,
            seed=seed,
        )
    if variant in (
        "A0-matched",
        "A0-A3batch",
        "A1-Cov",
        "A1-Cov-A5a",
        "A1-Cov-uniform",
    ):
        return coverage.training_batches
    return random_batch_table(
        user_count=user_count,
        batch_size=batch_size,
        batches=len(coverage.training_batches),
        seed=seed,
    )


def hash_batch_table(batch_table: Iterable[Iterable[int]]) -> str:
    digest = hashlib.sha256()
    for batch in batch_table:
        digest.update(",".join(str(int(index)) for index in batch).encode("ascii"))
        digest.update(b";")
    return digest.hexdigest()


def train_variant(
    *,
    variant: str,
    seed: int,
    train_values: np.ndarray,
    soft_labels: np.ndarray,
    batch_table: tuple[tuple[int, ...], ...],
    output_dir: Path,
    device: str,
    batch_size: int,
    tau_temp: float,
    uniform_labels: np.ndarray | None = None,
) -> dict[str, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if variant == "A1-Cov-uniform":
        selected_soft_labels = uniform_labels
    elif variant in ("A1-R0", "A1-Cov", "A1-Cov-A5a"):
        selected_soft_labels = soft_labels
    else:
        selected_soft_labels = None
    if variant in ("A1-Cov-uniform", "A1-Cov-A5a") and selected_soft_labels is None:
        raise ValueError(f"{variant} requires soft labels")
    model = CoverageSoftTS2Vec(
        input_dims=train_values.shape[-1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        lr=1e-3,
        batch_size=batch_size,
        lambda_=0.5,
        tau_temp=tau_temp,
        soft_instance=selected_soft_labels is not None,
        soft_temporal=False,
    )
    logs = model.fit_coverage_table(
        train_data=train_values,
        soft_labels=selected_soft_labels,
        batch_table=batch_table,
        n_epochs=1,
        verbose=False,
    )
    model.save(str(output_dir / "model.pth"))
    return {
        "variant": variant,
        "seed": seed,
        "optimizer_steps": len(batch_table),
        "batch_table_sha256": hash_batch_table(batch_table),
        "soft_instance": selected_soft_labels is not None,
        "soft_temporal": False,
        "loss_first": float(logs[0]["loss"]),
        "loss_last": float(logs[-1]["loss"]),
        "active_pair_mass_mean": float(
            np.mean([float(row["active_pair_mass"]) for row in logs])
        ),
        "realized_pairs_mean": float(
            np.mean([float(row["realized_pairs"]) for row in logs])
        ),
    }


def encode_dataset(
    model: CoverageSoftTS2Vec,
    values: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    encode_args = {
        "causal": True,
        "sliding_length": 1,
        "sliding_padding": 30,
        "batch_size": 32,
    }
    masked = np.asarray(
        model.encode(values, mask="mask_last", **encode_args),
        dtype=np.float32,
    )
    plain = np.asarray(model.encode(values, **encode_args), dtype=np.float32)
    return masked, plain


def run(args: argparse.Namespace) -> dict[str, object]:
    data_path = args.data_dir / "clue_user_day_dataset.npz"
    features_path = args.data_dir / "clue_profile_features.npy"
    graph_path = args.data_dir / "clue_prior_graphs.npz"
    profile_metadata_path = args.data_dir / "clue_profile_prior_metadata.json"
    dataset = np.load(data_path, allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active_mask = dataset["active_mask"].astype(bool)
    event_id = dataset["event_id"].astype(np.int16)
    train_start = int(dataset["train_start_index"])
    train_end = int(dataset["train_end_index"])
    eval_start = int(dataset["eval_start_index"])
    eval_end = int(dataset["eval_end_index"])
    user_count = len(users)
    if train_start != 0:
        raise ValueError("the current CLUE dataset must start at index 0")
    if eval_start != train_end + 1:
        raise ValueError("evaluation period must immediately follow training")

    profile_features = np.load(features_path)
    profile_metadata = json.loads(
        profile_metadata_path.read_text(encoding="utf-8")
    )
    if [str(user) for user in profile_metadata["users_order"]] != users.tolist():
        raise ValueError("profile user order does not match dataset user order")
    if len(profile_features) != user_count:
        raise ValueError("profile feature count does not match dataset users")
    soft_labels, soft_metadata = profile_soft_labels(profile_features)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "soft_instance_labels.npy", soft_labels)
    (output_dir / "soft_label_metadata.json").write_text(
        json.dumps(soft_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    base_steps = args.base_steps or int(
        np.ceil(user_count / args.batch_size) * args.epochs
    )
    graph_users, targets = load_graph_edges(graph_path)
    if graph_users != user_count:
        raise ValueError("graph user count does not match dataset users")
    uniform_labels = np.eye(user_count, dtype=np.float32)
    for left, right in targets["A3"]:
        uniform_labels[left, right] = 1.0
        uniform_labels[right, left] = 1.0

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    summary: dict[str, object] = {
        "data": str(data_path),
        "users": int(user_count),
        "train_end_index": train_end,
        "eval_start_index": eval_start,
        "eval_end_index": eval_end,
        "base_steps": base_steps,
        "seeds": args.seeds,
        "soft_label_metadata": soft_metadata,
        "variants": {},
    }

    for seed in args.seeds:
        coverage = build_common_coverage_table(
            user_count=user_count,
            target_edge_sets=targets,
            seed=seed,
            stage=0,
            batch_size=args.batch_size,
            target_coverage=0.8,
            min_training_batches=base_steps,
            max_coverage_batches=10_000,
        )
        stage_batches = len(coverage.training_batches)
        if args.steps is not None and args.steps != stage_batches:
            raise ValueError(
                f"--steps={args.steps} does not match seed {seed} "
                f"coverage length {stage_batches}"
            )
        coverage_summary = coverage.to_summary()
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "coverage_summary.json").write_text(
            json.dumps(coverage_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        variant_summaries: dict[str, object] = {}
        for variant in args.variants:
            variant_coverage = coverage
            if variant == "A0-A3batch":
                variant_coverage = build_common_coverage_table(
                    user_count=user_count,
                    target_edge_sets={"A3": targets["A3"]},
                    seed=seed,
                    stage=0,
                    batch_size=args.batch_size,
                    target_coverage=0.8,
                    min_training_batches=base_steps,
                    max_coverage_batches=10_000,
                )
            elif variant == "A1-Cov-A5a":
                variant_coverage = build_common_coverage_table(
                    user_count=user_count,
                    target_edge_sets={"A5a": targets["A5a"]},
                    seed=seed,
                    stage=0,
                    batch_size=args.batch_size,
                    target_coverage=0.8,
                    min_training_batches=base_steps,
                    max_coverage_batches=10_000,
                )
            variant_dir = seed_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            batch_table = choose_batch_table(
                variant=variant,
                user_count=user_count,
                batch_size=args.batch_size,
                seed=seed,
                base_steps=base_steps,
                coverage=variant_coverage,
            )
            (variant_dir / "batch_table.json").write_text(
                json.dumps(
                    [list(batch) for batch in batch_table],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            train_summary = train_variant(
                variant=variant,
                seed=seed,
                train_values=values[:, : train_end + 1],
                soft_labels=soft_labels,
                batch_table=batch_table,
                output_dir=variant_dir,
                device=device,
                batch_size=args.batch_size,
                tau_temp=args.tau_temp,
                uniform_labels=uniform_labels,
            )
            train_summary["coverage_summary"] = variant_coverage.to_summary()
            variant_summaries[variant] = train_summary

            if args.encode:
                model = CoverageSoftTS2Vec(
                    input_dims=values.shape[-1],
                    output_dims=64,
                    hidden_dims=64,
                    depth=5,
                    device=device,
                    lr=1e-3,
                    batch_size=args.batch_size,
                    lambda_=0.5,
                    tau_temp=args.tau_temp,
                    soft_instance=variant
                    in (
                        "A1-R0",
                        "A1-Cov",
                        "A1-Cov-A5a",
                        "A1-Cov-uniform",
                    ),
                    soft_temporal=False,
                )
                model.load(str(variant_dir / "model.pth"))
                masked, plain = encode_dataset(model, values, device=device)
                raw_error = np.abs(masked - plain).sum(axis=-1).astype(np.float32)
                train_mask = active_mask[:, : train_end + 1] & np.isfinite(
                    raw_error[:, : train_end + 1]
                )
                train_errors = raw_error[:, : train_end + 1][train_mask]
                if train_errors.size == 0:
                    raise RuntimeError("no active training errors for calibration")
                scores = robust_scores(train_errors, raw_error)
                np.savez_compressed(
                    variant_dir / "scores.npz",
                    users=users,
                    dates=dates,
                    raw_error=raw_error,
                    anomaly_score=scores.astype(np.float32),
                    event_id=event_id,
                    active_mask=active_mask,
                )
                train_summary["evaluation"] = evaluate_scores(
                    scores=scores,
                    event_id=event_id,
                    active_mask=active_mask,
                    eval_start=eval_start,
                    eval_end=eval_end,
                )
            (variant_dir / "summary.json").write_text(
                json.dumps(train_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        summary["variants"][str(seed)] = {
            "stage_batches": stage_batches,
            "coverage": coverage_summary,
            "models": variant_summaries,
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--base-steps", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--tau-temp", type=float, default=1.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
