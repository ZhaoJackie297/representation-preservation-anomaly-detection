from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score


CCF_B = Path(__file__).resolve().parents[1]
PHASE0 = CCF_B / "phase0"
REPO = Path(__file__).resolve().parents[3]
CERT_ROOT = Path(
    os.environ.get("RPAD_CERT_ROOT", REPO / "runtime" / "cert")
).resolve()
EXTERNAL_ROOT = Path(
    os.environ.get("RPAD_EXTERNAL_ROOT", CERT_ROOT / "external")
).resolve()
MODEL_ROOT = Path(
    os.environ.get(
        "RPAD_MODEL_ROOT",
        CERT_ROOT / "experiments" / "representation_preservation",
    )
).resolve()
if str(PHASE0) not in sys.path:
    sys.path.insert(0, str(PHASE0))

from coverage_soft_ts2vec import CoverageSoftTS2Vec  # noqa: E402


SEEDS = tuple(range(20261116, 20261126))
SIGNAL_TYPES = ("point", "sustained", "drift", "switch")
SIGNAL_DURATIONS = {
    "point": 1,
    "sustained": 7,
    "drift": 10,
    "switch": 7,
}
K_GRID = (5, 10)
COOLDOWN_DAYS = 7
CONTEXT_WINDOW = 5


@dataclass(frozen=True)
class Profile:
    data_path: Path
    model_root: Path
    model_layout: str
    train_end_key: str
    eval_start_key: str
    eval_end_key: str
    treatment: str
    control: str


PROFILES = {
    "r42-temporal": Profile(
        data_path=CERT_ROOT
        / "processed_r42"
        / "r4.2_sequence_dataset.npz",
        model_root=MODEL_ROOT
        / "e2_instance"
        / "missing_safe_v2_20261117",
        model_layout="variant/seed/model",
        train_end_key="train_end_index",
        eval_start_key="test_start_index",
        eval_end_key="test_end_index",
        treatment="T4_full_soft",
        control="T3_instance_soft",
    ),
    "r42-coverage": Profile(
        data_path=CERT_ROOT
        / "processed_r42"
        / "r4.2_sequence_dataset.npz",
        model_root=EXTERNAL_ROOT / "pilot_r42_10seed_matrix_v1",
        model_layout="seed/variant/model",
        train_end_key="train_end_index",
        eval_start_key="test_start_index",
        eval_end_key="test_end_index",
        treatment="A1-Cov",
        control="A0-official",
    ),
    "clue-coverage": Profile(
        data_path=EXTERNAL_ROOT
        / "clue_lds"
        / "processed_final"
        / "clue_user_day_dataset.npz",
        model_root=EXTERNAL_ROOT / "clue_dual_end_cap256_10seed_v1",
        model_layout="seed/variant/model",
        train_end_key="train_end_index",
        eval_start_key="eval_start_index",
        eval_end_key="eval_end_index",
        treatment="A1-Cov",
        control="A0-official",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic controlled-anomaly benchmark on real "
            "CERT or CLUE normal logs and compare temporal/covariance "
            "variants with point, context, and iForest signals."
        )
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        required=True,
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[SEEDS[0]],
    )
    parser.add_argument(
        "--label-seed",
        type=int,
        default=20261119,
        help="Seed used for users, event starts, and iForest injection.",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--signal-types",
        nargs="+",
        choices=SIGNAL_TYPES,
        default=list(SIGNAL_TYPES),
    )
    parser.add_argument(
        "--background-users",
        type=int,
        default=128,
        help="Users retained for the controlled evaluation.",
    )
    parser.add_argument(
        "--event-users",
        type=int,
        default=32,
        help="Users that receive one synthetic event per scenario.",
    )
    parser.add_argument(
        "--context-days",
        type=int,
        default=64,
        help="Real history retained before the evaluation split.",
    )
    parser.add_argument(
        "--calibration-days",
        type=int,
        default=64,
        help="Training tail used for score calibration.",
    )
    parser.add_argument(
        "--signal-channels",
        type=int,
        default=12,
        help="Highest-variance channels perturbed by each injected event.",
    )
    parser.add_argument(
        "--iforest-trees",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def model_path(
    root: Path,
    layout: str,
    seed: int,
    variant: str,
) -> Path:
    if layout == "variant/seed/model":
        path = root / variant / f"seed_{seed}" / "model.pth"
    elif layout == "seed/variant/model":
        path = root / f"seed_{seed}" / variant / "model.pth"
    else:
        raise ValueError(f"unknown model layout: {layout}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_model(path: Path, input_dims: int, device: str) -> CoverageSoftTS2Vec:
    model = CoverageSoftTS2Vec(
        input_dims=input_dims,
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=device,
        batch_size=16,
        lambda_=0.5,
        soft_instance=True,
        soft_temporal=False,
    )
    # The historical T3/T4 checkpoints use the same TSEncoder state layout.
    model.load(str(path))
    return model


def encode_pair(
    model: CoverageSoftTS2Vec,
    values: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    kwargs = {
        "causal": True,
        "sliding_length": 1,
        "sliding_padding": 30,
        "batch_size": batch_size,
    }
    masked = np.asarray(
        model.encode(values, mask="mask_last", **kwargs),
        dtype=np.float32,
    )
    plain = np.asarray(
        model.encode(values, **kwargs),
        dtype=np.float32,
    )
    return masked, plain


def robust_transform(
    train: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    finite = np.asarray(train)[np.isfinite(train)]
    if len(finite) == 0:
        raise ValueError("robust calibration has no finite values")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(np.std(finite))
    if scale < 1e-8:
        scale = 1.0
    output = (values - median) / scale
    return np.clip(output, 0.0, None).astype(np.float32)


def causal_context(
    values: np.ndarray,
    active: np.ndarray,
    window: int = CONTEXT_WINDOW,
) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float32)
    for user in range(values.shape[0]):
        history: list[float] = []
        for time in range(values.shape[1]):
            if not active[user, time] or not np.isfinite(values[user, time]):
                history.clear()
                continue
            history.append(float(values[user, time]))
            if len(history) > window:
                history.pop(0)
            output[user, time] = float(np.mean(history))
    return output


def combine_scores(
    point: np.ndarray,
    context: np.ndarray,
) -> np.ndarray:
    return (0.5 * point + 0.5 * context).astype(np.float32)


def select_metric_channels(
    values: np.ndarray,
    active: np.ndarray,
    train_end: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_values = values[:, : train_end + 1]
    train_active = active[:, : train_end + 1]
    scales = []
    for channel in range(values.shape[-1]):
        selected = train_values[:, :, channel][train_active]
        finite = selected[np.isfinite(selected)]
        scales.append(float(np.std(finite)) if len(finite) else 0.0)
    scales_array = np.asarray(scales, dtype=np.float64)
    available = np.flatnonzero(
        np.isfinite(scales_array) & (scales_array > 1e-8)
    )
    if len(available) == 0:
        raise RuntimeError("no variable input channels")
    order = available[np.argsort(-scales_array[available], kind="stable")]
    channels = order[: min(count, len(order))]
    safe_scales = np.where(scales_array > 1e-8, scales_array, 1.0)
    return channels.astype(np.int64), safe_scales.astype(np.float32)


def eligible_starts(
    active_user: np.ndarray,
    *,
    eval_start: int,
    eval_end: int,
    duration: int,
) -> np.ndarray:
    candidates = np.flatnonzero(active_user[eval_start : eval_end + 1])
    candidates = candidates + eval_start
    lower = eval_start + max(3, duration // 2)
    upper = eval_end - max(3, duration // 2)
    return candidates[(candidates >= lower) & (candidates <= upper)]


def select_users_and_events(
    users: np.ndarray,
    active: np.ndarray,
    *,
    eval_start: int,
    eval_end: int,
    train_end: int,
    background_users: int,
    event_users: int,
    seed: int,
    max_duration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_eval = active[:, eval_start : eval_end + 1].sum(axis=1)
    active_train = active[:, : train_end + 1].sum(axis=1)
    eligible = np.flatnonzero((active_eval >= 20) & (active_train >= 20))
    if len(eligible) < event_users:
        raise RuntimeError(
            f"only {len(eligible)} eligible users, need {event_users}"
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(eligible)
    selected = np.sort(
        shuffled[: min(background_users, len(shuffled))]
    )
    targets = np.sort(
        rng.choice(selected, size=event_users, replace=False)
    )
    starts = np.full(len(targets), -1, dtype=np.int64)
    for position, user in enumerate(targets):
        candidates = eligible_starts(
            active[user],
            eval_start=eval_start,
            eval_end=eval_end,
            duration=max_duration,
        )
        if len(candidates) == 0:
            raise RuntimeError(f"user {users[user]} has no valid event start")
        starts[position] = int(rng.choice(candidates))
    return selected, targets, starts


def inject_signal(
    values: np.ndarray,
    active: np.ndarray,
    targets: np.ndarray,
    starts: np.ndarray,
    *,
    signal_type: str,
    strength: float,
    channels: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    modified = values.copy()
    labels = np.zeros(active.shape, dtype=bool)
    event_id = np.full(active.shape, -1, dtype=np.int32)
    rows: list[dict[str, object]] = []
    duration = SIGNAL_DURATIONS[signal_type]
    donor_order = np.roll(np.arange(len(targets)), -1)
    for event_index, (user, start) in enumerate(zip(targets, starts)):
        end = min(start + duration - 1, active.shape[1] - 1)
        positive_days = 0
        for time in range(start, end + 1):
            if not active[user, time]:
                continue
            if signal_type == "point":
                delta = strength
            elif signal_type == "sustained":
                delta = strength
            elif signal_type == "drift":
                delta = strength * (time - start + 1) / duration
            else:
                delta = strength
            if signal_type == "switch":
                donor = targets[donor_order[event_index]]
                donor_values = values[donor, time, channels]
                current = modified[user, time, channels]
                blend = 1.0 - math.exp(-max(delta, 0.0))
                modified[user, time, channels] = (
                    (1.0 - blend) * np.nan_to_num(current)
                    + blend * np.nan_to_num(donor_values)
                )
            else:
                scale = scales[channels] * delta
                current = np.nan_to_num(
                    modified[user, time, channels],
                    nan=0.0,
                )
                modified[user, time, channels] = current + scale
            labels[user, time] = True
            event_id[user, time] = event_index
            positive_days += 1
        rows.append(
            {
                "event_index": event_index,
                "user_index": int(user),
                "start": int(start),
                "end": int(end),
                "signal_type": signal_type,
                "strength": float(strength),
                "positive_days": int(positive_days),
            }
        )
    return modified, labels, event_id, rows


def build_alerts(
    score: np.ndarray,
    active: np.ndarray,
    k: int,
    cooldown_days: int = COOLDOWN_DAYS,
) -> np.ndarray:
    alerts = np.zeros(score.shape, dtype=np.uint8)
    last_alert = np.full(score.shape[0], -10000, dtype=np.int64)
    for time in range(score.shape[1]):
        available = active[:, time] & (
            time - last_alert >= cooldown_days
        )
        candidates = np.flatnonzero(
            available & np.isfinite(score[:, time])
        )
        if len(candidates) == 0:
            continue
        ranked = candidates[
            np.lexsort((candidates, -score[candidates, time]))
        ][:k]
        alerts[ranked, time] = 1
        last_alert[ranked] = time
    return alerts


def event_metric(
    alerts: np.ndarray,
    events: pd.DataFrame,
    active: np.ndarray,
    start_index: int,
    end_index: int,
) -> tuple[float, list[float | None]]:
    hits = 0
    delays: list[float | None] = []
    for event in events.itertuples(index=False):
        user = int(event.user_index)
        start = int(event.start)
        end = int(event.end)
        candidate_times = np.flatnonzero(
            alerts[user, start : end + 1]
        )
        if len(candidate_times):
            first = start + int(candidate_times[0])
            hits += 1
            delays.append(float(first - start))
        else:
            delays.append(None)
    return hits / max(len(events), 1), delays


def point_metric(
    score: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
    start_index: int,
    end_index: int,
) -> tuple[float, float]:
    split_score = score[:, start_index : end_index + 1]
    split_labels = labels[:, start_index : end_index + 1]
    split_active = active[:, start_index : end_index + 1]
    valid = split_active & np.isfinite(split_score)
    y_true = split_labels[valid].astype(np.int8)
    y_score = split_score[valid]
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    return (
        float(average_precision_score(y_true, y_score)),
        float(roc_auc_score(y_true, y_score)),
    )


def evaluate_signal(
    score: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
    events: pd.DataFrame,
    *,
    start_index: int,
    end_index: int,
) -> dict[str, object]:
    pr_auc, roc_auc = point_metric(
        score,
        labels,
        active,
        start_index,
        end_index,
    )
    output: dict[str, object] = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }
    for k in K_GRID:
        alerts = build_alerts(score, active, k)
        recall, delays = event_metric(
            alerts,
            events,
            active,
            start_index,
            end_index,
        )
        observed_delays = [value for value in delays if value is not None]
        output[f"event_recall_k{k}"] = recall
        output[f"median_delay_k{k}"] = (
            float(np.median(observed_delays))
            if observed_delays
            else None
        )
        output[f"alerts_k{k}"] = int(alerts.sum())
    return output


def score_components(
    raw_error: np.ndarray,
    train_raw: np.ndarray,
    train_active: np.ndarray,
    active: np.ndarray,
) -> dict[str, np.ndarray]:
    point = robust_transform(train_raw[train_active], raw_error)
    context_raw = causal_context(raw_error, active)
    train_context_raw = causal_context(
        train_raw,
        train_active,
    )
    context = robust_transform(
        train_context_raw[np.isfinite(train_context_raw)],
        context_raw,
    )
    combined = combine_scores(point, context)
    return {
        "point": point,
        "context": context,
        "combined": combined,
    }


def iforest_components(
    train_values: np.ndarray,
    train_active: np.ndarray,
    eval_values: np.ndarray,
    eval_active: np.ndarray,
    *,
    seed: int,
    trees: int,
) -> dict[str, np.ndarray]:
    train_rows = train_values[train_active]
    if len(train_rows) < 10:
        raise RuntimeError("too few active training rows for iForest")
    model = IsolationForest(
        n_estimators=trees,
        max_samples="auto",
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    ).fit(train_rows)
    train_score = -model.score_samples(train_rows)
    eval_lookup = np.full(eval_active.shape, np.nan, dtype=np.float32)
    rows = np.flatnonzero(eval_active.reshape(-1))
    flat = eval_values.reshape(-1, eval_values.shape[-1])
    scores = -model.score_samples(flat[rows])
    eval_lookup.reshape(-1)[rows] = scores
    point = robust_transform(train_score, eval_lookup)
    point[~eval_active] = np.nan
    context = causal_context(point, eval_active)
    combined = combine_scores(point, context)
    return {
        "iforest-point": point,
        "iforest-context": context,
        "iforest-combined": combined,
    }


def add_event_metadata(
    rows: list[dict[str, object]],
    users: np.ndarray,
    dates: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["user"] = frame["user_index"].map(
        {index: str(user) for index, user in enumerate(users)}
    )
    frame["start_date"] = frame["start"].map(
        {index: str(date) for index, date in enumerate(dates)}
    )
    frame["end_date"] = frame["end"].map(
        {index: str(date) for index, date in enumerate(dates)}
    )
    return frame


def paired_rows(
    metrics: pd.DataFrame,
    *,
    profile: Profile,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_pair(
        treatment: str,
        control: str,
        comparison: str,
        metric_names: Iterable[str],
    ) -> None:
        left = metrics[metrics["method"] == treatment]
        right = metrics[metrics["method"] == control]
        merged = left.merge(
            right,
            on=[
                "seed",
                "signal_type",
                "strength",
                "split",
                "score_type",
            ],
            suffixes=("_treatment", "_control"),
        )
        for metric in metric_names:
            values = merged[f"{metric}_treatment"] - merged[
                f"{metric}_control"
            ]
            for keys, delta in zip(
                merged[
                    [
                        "seed",
                        "signal_type",
                        "strength",
                        "split",
                        "score_type",
                    ]
                ].itertuples(index=False, name=None),
                values,
            ):
                if not np.isfinite(delta):
                    continue
                rows.append(
                    {
                        "comparison": comparison,
                        "seed": int(keys[0]),
                        "signal_type": str(keys[1]),
                        "strength": float(keys[2]),
                        "split": str(keys[3]),
                        "score_type": str(keys[4]),
                        "metric": metric,
                        "delta": float(delta),
                    }
                )

    add_pair(
        profile.treatment,
        profile.control,
        f"{profile.treatment}_minus_{profile.control}",
        ("pr_auc", "roc_auc", "event_recall_k5", "event_recall_k10"),
    )
    # The model-level context ablation compares different score arrays from
    # the same checkpoint, so it is aggregated directly below instead of
    # through the model pair above.
    context_comparison = []
    for seed in sorted(metrics["seed"].unique()):
        for signal_type in sorted(metrics["signal_type"].unique()):
            for strength in sorted(metrics["strength"].unique()):
                for base_method in (
                    profile.control,
                    profile.treatment,
                    "iForest",
                ):
                    subset = metrics[
                        (metrics["seed"] == seed)
                        & (metrics["signal_type"] == signal_type)
                        & (metrics["strength"] == strength)
                        & (metrics["base_method"] == base_method)
                    ].set_index("score_type")
                    if not {"point", "context", "combined"}.issubset(
                        subset.index
                    ):
                        continue
                    for metric in (
                        "pr_auc",
                        "roc_auc",
                        "event_recall_k5",
                        "event_recall_k10",
                    ):
                        for score_type in ("context", "combined"):
                            context_comparison.append(
                                {
                                    "comparison": (
                                        f"{base_method}_"
                                        f"{score_type}_minus_point"
                                    ),
                                    "seed": int(seed),
                                    "signal_type": str(signal_type),
                                    "strength": float(strength),
                                    "split": "test",
                                    "score_type": score_type,
                                    "metric": metric,
                                    "delta": float(
                                        subset.loc[score_type, metric]
                                        - subset.loc["point", metric]
                                    ),
                                }
                            )
    if context_comparison:
        rows.extend(context_comparison)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summary = (
        frame.groupby(
            [
                "comparison",
                "signal_type",
                "strength",
                "score_type",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            mean_delta=("delta", "mean"),
            positive_seeds=("delta", lambda values: int((values > 0).sum())),
            negative_seeds=("delta", lambda values: int((values < 0).sum())),
            n=("delta", "size"),
        )
    )
    return summary


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    data_path = args.data or profile.data_path
    model_root = args.model_root or profile.model_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    with np.load(data_path, allow_pickle=False) as dataset:
        original = dataset["values"].astype(np.float32)
        users = dataset["users"].astype(str)
        dates = dataset["dates"].astype(str)
        if "active_mask" not in dataset.files:
            raise KeyError("dataset has no active_mask")
        active_full = dataset["active_mask"].astype(bool)
        if active_full.ndim == 3:
            active_full = active_full[:, :, 0]
        train_end = int(dataset[profile.train_end_key])
        eval_start = int(dataset[profile.eval_start_key])
        eval_end = int(dataset[profile.eval_end_key])

    crop_start = max(0, eval_start - args.context_days)
    crop_values = original[:, crop_start : eval_end + 1].copy()
    crop_active = active_full[:, crop_start : eval_end + 1].copy()
    local_eval_start = eval_start - crop_start
    local_eval_end = eval_end - crop_start
    max_duration = max(SIGNAL_DURATIONS.values())
    channels, scales = select_metric_channels(
        original,
        active_full,
        train_end,
        args.signal_channels,
    )

    selected_users, targets, starts = select_users_and_events(
        users,
        active_full,
        eval_start=eval_start,
        eval_end=eval_end,
        train_end=train_end,
        background_users=args.background_users,
        event_users=args.event_users,
        seed=args.label_seed,
        max_duration=max_duration,
    )
    selected_crop = crop_values[selected_users]
    selected_active = crop_active[selected_users]
    calibration_start = max(0, train_end - args.calibration_days + 1)
    calibration_values = original[
        :, calibration_start : train_end + 1
    ]
    calibration_active = active_full[
        :, calibration_start : train_end + 1
    ]

    manifest = {
        "profile": args.profile,
        "data": str(data_path),
        "model_root": str(model_root),
        "seeds": args.seeds,
        "label_seed": args.label_seed,
        "signal_types": args.signal_types,
        "strengths": args.strengths,
        "k_grid": list(K_GRID),
        "cooldown_days": COOLDOWN_DAYS,
        "context_window": CONTEXT_WINDOW,
        "background_users": int(len(selected_users)),
        "event_users": int(len(targets)),
        "selected_users": [str(users[index]) for index in selected_users],
        "target_users": [str(users[index]) for index in targets],
        "channels": [str(value) for value in channels],
        "crop_start": int(crop_start),
        "eval_start": int(eval_start),
        "eval_end": int(eval_end),
        "train_end": int(train_end),
        "claims": {
            "external_validation": False,
            "dataset_role": (
                "controlled anomaly-signal benchmark built on real normal "
                "behavior; not an external dataset"
            ),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metric_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        paths = {
            profile.control: model_path(
                model_root,
                profile.model_layout,
                seed,
                profile.control,
            ),
            profile.treatment: model_path(
                model_root,
                profile.model_layout,
                seed,
                profile.treatment,
            ),
        }
        calibration_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for variant, path in paths.items():
            model = load_model(path, original.shape[-1], device)
            masked_cal, plain_cal = encode_pair(
                model,
                calibration_values,
                batch_size=128,
            )
            calibration_raw = np.abs(masked_cal - plain_cal).sum(axis=-1)
            calibration_cache[variant] = (
                calibration_raw.astype(np.float32),
                np.asarray(calibration_active, dtype=bool),
            )
            print(
                f"profile={args.profile} seed={seed} "
                f"variant={variant} calibration=done",
                flush=True,
            )

            for signal_type in args.signal_types:
                for strength in args.strengths:
                    local_targets = np.searchsorted(
                        selected_users,
                        targets,
                    )
                    local_starts = starts - crop_start
                    modified, labels, event_ids, event_rows = inject_signal(
                        selected_crop,
                        selected_active,
                        local_targets,
                        local_starts,
                        signal_type=signal_type,
                        strength=float(strength),
                        channels=channels,
                        scales=scales,
                    )
                    local_events = pd.DataFrame(event_rows)
                    global_rows = []
                    for row in event_rows:
                        local_user = int(row["user_index"])
                        global_rows.append(
                            {
                                **row,
                                "user_index": int(selected_users[local_user]),
                                "start": int(row["start"] + crop_start),
                                "end": int(row["end"] + crop_start),
                                "seed": seed,
                            }
                        )
                    event_frames.append(
                        add_event_metadata(global_rows, users, dates)
                    )

                    masked, plain = encode_pair(
                        model,
                        modified,
                        batch_size=128,
                    )
                    raw_error = np.abs(masked - plain).sum(axis=-1)
                    train_raw, train_active = calibration_cache[variant]
                    # Recalibrate on the same active calibration cells.
                    components = score_components(
                        raw_error,
                        train_raw,
                        train_active,
                        selected_active,
                    )
                    for score_type, score in components.items():
                        metrics = evaluate_signal(
                            score,
                            labels,
                            selected_active,
                            local_events,
                            start_index=local_eval_start,
                            end_index=local_eval_end,
                        )
                        metric_rows.append(
                            {
                                "seed": seed,
                                "dataset": args.profile,
                                "base_method": variant,
                                "method": variant,
                                "score_type": score_type,
                                "signal_type": signal_type,
                                "strength": float(strength),
                                "split": "test",
                                **metrics,
                            }
                        )
                    np.savez_compressed(
                        args.output_dir
                        / (
                            f"seed_{seed}_{variant}_{signal_type}"
                            f"_s{strength:g}.npz"
                        ),
                        point=components["point"],
                        context=components["context"],
                        combined=components["combined"],
                        labels=labels,
                        active=selected_active,
                        event_id=event_ids,
                    )
                    print(
                        f"profile={args.profile} seed={seed} "
                        f"variant={variant} signal={signal_type} "
                        f"strength={strength:g} scored",
                        flush=True,
                    )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_values = original[
            selected_users,
            calibration_start : train_end + 1,
        ]
        train_active = active_full[
            selected_users,
            calibration_start : train_end + 1,
        ]
        for signal_type in args.signal_types:
            for strength in args.strengths:
                local_targets = np.searchsorted(selected_users, targets)
                local_starts = starts - crop_start
                modified, labels, event_ids, event_rows = inject_signal(
                    selected_crop,
                    selected_active,
                    local_targets,
                    local_starts,
                    signal_type=signal_type,
                    strength=float(strength),
                    channels=channels,
                    scales=scales,
                )
                local_events = pd.DataFrame(event_rows)
                components = iforest_components(
                    train_values,
                    train_active,
                    modified,
                    selected_active,
                    seed=seed,
                    trees=args.iforest_trees,
                )
                for score_name, score in components.items():
                    score_type = score_name.removeprefix("iforest-")
                    metrics = evaluate_signal(
                        score,
                        labels,
                        selected_active,
                        local_events,
                        start_index=local_eval_start,
                        end_index=local_eval_end,
                    )
                    metric_rows.append(
                        {
                            "seed": seed,
                            "dataset": args.profile,
                            "base_method": "iForest",
                            "method": "iForest",
                            "score_type": score_type,
                            "signal_type": signal_type,
                            "strength": float(strength),
                            "split": "test",
                            **metrics,
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        args.output_dir / "metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        metrics.groupby(
            [
                "base_method",
                "score_type",
                "signal_type",
                "strength",
            ],
            as_index=False,
        )
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_k5_mean=("event_recall_k5", "mean"),
            event_recall_k10_mean=("event_recall_k10", "mean"),
            median_delay_k10_mean=("median_delay_k10", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(
        args.output_dir / "summary.csv",
        index=False,
        encoding="utf-8",
    )
    pairs = paired_rows(metrics, profile=profile)
    pairs.to_csv(
        args.output_dir / "paired_comparisons.csv",
        index=False,
        encoding="utf-8",
    )
    events_all = pd.concat(
        event_frames,
        ignore_index=True,
    ).drop_duplicates(
        subset=[
            "seed",
            "event_index",
            "user_index",
            "start",
            "end",
            "signal_type",
            "strength",
        ]
    )
    events_all.to_csv(
        args.output_dir / "events.csv",
        index=False,
        encoding="utf-8",
    )
    result = {
        "manifest": manifest,
        "methods": sorted(metrics["base_method"].unique().tolist()),
        "score_types": sorted(metrics["score_type"].unique().tolist()),
        "summary_rows": summary.to_dict(orient="records"),
        "paired_comparison_rows": pairs.to_dict(orient="records"),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
