from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 30
MASK_RATIO = 0.30
K_INTERVENTIONS = 6
CLEAN_WEIGHT = 0.25
SCALE_LOW = -0.10
SCALE_HIGH = 0.10
CHUNK_USERS = 256

BEHAVIOR_GROUPS = {
    "logon": list(range(1, 8)),
    "device": [8, 9],
    "file": list(range(10, 14)),
    "email": list(range(14, 20)),
    "http": list(range(20, 24)),
}
BEHAVIOR_FEATURES = sorted(
    {index for indices in BEHAVIOR_GROUPS.values() for index in indices}
)
OPS = ("drop", "median_replace", "shuffle", "scale_jitter")

sys.path.insert(0, str(ROOT))
import e17_multigranular_mask as e17  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    load_r42,
    load_r52,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free channel-intervention variance scoring for "
            "frozen TS2Vec."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["r42", "r52", "all"],
        default="r42",
    )
    parser.add_argument(
        "--channel-mode",
        choices=["feature", "group"],
        default="feature",
    )
    parser.add_argument(
        "--score-head",
        choices=["mask_l1", "latent_distance"],
        default="mask_l1",
    )
    parser.add_argument("--k-interventions", type=int, default=K_INTERVENTIONS)
    parser.add_argument("--mask-ratio", type=float, default=MASK_RATIO)
    parser.add_argument("--clean-weight", type=float, default=CLEAN_WEIGHT)
    parser.add_argument(
        "--intervention-days",
        type=int,
        default=1,
        help=(
            "Perturb only the last N days of each causal window. "
            "The default matches the last-point TS2Vec score."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument(
        "--calibration-stride",
        type=int,
        default=3,
        help="Use every N training days for signal calibration.",
    )
    parser.add_argument(
        "--chunk-users",
        type=int,
        default=CHUNK_USERS,
    )
    parser.add_argument(
        "--smoke-days",
        type=int,
        default=0,
        help="Only score the first N evaluation days for a smoke test.",
    )
    return parser.parse_args()


def dataset_config(dataset_name: str) -> dict:
    if dataset_name == "r42":
        return {
            "processed": ROOT / "processed_r42",
            "dataset": (
                ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
            ),
            "model": ROOT / "processed_r42" / "r4.2_ts2vec_model.pth",
            "clean_raw": (
                OUT / "e17_r42_M0_masked_raw.npy"
            ),
            "clean_quantile": (
                OUT / "e17_r42_M0_masked_quantile.npy"
            ),
            "latent_raw": OUT / "e17_r42_M0_latent_raw.npy",
            "latent_quantile": (
                OUT / "e17_r42_M0_latent_quantile.npy"
            ),
            "plain": OUT / "e17_r42_M0_plain.npy",
            "data_loader": load_r42,
            "external": False,
        }
    return {
        "processed": ROOT / "processed_r52",
        "dataset": (
            ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
        ),
        "model": ROOT / "processed_r52" / "r5.2_ts2vec_model.pth",
        "clean_raw": OUT / "e17_r52_M0_masked_raw.npy",
        "clean_quantile": (
            OUT / "e17_r52_M0_masked_quantile.npy"
        ),
        "latent_raw": OUT / "e17_r52_M0_latent_raw.npy",
        "latent_quantile": (
            OUT / "e17_r52_M0_latent_quantile.npy"
        ),
        "plain": OUT / "e17_r52_M0_plain.npy",
        "data_loader": load_r52,
        "external": True,
    }


def channel_layout(
    mode: str,
    device,
    mask_ratio: float,
) -> dict:
    if mode == "feature":
        channel_indices = torch.tensor(
            BEHAVIOR_FEATURES,
            dtype=torch.long,
            device=device,
        )
        unit_to_channels = [
            torch.tensor([index], dtype=torch.long, device=device)
            for index in BEHAVIOR_FEATURES
        ]
    else:
        channel_indices = torch.tensor(
            BEHAVIOR_FEATURES,
            dtype=torch.long,
            device=device,
        )
        unit_to_channels = [
            torch.tensor(indices, dtype=torch.long, device=device)
            for indices in BEHAVIOR_GROUPS.values()
        ]
    return {
        "behavior_indices": channel_indices,
        "unit_to_channels": unit_to_channels,
        "n_units": len(unit_to_channels),
        "n_selected": max(
            1,
            int(round(mask_ratio * len(unit_to_channels))),
        ),
    }


def training_medians(
    values: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> torch.Tensor:
    train_values = values[:, : train_end + 1, :]
    train_active = active_mask[:, : train_end + 1]
    rows = train_values[train_active]
    if len(rows) == 0:
        raise RuntimeError("No active training rows.")
    return torch.from_numpy(
        np.median(rows, axis=0).astype(np.float32)
    )


def latent_profile(
    plain: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_active = active_mask[:, : train_end + 1]
    train_representations = plain[:, : train_end + 1]
    finite = np.isfinite(train_representations).all(axis=2)
    train_active = train_active & finite
    global_rows = train_representations[train_active]
    if len(global_rows) == 0:
        raise RuntimeError("No finite training representations.")
    global_center = global_rows.mean(axis=0)
    centers = np.full(
        (plain.shape[0], plain.shape[-1]),
        np.nan,
        dtype=np.float64,
    )
    residuals = []
    for user_index in range(plain.shape[0]):
        valid = np.flatnonzero(train_active[user_index])
        if len(valid):
            centers[user_index] = train_representations[
                user_index, valid
            ].mean(axis=0)
            residuals.append(
                train_representations[user_index, valid]
                - centers[user_index]
            )
        else:
            centers[user_index] = global_center
    centers = np.where(np.isfinite(centers), centers, global_center)
    residual_rows = np.concatenate(residuals, axis=0)
    scale = np.sqrt(np.mean(np.square(residual_rows), axis=0))
    scale = np.where(
        np.isfinite(scale) & (scale >= 1e-8),
        scale,
        1.0,
    )
    return (
        torch.from_numpy(centers.astype(np.float32)),
        torch.from_numpy(scale.astype(np.float32)),
    )


def build_windows(
    values: torch.Tensor,
    user_indices: torch.Tensor,
    time_index: int,
    max_len: int,
) -> torch.Tensor:
    left = max(0, time_index - WINDOW_DAYS)
    right = time_index + 1
    span = right - left
    source = values[user_indices, left:right, :]
    window = torch.full(
        (len(user_indices), max_len, values.shape[-1]),
        float("nan"),
        dtype=values.dtype,
        device=values.device,
    )
    window[:, -span:, :] = source
    return window


def selected_feature_masks(
    batch_size: int,
    layout: dict,
    generator: torch.Generator,
    device,
) -> torch.Tensor:
    n_units = int(layout["n_units"])
    n_selected = int(layout["n_selected"])
    choices = torch.rand(
        (batch_size, n_units),
        generator=generator,
        device=device,
    )
    selected_units = torch.topk(
        choices,
        n_selected,
        dim=1,
    ).indices
    unit_mask = torch.zeros(
        (batch_size, n_units),
        dtype=torch.bool,
        device=device,
    )
    unit_mask.scatter_(1, selected_units, True)

    feature_mask = torch.zeros(
        (batch_size, 27),
        dtype=torch.bool,
        device=device,
    )
    for unit_index, indices in enumerate(layout["unit_to_channels"]):
        feature_mask[:, indices] = unit_mask[:, unit_index : unit_index + 1]
    return feature_mask


def apply_single_intervention(
    base: torch.Tensor,
    train_median: torch.Tensor,
    layout: dict,
    *,
    op: str,
    intervention_days: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size, length, feature_dims = base.shape
    device = base.device
    feature_mask = selected_feature_masks(
        batch_size,
        layout,
        generator,
        device,
    )
    missing_rows = base.isnan().all(dim=-1)
    view = base.clone()
    start = max(0, length - intervention_days)
    recent = slice(start, length)

    if op == "drop":
        recent_view = view[:, recent, :]
        recent_view[
            feature_mask.unsqueeze(1).expand_as(recent_view)
        ] = 0.0
    elif op == "median_replace":
        recent_view = view[:, recent, :]
        median_view = train_median.view(1, 1, -1).expand_as(
            recent_view
        )
        view[:, recent, :] = torch.where(
            feature_mask.unsqueeze(1),
            median_view,
            recent_view,
        )
    elif op == "shuffle":
        if batch_size == 1:
            source = torch.zeros(
                (1,),
                dtype=torch.long,
                device=device,
            )
        else:
            offset = torch.randint(
                1,
                batch_size,
                (batch_size,),
                generator=generator,
                device=device,
            )
            source = (
                torch.arange(batch_size, device=device) + offset
            ) % batch_size
        shuffled = base[source][:, recent, :]
        recent_view = view[:, recent, :]
        view[:, recent, :] = torch.where(
            feature_mask.unsqueeze(1),
            shuffled,
            recent_view,
        )
    elif op == "scale_jitter":
        factors = (
            torch.rand(
                (batch_size, feature_dims),
                generator=generator,
                device=device,
            )
            * (SCALE_HIGH - SCALE_LOW)
            + SCALE_LOW
            + 1.0
        )
        recent_view = view[:, recent, :]
        view[:, recent, :] = torch.where(
            feature_mask.unsqueeze(1),
            recent_view * factors.unsqueeze(1),
            recent_view,
        )
    else:
        raise ValueError(f"Unknown op: {op}")

    view[missing_rows.unsqueeze(-1).expand_as(view)] = float("nan")
    return view


def build_intervention_views(
    base: torch.Tensor,
    train_median: torch.Tensor,
    layout: dict,
    *,
    k_interventions: int,
    intervention_days: int,
    generator: torch.Generator,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, list[str]]:
    batch_size = base.shape[0]
    views = []
    operations = []
    for _ in range(k_interventions):
        op = OPS[int(rng.integers(0, len(OPS)))]
        operations.append(op)
        views.append(
            apply_single_intervention(
                base,
                train_median,
                layout,
                op=op,
                intervention_days=intervention_days,
                generator=generator,
            )
        )
    stacked = torch.stack(views, dim=1)
    return (
        stacked.reshape(
            batch_size * k_interventions,
            base.shape[1],
            base.shape[2],
        ),
        operations,
    )


def score_intervention_views(
    student: torch.nn.Module,
    views: torch.Tensor,
    *,
    score_head: str,
    user_indices: torch.Tensor | None = None,
    centers: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.inference_mode():
        if score_head == "mask_l1":
            full = student(views.clone(), mask="all_true")
            masked = student(views.clone(), mask="mask_last")
            return (masked[:, -1] - full[:, -1]).abs().sum(dim=-1)
        if score_head == "latent_distance":
            if (
                user_indices is None
                or centers is None
                or scale is None
            ):
                raise ValueError(
                    "Latent scoring requires centers and scale."
                )
            representation = student(
                views.clone(),
                mask="all_true",
            )[:, -1]
            standardized = (
                representation - centers[user_indices]
            ) / scale
            return torch.sqrt(
                standardized.square().mean(dim=-1)
            )
        raise ValueError(f"Unknown score head: {score_head}")


def score_day(
    student: torch.nn.Module,
    values: torch.Tensor,
    active_mask: torch.Tensor,
    train_median: torch.Tensor,
    layout: dict,
    centers: torch.Tensor | None,
    scale: torch.Tensor | None,
    *,
    score_head: str,
    time_index: int,
    chunk_users: int,
    k_interventions: int,
    intervention_days: int,
    generator: torch.Generator,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
]:
    active_users = torch.nonzero(
        active_mask[:, time_index],
        as_tuple=False,
    ).flatten()
    variances = np.full(len(active_mask), np.nan, dtype=np.float32)
    clean_scores = np.full(len(active_mask), np.nan, dtype=np.float32)
    intervention_means = np.full(
        len(active_mask),
        np.nan,
        dtype=np.float32,
    )
    intervention_mins = np.full(
        len(active_mask),
        np.nan,
        dtype=np.float32,
    )
    op_counts = {name: 0 for name in OPS}

    for start in range(0, len(active_users), chunk_users):
        user_indices = active_users[start : start + chunk_users]
        windows = build_windows(
            values,
            user_indices,
            time_index,
            WINDOW_DAYS + 1,
        )
        if score_head == "latent_distance":
            clean_score = score_intervention_views(
                student,
                windows,
                score_head=score_head,
                user_indices=user_indices,
                centers=centers,
                scale=scale,
            ).cpu().numpy()
        else:
            clean_score = score_intervention_views(
                student,
                windows,
                score_head=score_head,
            ).cpu().numpy()
        views, operations = build_intervention_views(
            windows,
            train_median,
            layout,
            k_interventions=k_interventions,
            intervention_days=intervention_days,
            generator=generator,
            rng=rng,
        )
        repeated_users = user_indices.repeat_interleave(
            k_interventions
        )
        scores = score_intervention_views(
            student,
            views,
            score_head=score_head,
            user_indices=repeated_users,
            centers=centers,
            scale=scale,
        )
        scores = scores.reshape(len(user_indices), k_interventions)
        mean_score = scores.mean(dim=1).cpu().numpy()
        min_score = scores.min(dim=1).values.cpu().numpy()
        variance = scores.var(dim=1, unbiased=True).cpu().numpy()

        users_np = user_indices.cpu().numpy()
        variances[users_np] = variance
        clean_scores[users_np] = clean_score
        intervention_means[users_np] = mean_score
        intervention_mins[users_np] = min_score
        for op in operations:
            op_counts[op] += len(user_indices)
    return (
        clean_scores,
        intervention_means,
        intervention_mins,
        variances,
        op_counts,
    )


def finite_training_values(
    scores: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    mask = active_mask[:, : train_end + 1] & np.isfinite(
        scores[:, : train_end + 1]
    )
    values = scores[:, : train_end + 1][mask]
    if len(values) == 0:
        raise RuntimeError("No finite training scores.")
    return values.astype(np.float64)


def global_z_scores(
    scores: np.ndarray,
    train_end: int,
    active_mask: np.ndarray,
) -> np.ndarray:
    train_values = finite_training_values(
        scores,
        active_mask,
        train_end,
    )
    mean = float(train_values.mean())
    std = max(float(train_values.std()), 1e-8)
    return ((scores - mean) / std).astype(np.float32)


def per_user_quantiles_allow_missing(
    raw_scores: np.ndarray,
    active_mask: np.ndarray,
    train_end: int,
) -> np.ndarray:
    calibrated = np.full(raw_scores.shape, np.nan, dtype=np.float32)
    for user_index in range(raw_scores.shape[0]):
        train_values = raw_scores[
            user_index, : train_end + 1
        ][
            active_mask[user_index, : train_end + 1]
            & np.isfinite(raw_scores[user_index, : train_end + 1])
        ]
        if len(train_values) == 0:
            continue
        ordered = np.sort(train_values)
        n_values = len(ordered)
        finite = active_mask[user_index] & np.isfinite(
            raw_scores[user_index]
        )
        values = raw_scores[user_index, finite]
        ranks = np.searchsorted(
            ordered,
            values,
            side="right",
        )
        tail = (n_values - ranks + 1) / (n_values + 1)
        calibrated[user_index, finite] = np.maximum(
            0.0,
            -np.log(tail),
        ).astype(np.float32)
    return calibrated


def replace_nan_for_ranking(scores: np.ndarray) -> np.ndarray:
    result = scores.copy()
    result[~np.isfinite(result)] = -np.inf
    return result


def evaluate_variants(
    data: dict,
    *,
    external: bool,
    clean_quantile: np.ndarray,
    variance_quantile: np.ndarray,
    variance_raw: np.ndarray,
    intervention_mean: np.ndarray,
    intervention_min: np.ndarray,
    clean_raw: np.ndarray,
    clean_name: str,
    train_end: int,
    alpha: float,
) -> tuple[list[dict], np.ndarray]:
    clean_z = global_z_scores(
        clean_raw,
        train_end,
        data["active_mask"],
    )
    variance_z = global_z_scores(
        variance_raw,
        train_end,
        data["active_mask"],
    )
    full_score = alpha * clean_z + variance_z
    full_score = replace_nan_for_ranking(full_score)
    variance_only = replace_nan_for_ranking(variance_raw)
    variance_q = replace_nan_for_ranking(variance_quantile)
    intervention_mean_q = replace_nan_for_ranking(
        per_user_quantiles_allow_missing(
            intervention_mean,
            data["active_mask"],
            train_end,
        )
    )
    intervention_min_q = replace_nan_for_ranking(
        per_user_quantiles_allow_missing(
            intervention_min,
            data["active_mask"],
            train_end,
        )
    )
    lcb_raw = intervention_mean - np.sqrt(variance_raw)
    lcb_q = replace_nan_for_ranking(
        per_user_quantiles_allow_missing(
            lcb_raw,
            data["active_mask"],
            train_end,
        )
    )
    clean_q = replace_nan_for_ranking(clean_quantile)
    clean_z_rank = replace_nan_for_ranking(clean_z)

    variants = [
        (f"official_{clean_name}_quantile", clean_q),
        ("clean_raw_z", clean_z_rank),
        ("variance_only", variance_only),
        ("variance_quantile", variance_q),
        ("intervention_mean_q", intervention_mean_q),
        ("intervention_min_q", intervention_min_q),
        ("lower_confidence_bound_q", lcb_q),
        (
            f"full_alpha_{alpha:.2f}",
            full_score,
        ),
    ]
    rows = []
    for method, signal in variants:
        for k in (5, 10, 20):
            e17.K = k
            metrics = e17.evaluate_signal(
                data,
                signal.astype(np.float32),
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
    return rows, full_score.astype(np.float32)


def run_dataset(args: argparse.Namespace, dataset_name: str) -> dict:
    start_time = time.time()
    config = dataset_config(dataset_name)
    dataset = np.load(config["dataset"], allow_pickle=False)
    values_np = dataset["values"].astype(np.float32)
    active_np = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    time_steps = values_np.shape[1]

    data = config["data_loader"]()
    student, device = e17.load_base_student(
        config["model"],
        values_np.shape[-1],
    )
    student.eval()
    values = torch.from_numpy(values_np).to(device)
    active_mask = torch.from_numpy(active_np).to(device)
    train_median = training_medians(
        values_np,
        active_np,
        train_end,
    ).to(device)
    layout = channel_layout(
        args.channel_mode,
        device,
        args.mask_ratio,
    )

    if args.score_head == "latent_distance":
        plain = np.load(
            config["plain"],
            mmap_mode="r",
            allow_pickle=False,
        )
        centers, scale = latent_profile(
            plain,
            active_np,
            train_end,
        )
        centers = centers.to(device)
        scale = scale.to(device)
        clean_raw_path = config["latent_raw"]
        clean_quantile_path = config["latent_quantile"]
        clean_name = "latent"
    else:
        centers = None
        scale = None
        clean_raw_path = config["clean_raw"]
        clean_quantile_path = config["clean_quantile"]
        clean_name = "mask"

    clean_cached = np.load(
        clean_raw_path,
        allow_pickle=False,
    ).astype(np.float32)
    clean_quantile = np.load(
        clean_quantile_path,
        allow_pickle=False,
    ).astype(np.float32)
    variance_raw = np.full(values_np.shape[:2], np.nan, dtype=np.float32)
    intervention_mean = np.full_like(variance_raw, np.nan)
    intervention_min = np.full_like(variance_raw, np.nan)
    clean_recomputed = np.full_like(variance_raw, np.nan)

    calibration_start = min(WINDOW_DAYS, train_end)
    calibration_days = list(
        range(
            calibration_start,
            train_end + 1,
            max(args.calibration_stride, 1),
        )
    )
    if external := bool(config["external"]):
        evaluation_start = int(dataset["test_start_index"])
    else:
        evaluation_start = train_end + 1
    evaluation_days = list(range(evaluation_start, time_steps))
    if args.smoke_days > 0:
        evaluation_days = evaluation_days[: args.smoke_days]
    scoring_days = calibration_days + evaluation_days

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    op_totals = {name: 0 for name in OPS}

    print(
        f"{dataset_name}/{args.score_head}/{args.channel_mode}: "
        f"{len(calibration_days)} calibration days, "
        f"{len(evaluation_days)} evaluation days, "
        f"K={args.k_interventions}, "
        f"units={layout['n_units']}, selected={layout['n_selected']}",
        flush=True,
    )
    for processed, time_index in enumerate(scoring_days, start=1):
        (
            clean_day,
            mean_day,
            min_day,
            variance_day,
            op_counts,
        ) = score_day(
            student,
            values,
            active_mask,
            train_median,
            layout,
            centers,
            scale,
            score_head=args.score_head,
            time_index=time_index,
            chunk_users=args.chunk_users,
            k_interventions=args.k_interventions,
            intervention_days=args.intervention_days,
            generator=generator,
            rng=rng,
        )
        intervention_mean[:, time_index] = mean_day
        intervention_min[:, time_index] = min_day
        variance_raw[:, time_index] = variance_day
        clean_recomputed[:, time_index] = clean_day
        for op, count in op_counts.items():
            op_totals[op] += count
        if processed == 1 or processed % 25 == 0 or processed == len(scoring_days):
            elapsed = time.time() - start_time
            print(
                f"  scored {processed}/{len(scoring_days)} days "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    finite_clean = (
        np.isfinite(clean_recomputed)
        & np.isfinite(clean_cached)
    )
    clean_difference = float(
        np.max(np.abs(clean_recomputed[finite_clean] - clean_cached[finite_clean]))
    )
    if clean_difference > 1e-3:
        print(
            "WARNING: recomputed clean score differs from cached score: "
            f"max_abs_diff={clean_difference}",
            flush=True,
        )

    variance_quantile = per_user_quantiles_allow_missing(
        variance_raw,
        active_np,
        train_end,
    )
    metric_rows, full_score = evaluate_variants(
        data,
        external=bool(config["external"]),
        clean_quantile=clean_quantile,
        variance_quantile=variance_quantile,
        variance_raw=variance_raw,
        intervention_mean=intervention_mean,
        intervention_min=intervention_min,
        clean_raw=clean_cached,
        clean_name=clean_name,
        train_end=train_end,
        alpha=args.clean_weight,
    )

    prefix = (
        f"tfciv_{dataset_name}_{args.score_head}_{args.channel_mode}"
        f"_d{args.intervention_days}_s{args.seed}"
    )
    np.save(OUT / f"{prefix}_variance_raw.npy", variance_raw)
    np.save(OUT / f"{prefix}_variance_quantile.npy", variance_quantile)
    np.save(
        OUT / f"{prefix}_intervention_mean.npy",
        intervention_mean,
    )
    np.save(
        OUT / f"{prefix}_intervention_min.npy",
        intervention_min,
    )
    np.save(OUT / f"{prefix}_full_signal.npy", full_score)

    finite_valid = (
        active_np[:, : train_end + 1]
        & np.isfinite(clean_cached[:, : train_end + 1])
        & np.isfinite(variance_raw[:, : train_end + 1])
    )
    correlation = float(
        np.corrcoef(
            clean_cached[:, : train_end + 1][finite_valid],
            variance_raw[:, : train_end + 1][finite_valid],
        )[0, 1]
    )
    summary = {
        "dataset": dataset_name,
        "channel_mode": args.channel_mode,
            "score_head": args.score_head,
        "protocol": {
            "k_interventions": args.k_interventions,
            "mask_ratio": args.mask_ratio,
            "clean_weight": args.clean_weight,
            "intervention_days": args.intervention_days,
            "seed": args.seed,
            "calibration_stride": args.calibration_stride,
            "calibration_days": calibration_days,
            "evaluation_days": evaluation_days,
            "units": layout["n_units"],
            "selected_units": layout["n_selected"],
            "operators": list(OPS),
            "operator_counts": op_totals,
            "window_days": WINDOW_DAYS,
            "labels_used_for_scoring": False,
        },
        "correlation_clean_vs_variance_train": correlation,
        "recomputed_clean_max_abs_difference": float(clean_difference),
        "metrics": metric_rows,
        "elapsed_seconds": time.time() - start_time,
        "outputs": {
            "variance_raw": str(OUT / f"{prefix}_variance_raw.npy"),
            "variance_quantile": str(
                OUT / f"{prefix}_variance_quantile.npy"
            ),
            "intervention_mean": str(
                OUT / f"{prefix}_intervention_mean.npy"
            ),
            "intervention_min": str(
                OUT / f"{prefix}_intervention_min.npy"
            ),
            "full_signal": str(OUT / f"{prefix}_full_signal.npy"),
        },
    }
    (OUT / f"{prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    __import__("pandas").DataFrame(metric_rows).to_csv(
        OUT / f"{prefix}_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    print(
        json.dumps(summary, ensure_ascii=False, indent=2),
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.dataset in {"r42", "all"}:
        run_dataset(args, "r42")
    if args.dataset in {"r52", "all"}:
        run_dataset(args, "r52")


if __name__ == "__main__":
    main()
