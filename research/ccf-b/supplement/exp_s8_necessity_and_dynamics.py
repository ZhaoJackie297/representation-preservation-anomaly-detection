"""Exp S8: necessity interventions, training dynamics, and soft-kernel dose response.

This script is intentionally isolated from historical S1-S7 result directories.
It reuses only code and read-only checkpoints/tables from Exp S5a.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.special import xlogy
from scipy.stats import spearmanr, t as student_t, ttest_rel
from sklearn.metrics import average_precision_score, pairwise_distances


REPO_ROOT = Path(__file__).resolve().parents[3]
SUPPLEMENT_DIR = REPO_ROOT / "research" / "ccf-b" / "supplement"
CERT_DIR = REPO_ROOT / "runtime" / "cert"
PHASE0_DIR = REPO_ROOT / "research" / "ccf-b" / "phase0"
sys.path.insert(0, str(SUPPLEMENT_DIR))
sys.path.insert(0, str(CERT_DIR))
sys.path.insert(0, str(PHASE0_DIR))

import exp_s5a_norm_intervention as s5  # noqa: E402


OUT_ROOT = (
    REPO_ROOT
    / "research"
    / "ccf-b"
    / "supplement_results"
    / "exp_s8_necessity_and_dynamics"
)
S5A_ROOT = REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s5a_norm_intervention"
S5A_L2_ROOT = (
    REPO_ROOT
    / "research"
    / "ccf-b"
    / "supplement_results"
    / "exp_s5a_norm_intervention_l2reg_b0.2"
)

SEEDS = [20261116, 20261117, 20261118, 20261119, 20261120]
K_GRID = (5, 10, 20)
SCORERS = ("plain_l1", "bank_euclidean", "bank_cosine", "pai_a1", "pai_official")
SOFT_SPHERE_TAUS = (0.1, 0.2, 0.5)
ALPHA_GRID = (0.0, 0.05, 0.15, 0.25, 0.50, 0.75, 1.00)
TAU_MULT_GRID = (0.25, 0.50, 1.0, 2.0, 4.0)
DYNAMICS_ARMS = (
    "Hard-Rand",
    "Hard-L2Reg-b0.20",
    "Hard-L2Reg-b1.00",
    "PermSoft-Rand",
    "SoftInst-Rand",
    "Soft-Sphere-t0.1",
)
DIRECTION_ARMS = (
    "Hard-Rand",
    "Hard-L2Reg-b0.20",
    "Hard-Sphere-t0.1",
    "Hard-Sphere-t0.2",
    "Hard-Sphere-t0.5",
    "UniformSoft-Rand",
    "PermSoft-Rand",
    "SoftInst-Rand",
)
STEPS_PER_EPOCH = 62
EPS = 1e-12


@dataclass(frozen=True)
class ArmRecipe:
    name: str
    use_soft: bool
    loss_mode: str
    loss_param: float
    soft_variant: str
    alpha: float
    tau_mult: float
    expected_hook_calls: int


@dataclass
class DataBundle:
    values: np.ndarray
    users: np.ndarray
    dates: np.ndarray
    active: np.ndarray
    labels: np.ndarray
    train_end: int
    val_end: int
    train_values: np.ndarray
    profiles: np.ndarray
    data: dict
    n_steps: int
    device_id: int | str
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp S8 causal necessity and dynamics")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=(
            "smoke",
            "s8a",
            "s8b",
            "s8c-alpha",
            "s8c-tau",
            "cross",
            "finalize",
            "all",
        ),
        default=["all"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return (
        value.replace("-", "_")
        .replace(".", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def load_data(args: argparse.Namespace) -> DataBundle:
    processed = CERT_DIR / "processed_r42"
    dataset = np.load(processed / "r4.2_sequence_dataset.npz", allow_pickle=False)
    values = dataset["values"].astype(np.float32)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    data = s5.load_r42()
    labels = np.asarray(data["labels"], dtype=np.int8)
    profiles = s5.build_user_profiles(
        s5.align_profile_raw(
            processed / "r4.2_user_day_features.csv",
            users,
            dates,
        ),
        active,
        train_end,
    )
    n_steps = (len(users) // args.batch_size) * args.epochs
    if n_steps != 620:
        raise ValueError(f"Exp S8 requires exactly 620 steps, got {n_steps}")
    device_id = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{device_id}" if isinstance(device_id, int) else "cpu")
    return DataBundle(
        values=values,
        users=users,
        dates=dates,
        active=active,
        labels=labels,
        train_end=train_end,
        val_end=val_end,
        train_values=values[:, : train_end + 1],
        profiles=profiles,
        data=data,
        n_steps=n_steps,
        device_id=device_id,
        device=device,
    )


def recipe_from_name(name: str) -> ArmRecipe:
    if name.startswith("Soft-Sphere-t"):
        tau = float(name.split("Soft-Sphere-t", 1)[1])
        return ArmRecipe(
            name=name,
            use_soft=True,
            loss_mode="soft_sphere",
            loss_param=tau,
            soft_variant="profile",
            alpha=0.5,
            tau_mult=1.0,
            expected_hook_calls=620,
        )
    use_soft, loss_mode, param, variant = s5.parse_arm_spec(name)
    return ArmRecipe(
        name=name,
        use_soft=use_soft,
        loss_mode=loss_mode,
        loss_param=param,
        soft_variant=variant,
        alpha=0.5,
        tau_mult=1.0,
        expected_hook_calls=620 if loss_mode != "none" else 0,
    )


def alpha_recipe(alpha: float) -> ArmRecipe:
    name = f"Alpha-a{alpha:.2f}"
    if alpha <= EPS:
        return ArmRecipe(name, False, "none", 0.0, "none", 0.0, 1.0, 0)
    return ArmRecipe(name, True, "none", 0.0, "profile", alpha, 1.0, 0)


def tau_recipe(tau_mult: float) -> ArmRecipe:
    return ArmRecipe(
        name=f"TauMult-m{tau_mult:.2f}",
        use_soft=True,
        loss_mode="none",
        loss_param=0.0,
        soft_variant="profile",
        alpha=0.5,
        tau_mult=tau_mult,
        expected_hook_calls=0,
    )


def densify(values: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    return (
        (2.0 * alpha) / (1.0 + np.exp(-tau * values))
        + (1.0 - alpha) * np.eye(values.shape[0])
    )


def soft_matrix_scaled(
    profiles: np.ndarray,
    alpha: float,
    tau_mult: float,
) -> tuple[np.ndarray, dict]:
    distances = pairwise_distances(profiles, metric="euclidean")
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
    median_distance = float(
        np.median(normalized[np.triu_indices(len(normalized), k=1)])
    )
    tau_inst = 1.0 / max(median_distance, 1e-6)
    tau_effective = tau_inst * tau_mult
    soft = densify(-normalized, tau_effective, alpha).astype(np.float32)
    stats = target_weight_stats(soft)
    stats.update(
        {
            "alpha": alpha,
            "tau_mult": tau_mult,
            "tau_inst": tau_inst,
            "tau_effective": tau_effective,
            "median_normalized_distance": median_distance,
        }
    )
    return soft, stats


def soft_matrix_for(
    recipe: ArmRecipe,
    profiles: np.ndarray,
    seed: int,
) -> tuple[np.ndarray | None, dict]:
    if recipe.soft_variant == "none":
        return None, target_weight_stats(None)
    if recipe.tau_mult == 1.0:
        base, meta = s5.soft_instance_labels(profiles, alpha=recipe.alpha)
    else:
        base, meta = soft_matrix_scaled(profiles, recipe.alpha, recipe.tau_mult)
    if recipe.soft_variant == "profile":
        matrix = base
    elif recipe.soft_variant == "perm":
        rng = np.random.default_rng(seed + 77777)
        permutation = rng.permutation(base.shape[0])
        matrix = base[np.ix_(permutation, permutation)].copy()
    elif recipe.soft_variant == "uniform":
        off_mask = ~np.eye(base.shape[0], dtype=bool)
        mean_off = float(base[off_mask].mean())
        matrix = np.full_like(base, mean_off)
        np.fill_diagonal(matrix, np.diag(base))
    else:
        raise ValueError(f"Unknown soft variant: {recipe.soft_variant}")
    stats = target_weight_stats(matrix)
    stats.update(
        {
            "alpha": recipe.alpha,
            "tau_mult": recipe.tau_mult,
            **{
                key: value
                for key, value in meta.items()
                if key
                in {
                    "tau_inst_from_median_distance",
                    "median_pairwise_distance",
                    "tau_effective",
                }
            },
        }
    )
    return matrix.astype(np.float32), stats


def target_weight_stats(matrix: np.ndarray | None) -> dict:
    if matrix is None:
        return {
            "target_entropy": 0.0,
            "offdiag_entropy": 0.0,
            "offdiag_mass_ratio": 0.0,
            "effective_neighbors": 1.0,
            "mean_row_offdiag_mass": 0.0,
        }
    values = np.asarray(matrix, dtype=np.float64)
    off = values.copy()
    np.fill_diagonal(off, 0.0)
    row_total = values.sum(axis=1, keepdims=True)
    full_prob = values / np.maximum(row_total, 1e-12)
    entropy = -np.sum(xlogy(full_prob, full_prob), axis=1)
    off_total = off.sum(axis=1, keepdims=True)
    off_prob = off / np.maximum(off_total, 1e-12)
    off_entropy = -np.sum(xlogy(off_prob, off_prob), axis=1)
    total_mass = float(values.sum())
    offdiag_mass_ratio = (
        float(off.sum()) / total_mass if total_mass > EPS else 0.0
    )
    return {
        "target_entropy": float(entropy.mean()),
        "offdiag_entropy": float(off_entropy.mean()),
        "offdiag_mass_ratio": offdiag_mass_ratio,
        "effective_neighbors": float(np.exp(off_entropy.mean())),
        "mean_row_offdiag_mass": float(off.sum(axis=1).mean()),
    }


def _dup_matrix(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.tril(matrix, diagonal=-1)[:, :-1]
    left = left + torch.triu(matrix, diagonal=1)[:, 1:]
    right = torch.cat([left, matrix], dim=1)
    left = torch.cat([matrix, left], dim=1)
    return right, left


def _weighted_instance_spherical(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels_left: torch.Tensor,
    labels_right: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    batch_size, time_steps = z1.size(0), z1.size(1)
    if batch_size == 1:
        return z1.new_tensor(0.0)
    z1_unit = F.normalize(z1, dim=-1, eps=1e-8)
    z2_unit = F.normalize(z2, dim=-1, eps=1e-8)
    z = torch.cat([z1_unit, z2_unit], dim=0).transpose(0, 1)
    similarity = torch.matmul(z, z.transpose(1, 2)) / tau
    logits = torch.tril(similarity, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(similarity, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    index = torch.arange(batch_size, device=z1.device)
    loss = torch.sum(logits[:, index] * labels_left)
    loss = loss + torch.sum(logits[:, batch_size + index] * labels_right)
    return loss / (2.0 * batch_size * time_steps)


def _hard_temporal_spherical(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    batch_size, time_steps = z1.size(0), z1.size(1)
    if time_steps == 1:
        return z1.new_tensor(0.0)
    z1_unit = F.normalize(z1, dim=-1, eps=1e-8)
    z2_unit = F.normalize(z2, dim=-1, eps=1e-8)
    z = torch.cat([z1_unit, z2_unit], dim=1)
    similarity = torch.matmul(z, z.transpose(1, 2)) / tau
    logits = torch.tril(similarity, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(similarity, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    index = torch.arange(time_steps, device=z1.device)
    loss = (
        logits[:, index, time_steps + index - 1].mean()
        + logits[:, time_steps + index, index].mean()
    ) / 2.0
    return loss


def soft_sphere_hierarchical_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    soft_labels: np.ndarray | torch.Tensor | None,
    tau: float,
    tau_temp: float = 2.0,
    lambda_: float = 0.5,
    temporal_unit: int = 0,
    soft_temporal: bool = False,
    soft_instance: bool = True,
    temporal_hierarchy: bool = True,
) -> torch.Tensor:
    if not soft_instance or soft_labels is None:
        raise RuntimeError("Soft-Sphere requires non-empty soft instance labels")
    weights = torch.as_tensor(soft_labels, dtype=z1.dtype, device=z1.device)
    left, right = _dup_matrix(weights)
    loss = z1.new_tensor(0.0)
    depth = 0
    while z1.size(1) > 1:
        if lambda_ != 0.0:
            loss = loss + lambda_ * _weighted_instance_spherical(
                z1,
                z2,
                left,
                right,
                tau,
            )
        if depth >= temporal_unit and (1.0 - lambda_) != 0.0:
            if soft_temporal:
                raise NotImplementedError(
                    "Soft-Sphere t>0 temporal-soft mode is not used in Exp S8"
                )
            loss = loss + (1.0 - lambda_) * _hard_temporal_spherical(
                z1,
                z2,
                tau,
            )
        depth += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if lambda_ != 0.0:
            loss = loss + lambda_ * _weighted_instance_spherical(
                z1,
                z2,
                left,
                right,
                tau,
            )
        depth += 1
    return loss / max(depth, 1)


@contextlib.contextmanager
def patched_loss(
    mode: str,
    param: float,
    stats: dict[str, float | int],
):
    if mode == "none":
        yield
        return

    modules = [s5.cst_mod]
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue
        if any(
            token in module_name
            for token in ("coverage_soft", "soft_ts2vec", "soft_losses", "losses")
        ) and module not in modules:
            modules.append(module)

    targets: list[tuple[object, str, Callable]] = []
    for module in modules:
        for attribute in ("hier_CL_soft", "hierarchical_contrastive_loss"):
            function = getattr(module, attribute, None)
            if callable(function):
                targets.append((module, attribute, function))
    if not targets:
        raise RuntimeError("Could not locate the hierarchical soft contrastive loss")

    active_depth = [0]
    pre_norm_sum = [0.0]
    post_norm_sum = [0.0]
    norm_calls = [0]
    offdiag_mass_sum = [0.0]
    gradient_hook_calls = [0]

    def make_wrapper(original: Callable) -> Callable:
        def wrapped(
            z1: torch.Tensor,
            z2: torch.Tensor,
            soft_labels=None,
            *args,
            **kwargs,
        ):
            if active_depth[0] > 0:
                return original(z1, z2, soft_labels, *args, **kwargs)
            active_depth[0] += 1
            try:
                stats["hook_calls"] = int(stats.get("hook_calls", 0)) + 1
                if mode == "l2reg":
                    base = original(z1, z2, soft_labels, *args, **kwargs)
                    sq_norm = 0.5 * (
                        z1.pow(2).sum(dim=-1).mean()
                        + z2.pow(2).sum(dim=-1).mean()
                    )
                    return base + param * sq_norm
                if mode == "sphere":
                    lambda_ = float(kwargs.get("lambda_", 0.5))
                    temporal_unit = int(kwargs.get("temporal_unit", 0))
                    return s5.spherical_hier_cl(
                        z1,
                        z2,
                        tau=param,
                        lambda_=lambda_,
                        temporal_unit=temporal_unit,
                    )
                if mode == "soft_sphere":
                    if soft_labels is None:
                        raise RuntimeError("Soft-Sphere received no soft labels")
                    weights = torch.as_tensor(
                        soft_labels,
                        dtype=z1.dtype,
                        device=z1.device,
                    )
                    off = weights.clone()
                    off.fill_diagonal_(0.0)
                    if float(off.sum().detach().cpu()) <= EPS:
                        raise RuntimeError("Soft-Sphere weights have zero off-diagonal mass")
                    offdiag_mass_sum[0] += float(off.sum().detach().cpu())
                    pre_norm_sum[0] += float(
                        (z1.norm(dim=-1).mean() + z2.norm(dim=-1).mean())
                        .detach()
                        .cpu()
                    ) / 2.0
                    post_norm_sum[0] += float(
                        (
                            F.normalize(z1, dim=-1, eps=1e-8).norm(dim=-1).mean()
                            + F.normalize(z2, dim=-1, eps=1e-8).norm(dim=-1).mean()
                        )
                        .detach()
                        .cpu()
                    ) / 2.0
                    norm_calls[0] += 1
                    if z1.requires_grad:
                        z1.register_hook(
                            lambda grad: gradient_hook_calls.__setitem__(
                                0,
                                gradient_hook_calls[0] + 1,
                            )
                        )
                    if z2.requires_grad:
                        z2.register_hook(
                            lambda grad: gradient_hook_calls.__setitem__(
                                0,
                                gradient_hook_calls[0] + 1,
                            )
                        )
                    return soft_sphere_hierarchical_loss(
                        z1,
                        z2,
                        soft_labels,
                        tau=param,
                        **kwargs,
                    )
                raise ValueError(f"Unknown intervention mode: {mode}")
            finally:
                active_depth[0] -= 1

        return wrapped

    for module, attribute, original in targets:
        setattr(module, attribute, make_wrapper(original))
    try:
        yield
    finally:
        for module, attribute, original in targets:
            setattr(module, attribute, original)
        stats["mean_pre_norm"] = (
            pre_norm_sum[0] / norm_calls[0] if norm_calls[0] else float("nan")
        )
        stats["mean_post_norm"] = (
            post_norm_sum[0] / norm_calls[0] if norm_calls[0] else float("nan")
        )
        stats["mean_offdiag_mass"] = (
            offdiag_mass_sum[0] / norm_calls[0]
            if norm_calls[0]
            else float("nan")
        )
        stats["gradient_hook_calls"] = gradient_hook_calls[0]


def sample_mask_indices(
    mask: np.ndarray,
    limit: int,
    seed: int = 20261116,
) -> np.ndarray:
    indices = np.flatnonzero(mask.ravel())
    if len(indices) > limit:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=limit, replace=False))
    return indices.astype(np.int64)


def direction_geometry(
    reprs: np.ndarray,
    active: np.ndarray,
    labels: np.ndarray,
    train_end: int,
    val_end: int,
) -> dict[str, float]:
    train_normal = np.zeros_like(active, dtype=bool)
    train_normal[:, : train_end + 1] = (
        active[:, : train_end + 1] & (labels[:, : train_end + 1] == 0)
    )
    test_normal = np.zeros_like(active, dtype=bool)
    test_normal[:, val_end + 1 :] = (
        active[:, val_end + 1 :] & (labels[:, val_end + 1 :] == 0)
    )
    train_indices = sample_mask_indices(train_normal, 4096)
    pairwise_train_indices = sample_mask_indices(train_normal, 2048)
    test_indices = sample_mask_indices(test_normal, 2048)
    active_indices = sample_mask_indices(active, 8192)

    def vectors(flat_indices: np.ndarray) -> np.ndarray:
        users, times = np.unravel_index(flat_indices, active.shape)
        values = reprs[users, times].astype(np.float64)
        return values / np.maximum(
            np.linalg.norm(values, axis=1, keepdims=True),
            1e-12,
        )

    train_unit = vectors(train_indices)
    pairwise_train = vectors(pairwise_train_indices)
    test_unit = vectors(test_indices)
    active_unit = vectors(active_indices)
    train_mean_cosine = mean_offdiag_cosine(pairwise_train)
    test_mean_cosine = mean_offdiag_cosine(test_unit)
    margin, top1, top5 = cosine_nearest_margin(test_unit, train_unit)
    return {
        "unit_effective_rank_active": s5.effective_rank(active_unit),
        "unit_effective_rank_train_normal": s5.effective_rank(pairwise_train),
        "unit_effective_rank_test_normal": s5.effective_rank(test_unit),
        "mean_pairwise_cosine_train_normal": train_mean_cosine,
        "mean_pairwise_cosine_test_normal": test_mean_cosine,
        "cosine_nn_margin_test_normal": margin,
        "cosine_nn_top1_test_normal": top1,
        "cosine_nn_top5_mean_test_normal": top5,
        "direction_train_samples": int(len(train_unit)),
        "direction_test_samples": int(len(test_unit)),
    }


def mean_offdiag_cosine(unit_vectors: np.ndarray) -> float:
    if len(unit_vectors) < 2:
        return float("nan")
    similarity = unit_vectors @ unit_vectors.T
    count = len(unit_vectors)
    return float((similarity.sum() - count) / (count * (count - 1)))


def cosine_nearest_margin(
    query: np.ndarray,
    bank: np.ndarray,
) -> tuple[float, float, float]:
    if len(query) == 0 or len(bank) == 0:
        return float("nan"), float("nan"), float("nan")
    k = min(5, len(bank))
    top_similarity = []
    for start in range(0, len(query), 512):
        similarity = query[start : start + 512] @ bank.T
        top_similarity.append(np.partition(similarity, -k, axis=1)[:, -k:])
    top = np.concatenate(top_similarity, axis=0)
    top = np.sort(top, axis=1)[:, ::-1]
    top1 = float(top[:, 0].mean())
    top5 = float(top[:, :k].mean())
    return top1 - top5, top1, top5


def geometry_metrics(
    reprs: np.ndarray,
    bundle: DataBundle,
) -> dict[str, float]:
    active_norms = np.linalg.norm(reprs[bundle.active], axis=-1)
    test_mask = np.zeros_like(bundle.active, dtype=bool)
    test_mask[:, bundle.val_end + 1 :] = bundle.active[:, bundle.val_end + 1 :]
    q_norms = np.linalg.norm(reprs[test_mask], axis=-1)
    x_norms = np.linalg.norm(
        bundle.values[:, :, :20],
        axis=-1,
    )[test_mask]
    finite = np.isfinite(q_norms) & np.isfinite(x_norms)
    rho = (
        float(spearmanr(q_norms[finite], x_norms[finite])[0])
        if finite.sum() > 2
        else float("nan")
    )
    user_geometry = s5.user_geometry_metrics(
        reprs,
        bundle.active,
        bundle.profiles,
        bundle.train_end,
        bundle.val_end,
    )
    return {
        "mean_norm_active": float(active_norms.mean()),
        "std_norm_active": float(active_norms.std()),
        "cv_norm_active": float(active_norms.std() / max(active_norms.mean(), 1e-8)),
        "effective_rank_active": s5.effective_rank(reprs[bundle.active]),
        "spearman_norm_vs_xnorm": rho,
        **user_geometry,
        **direction_geometry(
            reprs,
            bundle.active,
            bundle.labels,
            bundle.train_end,
            bundle.val_end,
        ),
    }


def prepare_trajectory_samples(bundle: DataBundle) -> dict[str, np.ndarray]:
    train_mask = np.zeros_like(bundle.active, dtype=bool)
    train_mask[:, : bundle.train_end + 1] = bundle.active[
        :, : bundle.train_end + 1
    ]
    train_indices = sample_mask_indices(train_mask, 4096)

    validation_labels = bundle.labels[
        :, bundle.train_end + 1 : bundle.val_end + 1
    ]
    validation_active = bundle.active[
        :, bundle.train_end + 1 : bundle.val_end + 1
    ]
    positive_flat = np.flatnonzero(
        (validation_active & (validation_labels == 1)).ravel()
    )
    negative_flat = np.flatnonzero(
        (validation_active & (validation_labels == 0)).ravel()
    )
    rng = np.random.default_rng(20261116)
    if len(negative_flat) > 5000:
        negative_flat = np.sort(
            rng.choice(negative_flat, size=5000, replace=False)
        )
    validation_flat = np.concatenate([positive_flat, negative_flat])
    validation_users, validation_times = np.unravel_index(
        validation_flat,
        validation_active.shape,
    )
    validation_indices = (
        validation_users * bundle.values.shape[1]
        + validation_times
        + bundle.train_end
        + 1
    )
    validation_labels_sample = bundle.labels.ravel()[validation_indices]
    return {
        "train_indices": train_indices,
        "validation_indices": np.sort(validation_indices),
        "validation_labels": validation_labels_sample[
            np.argsort(validation_indices)
        ].astype(np.int8),
        "train_windows": slice_windows(bundle.values, train_indices),
        "validation_windows": slice_windows(bundle.values, validation_indices),
    }


def slice_windows(
    values: np.ndarray,
    flat_indices: np.ndarray,
    window_days: int = 31,
) -> np.ndarray:
    users, times = np.unravel_index(flat_indices, values.shape[:2])
    result = np.zeros(
        (len(flat_indices), window_days, values.shape[-1]),
        dtype=np.float32,
    )
    for index, (user, time_index) in enumerate(zip(users, times)):
        left = max(0, int(time_index) - window_days + 1)
        block = values[user, left : int(time_index) + 1]
        result[index, -len(block) :] = block
    return result


def representative_windows(
    model,
    windows: np.ndarray,
) -> np.ndarray:
    representation = np.asarray(
        model.encode(
            windows,
            causal=True,
            batch_size=256,
        ),
        dtype=np.float32,
    )
    return representation[:, -1, :]


def snapshot_metric(
    model,
    samples: dict[str, np.ndarray],
    step: int,
    train_loss: float,
    bundle: DataBundle,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    np_state = np.random.get_state()
    random_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        snapshot_model = s5.CoverageSoftTS2Vec(
            input_dims=bundle.values.shape[-1],
            output_dims=64,
            hidden_dims=64,
            depth=5,
            device="cpu",
            lr=1e-3,
            batch_size=args.batch_size,
            lambda_=0.5,
            tau_temp=1.5,
            soft_instance=True,
            soft_temporal=False,
        )
        snapshot_model.net.load_state_dict(
            {
                key: value.detach().cpu()
                for key, value in model.net.state_dict().items()
            }
        )
        snapshot_model.net.eval()
        train_repr = representative_windows(
            snapshot_model,
            samples["train_windows"],
        )
        validation_repr = representative_windows(
            snapshot_model,
            samples["validation_windows"],
        )
        train_norms = np.linalg.norm(train_repr, axis=1)
        train_unit = train_repr / np.maximum(
            np.linalg.norm(train_repr, axis=1, keepdims=True),
            1e-12,
        )
        validation_unit = validation_repr / np.maximum(
            np.linalg.norm(validation_repr, axis=1, keepdims=True),
            1e-12,
        )
        max_similarity = []
        train_tensor = torch.from_numpy(train_unit)
        for start in range(0, len(validation_unit), 512):
            query = torch.from_numpy(validation_unit[start : start + 512])
            with torch.inference_mode():
                max_similarity.append(
                    (query @ train_tensor.T).max(dim=1).values.cpu().numpy()
                )
        cosine_score = 1.0 - np.concatenate(max_similarity)
        validation_labels = samples["validation_labels"]
        if len(np.unique(validation_labels)) == 2:
            val_pr_auc = float(
                average_precision_score(validation_labels, cosine_score)
            )
        else:
            val_pr_auc = float("nan")
        return {
            "step": int(step),
            "epoch": int(step // STEPS_PER_EPOCH),
            "train_loss": float(train_loss),
            "mean_norm_active": float(train_norms.mean()),
            "cv_norm_active": float(
                train_norms.std() / max(train_norms.mean(), 1e-8)
            ),
            "effective_rank_active": s5.effective_rank(train_repr),
            "unit_effective_rank": s5.effective_rank(train_unit),
            "val_cos_pr_auc": val_pr_auc,
            "val_positive_samples": int(validation_labels.sum()),
            "val_negative_samples": int(
                len(validation_labels) - validation_labels.sum()
            ),
            "train_trajectory_samples": int(len(train_repr)),
        }
    finally:
        np.random.set_state(np_state)
        random.setstate(random_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def build_model(bundle: DataBundle, args: argparse.Namespace):
    return s5.CoverageSoftTS2Vec(
        input_dims=bundle.values.shape[-1],
        output_dims=64,
        hidden_dims=64,
        depth=5,
        device=bundle.device_id,
        lr=1e-3,
        batch_size=args.batch_size,
        lambda_=0.5,
        tau_temp=1.5,
        soft_instance=True,
        soft_temporal=False,
    )


def run_training(
    *,
    recipe: ArmRecipe,
    seed: int,
    stage: str,
    collect_trajectory: bool,
    output_root: Path,
    bundle: DataBundle,
    args: argparse.Namespace,
) -> dict:
    run_dir = output_root / "training_runs" / stage / safe_name(recipe.name) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    geometry_path = run_dir / "geometry.json"
    model_path = run_dir / "model.pth"
    trajectory_path = run_dir / "trajectory.csv"
    cached = (
        metrics_path.is_file()
        and geometry_path.is_file()
        and model_path.is_file()
        and (not collect_trajectory or trajectory_path.is_file())
    )
    if cached and not args.force:
        metrics = pd.read_csv(metrics_path)
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        trajectory = (
            pd.read_csv(trajectory_path) if collect_trajectory else pd.DataFrame()
        )
        print(f"[SKIP] {recipe.name} seed={seed}", flush=True)
        return {
            "metrics": metrics,
            "geometry": geometry,
            "trajectory": trajectory,
            "elapsed_sec": geometry.get("elapsed_sec", float("nan")),
        }

    start_time = time.time()
    batch_table = s5.build_random_batch_table(
        len(bundle.users),
        args.batch_size,
        bundle.n_steps,
        seed,
    )
    soft_matrix, soft_stats = soft_matrix_for(recipe, bundle.profiles, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(bundle, args)
    if not recipe.use_soft:
        model.soft_instance = False
    hook_stats: dict[str, float | int] = {"hook_calls": 0}
    trajectory_rows: list[dict] = []
    trajectory_samples = (
        prepare_trajectory_samples(bundle) if collect_trajectory else None
    )

    if collect_trajectory:
        trajectory_rows.append(
            snapshot_metric(
                model,
                trajectory_samples,
                step=0,
                train_loss=float("nan"),
                bundle=bundle,
                args=args,
            )
        )

        def after_iter(model_ref, loss_value: float) -> None:
            step = int(model_ref.n_iters)
            if step > 0 and step % STEPS_PER_EPOCH == 0:
                trajectory_rows.append(
                    snapshot_metric(
                        model_ref,
                        trajectory_samples,
                        step=step,
                        train_loss=loss_value,
                        bundle=bundle,
                        args=args,
                    )
                )

        model.after_iter_callback = after_iter

    with patched_loss(recipe.loss_mode, recipe.loss_param, hook_stats):
        logs = model.fit_coverage_table(
            train_data=bundle.train_values,
            soft_labels=soft_matrix,
            batch_table=batch_table,
            n_epochs=1,
            verbose=False,
        )

    if recipe.loss_mode != "none":
        if int(hook_stats.get("hook_calls", 0)) != bundle.n_steps:
            raise RuntimeError(
                f"{recipe.name}: loss hook fired {hook_stats.get('hook_calls')} "
                f"times, expected exactly {bundle.n_steps}"
            )
    if recipe.loss_mode == "soft_sphere":
        gradient_calls = int(hook_stats.get("gradient_hook_calls", 0))
        if gradient_calls < bundle.n_steps * 2:
            raise RuntimeError(
                f"{recipe.name}: gradient hooks fired {gradient_calls} times, "
                f"expected at least {bundle.n_steps * 2}"
            )
        if abs(float(hook_stats.get("mean_post_norm", 0.0)) - 1.0) > 1e-4:
            raise RuntimeError(
                f"{recipe.name}: normalized representations did not have unit "
                f"norm (mean={hook_stats.get('mean_post_norm')})"
            )

    encode_args = {
        "causal": True,
        "sliding_length": 1,
        "sliding_padding": 30,
        "batch_size": 256,
    }
    plain_repr = np.asarray(
        model.encode(bundle.values, **encode_args),
        dtype=np.float32,
    )
    masked_repr = np.asarray(
        model.encode(bundle.values, mask="mask_last", **encode_args),
        dtype=np.float32,
    )
    masked_raw = np.abs(masked_repr - plain_repr).sum(axis=-1)
    cosine_raw, euclidean_raw = s5.bank_scores(
        plain_repr,
        plain_repr,
        bundle.active,
        bundle.train_end,
        bundle.device,
    )
    masked_z = s5.global_train_z(
        masked_raw,
        bundle.active,
        bundle.train_end,
    )
    cosine_z = s5.global_train_z(
        cosine_raw,
        bundle.active,
        bundle.train_end,
    )
    euclidean_z = s5.global_train_z(
        euclidean_raw,
        bundle.active,
        bundle.train_end,
    )
    raw_20 = s5.align_raw_features(
        CERT_DIR / "processed_r42" / "r4.2_user_day_features.csv",
        bundle.users,
        bundle.dates,
    )
    G, Q = s5.aggregate_components(
        raw_20,
        bundle.active,
        bundle.train_end,
        user_specific=True,
        aggregation="channel_mean",
    )
    G_z = s5.global_train_z(G, bundle.active, bundle.train_end)
    Q_z = s5.global_train_z(Q, bundle.active, bundle.train_end)
    pai_mean = euclidean_z + s5.PAI_W_G * G_z + s5.PAI_W_Q * Q_z
    pai_official = (
        s5.PAI_OFF_W_EUC * euclidean_z
        + s5.PAI_OFF_W_G * G_z
        + s5.PAI_OFF_W_Q * Q_z
    )
    scores = {
        "plain_l1": masked_z,
        "bank_cosine": cosine_z,
        "bank_euclidean": euclidean_z,
        "pai_a1": pai_mean,
        "pai_official": pai_official,
    }
    metrics_rows: list[dict] = []
    hit_map: dict[str, list[str]] = {}
    for scorer, signal in scores.items():
        rows, hits = s5.evaluate_r42(
            f"{recipe.name}__{scorer}",
            signal,
            bundle.data,
            bundle.active,
            bundle.train_end,
            bundle.val_end,
            seed,
        )
        metrics_rows.extend(rows)
        hit_map.update(hits)
    metrics = pd.DataFrame(metrics_rows)
    metrics["arm"] = recipe.name
    metrics["scorer"] = metrics["method"].str.split("__").str[-1]
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    (run_dir / "hit_ids_k10.json").write_text(
        json.dumps(hit_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    geometry = {
        "arm": recipe.name,
        "seed": seed,
        "n_steps": bundle.n_steps,
        "final_loss": float(logs[-1]["loss"]) if logs else float("nan"),
        **soft_stats,
        **hook_stats,
        **geometry_metrics(plain_repr, bundle),
        "elapsed_sec": round(time.time() - start_time, 2),
        "loss_mode": recipe.loss_mode,
        "loss_param": recipe.loss_param,
        "soft_variant": recipe.soft_variant,
    }
    geometry_path.write_text(
        json.dumps(geometry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(model.net.state_dict(), model_path)
    pd.DataFrame(logs).to_csv(
        run_dir / "training_steps.csv",
        index=False,
        encoding="utf-8",
    )
    trajectory = pd.DataFrame(trajectory_rows)
    if collect_trajectory:
        expected_steps = list(range(0, bundle.n_steps + 1, STEPS_PER_EPOCH))
        if trajectory["step"].tolist() != expected_steps:
            raise RuntimeError(
                f"{recipe.name}: trajectory steps={trajectory['step'].tolist()}, "
                f"expected={expected_steps}"
            )
        trajectory.insert(0, "seed", seed)
        trajectory.insert(0, "arm", recipe.name)
        trajectory.to_csv(trajectory_path, index=False, encoding="utf-8")

    print(
        f"[DONE] {recipe.name:20s} seed={seed} "
        f"norm={geometry['mean_norm_active']:.3f} "
        f"unitER={geometry['unit_effective_rank_active']:.2f} "
        f"({geometry['elapsed_sec']:.1f}s)",
        flush=True,
    )
    return {
        "metrics": metrics,
        "geometry": geometry,
        "trajectory": trajectory,
        "elapsed_sec": geometry["elapsed_sec"],
    }


def summarize_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    return (
        frame.groupby([*group_columns, "scorer", "split", "k"], as_index=False)
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
        .sort_values([*group_columns, "split", "k", "scorer"])
    )


def summarize_geometry(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    metrics = [
        "mean_norm_active",
        "cv_norm_active",
        "effective_rank_active",
        "unit_effective_rank_active",
        "unit_effective_rank_train_normal",
        "unit_effective_rank_test_normal",
        "mean_pairwise_cosine_train_normal",
        "mean_pairwise_cosine_test_normal",
        "cosine_nn_margin_test_normal",
        "spearman_norm_vs_xnorm",
        "val_knn10_agreement",
        "val_local_dist_spearman",
        "target_entropy",
        "offdiag_mass_ratio",
        "effective_neighbors",
    ]
    available = [
        metric
        for metric in metrics
        if metric in frame.columns
        and pd.api.types.is_numeric_dtype(frame[metric])
    ]
    rows: list[dict] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for metric in available:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"n_seeds"] = int(group["seed"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    result = [float("nan")] * count
    ranked = sorted(
        (
            (value, index)
            for index, value in enumerate(p_values)
            if math.isfinite(value)
        ),
        key=lambda item: item[0],
    )
    running = 0.0
    for rank, (value, index) in enumerate(ranked):
        adjusted = min(1.0, (count - rank) * value)
        running = max(running, adjusted)
        result[index] = running
    return result


def paired_series(
    candidate: pd.Series,
    reference: pd.Series,
) -> dict:
    aligned = pd.concat(
        [candidate.rename("candidate"), reference.rename("reference")],
        axis=1,
        join="inner",
    ).dropna()
    delta = aligned["candidate"] - aligned["reference"]
    n = len(delta)
    if n < 2:
        return {
            "n_pairs": n,
            "delta_mean": float("nan"),
            "delta_std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "positive_wins": 0,
            "losses": 0,
            "ties": n,
            "wins_text": f"0/{n}",
            "p_value": float("nan"),
            "test_note": "insufficient_pairs",
        }
    mean = float(delta.mean())
    std = float(delta.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=n - 1))
    half = critical * std / math.sqrt(n)
    wins = int((delta > EPS).sum())
    losses = int((delta < -EPS).sum())
    ties = n - wins - losses
    if np.all(np.abs(delta.to_numpy()) <= EPS):
        p_value = 1.0
        note = "all_paired_differences_zero"
    elif std <= EPS:
        p_value = 0.0
        note = "constant_nonzero_delta_due_to_zero_sample_variance"
    else:
        p_value = float(
            ttest_rel(
                aligned["candidate"],
                aligned["reference"],
            ).pvalue
        )
        note = ""
    return {
        "n_pairs": n,
        "delta_mean": mean,
        "delta_std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "positive_wins": wins,
        "losses": losses,
        "ties": ties,
        "wins_text": f"{wins}/{n}",
        "p_value": p_value,
        "test_note": note,
    }


def paired_metric_tests(
    metrics: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
    *,
    scorers: Iterable[str] = SCORERS,
    prefix: str = "",
) -> pd.DataFrame:
    metric_specs = [
        ("PR-AUC", "pr_auc"),
        ("ROC-AUC", "roc_auc"),
        ("Event Recall@K=5", "event_recall_3d", 5),
        ("Event Recall@K=10", "event_recall_3d", 10),
        ("Event Recall@K=20", "event_recall_3d", 20),
    ]
    rows: list[dict] = []
    test = metrics[metrics["split"] == "test"]
    for scorer in scorers:
        scorer_frame = test[test["scorer"] == scorer]
        for metric_label, field, *k_value in metric_specs:
            for candidate_arm, reference_arm, family in comparisons:
                candidate = scorer_frame[scorer_frame["arm"] == candidate_arm]
                reference = scorer_frame[scorer_frame["arm"] == reference_arm]
                if k_value:
                    candidate = candidate[candidate["k"] == k_value[0]]
                    reference = reference[reference["k"] == k_value[0]]
                else:
                    candidate = candidate[candidate["k"] == 10]
                    reference = reference[reference["k"] == 10]
                if candidate.empty or reference.empty:
                    continue
                result = paired_series(
                    candidate.set_index("seed")[field],
                    reference.set_index("seed")[field],
                )
                rows.append(
                    {
                        "candidate_arm": candidate_arm,
                        "reference_arm": reference_arm,
                        "scorer": scorer,
                        "metric": f"{prefix}{metric_label}",
                        "comparison_family": family,
                        "delta_definition": "candidate - reference",
                        **result,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_holm"] = float("nan")
    for _, indices in frame.groupby(
        ["reference_arm", "scorer", "metric"],
        dropna=False,
    ).groups.items():
        adjusted = holm_adjust(
            frame.loc[indices, "p_value"].astype(float).tolist()
        )
        frame.loc[indices, "p_holm"] = adjusted
    return frame


def paired_geometry_tests(
    geometry: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
) -> pd.DataFrame:
    geometry_metrics_list = [
        "mean_norm_active",
        "unit_effective_rank_test_normal",
        "mean_pairwise_cosine_test_normal",
        "cosine_nn_margin_test_normal",
    ]
    rows: list[dict] = []
    for candidate_arm, reference_arm, family in comparisons:
        candidate = geometry[geometry["arm"] == candidate_arm].set_index("seed")
        reference = geometry[geometry["arm"] == reference_arm].set_index("seed")
        if candidate.empty or reference.empty:
            continue
        for metric in geometry_metrics_list:
            rows.append(
                {
                    "candidate_arm": candidate_arm,
                    "reference_arm": reference_arm,
                    "scorer": "geometry",
                    "metric": metric,
                    "comparison_family": family,
                    "delta_definition": "candidate - reference",
                    **paired_series(candidate[metric], reference[metric]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_holm"] = float("nan")
    for _, indices in frame.groupby(
        ["reference_arm", "metric"],
        dropna=False,
    ).groups.items():
        frame.loc[indices, "p_holm"] = holm_adjust(
            frame.loc[indices, "p_value"].astype(float).tolist()
        )
    return frame


def load_s5a_tables(
    arms: Iterable[str],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    arm_list = list(arms)
    main_metrics = pd.read_csv(S5A_ROOT / "metrics_by_seed.csv")
    main_geometry = pd.read_csv(S5A_ROOT / "geometry_by_seed.csv")
    metric_parts = []
    geometry_parts = []
    for arm in arm_list:
        if arm == "Hard-L2Reg-b0.20":
            metrics = pd.read_csv(S5A_L2_ROOT / "metrics_by_seed.csv")
            geometry = pd.read_csv(S5A_L2_ROOT / "geometry_by_seed.csv")
            metrics = metrics[metrics["arm"] == "Hard-L2Reg-b0.2"].copy()
            geometry = geometry[geometry["arm"] == "Hard-L2Reg-b0.2"].copy()
            metrics["arm"] = arm
            geometry["arm"] = arm
        else:
            metrics = main_metrics[main_metrics["arm"] == arm].copy()
            geometry = main_geometry[main_geometry["arm"] == arm].copy()
        metric_parts.append(metrics[metrics["seed"].isin(seeds)])
        geometry_parts.append(geometry[geometry["seed"].isin(seeds)])
    metrics = pd.concat(metric_parts, ignore_index=True)
    geometry = pd.concat(geometry_parts, ignore_index=True)
    if metrics.empty or geometry.empty:
        raise ValueError("Requested S5a arms/seeds are not present")
    return metrics, geometry


def load_checkpoint_model(
    arm: str,
    seed: int,
    bundle: DataBundle,
    args: argparse.Namespace,
):
    checkpoint_arm = "Hard-L2Reg-b0.2" if arm == "Hard-L2Reg-b0.20" else arm
    checkpoint_root = S5A_L2_ROOT if arm == "Hard-L2Reg-b0.20" else S5A_ROOT
    path = checkpoint_root / checkpoint_arm / f"seed_{seed}" / "model.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    model = build_model(bundle, args)
    state = torch.load(
        path,
        map_location=bundle.device,
        weights_only=True,
    )
    model.net.load_state_dict(state)
    model.net.eval()
    return model


def reevaluate_s5a_checkpoints(
    bundle: DataBundle,
    args: argparse.Namespace,
    output_root: Path,
    arms: Iterable[str] = DIRECTION_ARMS,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    seeds = seeds or SEEDS
    arm_list = list(arms)
    cache_path = output_root / "s5a_checkpoint_geometry_by_seed.csv"
    if cache_path.is_file() and not args.force:
        cached = pd.read_csv(cache_path)
        if set(arm_list).issubset(set(cached["arm"].unique())) and cached[
            "seed"
        ].nunique() == len(seeds):
            return cached[cached["arm"].isin(arm_list)].copy()
    _, old_geometry = load_s5a_tables(arm_list, seeds)
    rows = []
    for arm in arm_list:
        recipe = recipe_from_name(arm)
        for seed in seeds:
            model = load_checkpoint_model(arm, seed, bundle, args)
            representation = np.asarray(
                model.encode(
                    bundle.values,
                    causal=True,
                    sliding_length=1,
                    sliding_padding=30,
                    batch_size=256,
                ),
                dtype=np.float32,
            )
            _, soft_stats = soft_matrix_for(recipe, bundle.profiles, seed)
            row = {
                "arm": arm,
                "seed": seed,
                **geometry_metrics(representation, bundle),
                **soft_stats,
                "source": "S5a checkpoint reevaluation",
            }
            rows.append(row)
            print(f"[S5a-GEOM] {arm} seed={seed}", flush=True)
    frame = pd.DataFrame(rows)
    old = old_geometry[["arm", "seed", "mean_norm_active", "effective_rank_active"]]
    check = frame.merge(
        old,
        on=["arm", "seed"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    check["norm_abs_delta"] = (
        check["mean_norm_active_new"] - check["mean_norm_active_old"]
    ).abs()
    check["rank_abs_delta"] = (
        check["effective_rank_active_new"]
        - check["effective_rank_active_old"]
    ).abs()
    if check[["norm_abs_delta", "rank_abs_delta"]].to_numpy().max() > 1e-4:
        raise RuntimeError("S5a checkpoint geometry did not reproduce stored geometry")
    frame.to_csv(cache_path, index=False, encoding="utf-8")
    check.to_csv(
        output_root / "s5a_checkpoint_geometry_reproduction.csv",
        index=False,
        encoding="utf-8",
    )
    return frame


def metrics_for_recipe(
    recipe: ArmRecipe,
    seed: int,
    output_root: Path,
    bundle: DataBundle,
    args: argparse.Namespace,
    *,
    stage: str,
    collect_trajectory: bool = False,
) -> dict:
    return run_training(
        recipe=recipe,
        seed=seed,
        stage=stage,
        collect_trajectory=collect_trajectory,
        output_root=output_root,
        bundle=bundle,
        args=args,
    )


def run_s8a(
    bundle: DataBundle,
    args: argparse.Namespace,
    output_root: Path,
) -> pd.DataFrame:
    metrics_frames = []
    geometry_rows = []
    for tau in SOFT_SPHERE_TAUS:
        recipe = recipe_from_name(f"Soft-Sphere-t{tau:g}")
        for seed in args.seeds:
            result = run_training(
                recipe=recipe,
                seed=seed,
                stage="s8a",
                collect_trajectory=(abs(tau - 0.1) <= EPS),
                output_root=output_root,
                bundle=bundle,
                args=args,
            )
            metrics_frames.append(result["metrics"])
            geometry_rows.append(result["geometry"])
    metrics = pd.concat(metrics_frames, ignore_index=True)
    geometry = pd.DataFrame(geometry_rows)
    metrics.to_csv(
        output_root / "soft_sphere_metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summarize_metrics(metrics, ["arm"]).to_csv(
        output_root / "soft_sphere_metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    geometry.to_csv(
        output_root / "soft_sphere_geometry_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summarize_geometry(geometry, ["arm"]).to_csv(
        output_root / "soft_sphere_geometry_summary.csv",
        index=False,
        encoding="utf-8",
    )

    reference_arms = [
        "SoftInst-Rand",
        "Hard-Sphere-t0.1",
        "Hard-Sphere-t0.2",
        "Hard-Sphere-t0.5",
    ]
    s5_metrics, _ = load_s5a_tables(reference_arms, args.seeds)
    combined = pd.concat([metrics, s5_metrics], ignore_index=True)
    comparisons = []
    for tau in SOFT_SPHERE_TAUS:
        candidate = f"Soft-Sphere-t{tau:g}"
        comparisons.append((candidate, "SoftInst-Rand", f"t{tau:g}_vs_softinst"))
        comparisons.append(
            (
                candidate,
                f"Hard-Sphere-t{tau:g}",
                f"t{tau:g}_vs_hardsphere",
            )
        )
    paired_metric_tests(combined, comparisons).to_csv(
        output_root / "soft_sphere_paired_tests.csv",
        index=False,
        encoding="utf-8",
    )
    return metrics


def run_s8b(
    bundle: DataBundle,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_frames = []
    geometry_rows = []
    trajectories = []
    for arm in DYNAMICS_ARMS:
        recipe = recipe_from_name(arm)
        for seed in args.seeds:
            if arm == "Soft-Sphere-t0.1":
                run_dir = (
                    output_root
                    / "training_runs"
                    / "s8a"
                    / safe_name(arm)
                    / f"seed_{seed}"
                )
                metrics_path = run_dir / "metrics.csv"
                geometry_path = run_dir / "geometry.json"
                trajectory_path = run_dir / "trajectory.csv"
                if not (
                    metrics_path.is_file()
                    and geometry_path.is_file()
                    and trajectory_path.is_file()
                ):
                    raise FileNotFoundError(
                        "Run S8a first so Soft-Sphere trajectory can be reused"
                    )
                metrics_frames.append(pd.read_csv(metrics_path))
                geometry_rows.append(
                    json.loads(geometry_path.read_text(encoding="utf-8"))
                )
                trajectories.append(pd.read_csv(trajectory_path))
                continue
            result = run_training(
                recipe=recipe,
                seed=seed,
                stage="s8b",
                collect_trajectory=True,
                output_root=output_root,
                bundle=bundle,
                args=args,
            )
            metrics_frames.append(result["metrics"])
            geometry_rows.append(result["geometry"])
            trajectories.append(result["trajectory"])

    metrics = pd.concat(metrics_frames, ignore_index=True)
    geometry = pd.DataFrame(geometry_rows)
    trajectory = pd.concat(trajectories, ignore_index=True)
    trajectory = trajectory.sort_values(["arm", "seed", "step"]).reset_index(drop=True)
    metrics.to_csv(
        output_root / "dynamics_final_metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    geometry.to_csv(
        output_root / "dynamics_final_geometry_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    trajectory.to_csv(
        output_root / "dynamics_trajectory_by_seed.csv",
        index=False,
        encoding="utf-8",
    )

    summary_rows = []
    trajectory_metrics = [
        "train_loss",
        "mean_norm_active",
        "cv_norm_active",
        "effective_rank_active",
        "unit_effective_rank",
        "val_cos_pr_auc",
    ]
    for keys, group in trajectory.groupby(["arm", "step", "epoch"], as_index=False):
        row = {
            "arm": keys[0],
            "step": int(keys[1]),
            "epoch": int(keys[2]),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in trajectory_metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        output_root / "dynamics_trajectory_summary.csv",
        index=False,
        encoding="utf-8",
    )

    reproduction_audit(
        metrics,
        arms=(
            "Hard-Rand",
            "Hard-L2Reg-b0.20",
            "Hard-L2Reg-b1.00",
            "PermSoft-Rand",
            "SoftInst-Rand",
        ),
        seeds=args.seeds,
        output_root=output_root,
    )
    return metrics, trajectory


def reproduction_audit(
    new_metrics: pd.DataFrame,
    *,
    arms: Iterable[str],
    seeds: list[int],
    output_root: Path,
) -> pd.DataFrame:
    old, _ = load_s5a_tables(arms, seeds)
    keys = ["seed", "arm", "scorer", "split", "k"]
    fields = ["pr_auc", "roc_auc", "event_recall_3d"]
    merged = old[keys + fields].merge(
        new_metrics[keys + fields],
        on=keys,
        suffixes=("_s5a", "_s8"),
        validate="one_to_one",
    )
    for field in fields:
        merged[f"{field}_abs_delta"] = (
            merged[f"{field}_s5a"] - merged[f"{field}_s8"]
        ).abs()
    merged.to_csv(
        output_root / "s5a_reproduction_audit_by_row.csv",
        index=False,
        encoding="utf-8",
    )
    summary_rows = []
    for arm, group in merged.groupby("arm"):
        row = {"arm": arm, "rows_compared": len(group)}
        for field in fields:
            row[f"max_abs_delta_{field}"] = float(
                group[f"{field}_abs_delta"].max()
            )
        row["exact_within_1e-12"] = bool(
            group[[f"{field}_abs_delta" for field in fields]]
            .to_numpy()
            .max()
            <= EPS
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_root / "s5a_reproduction_audit_summary.csv",
        index=False,
        encoding="utf-8",
    )
    return summary


def attach_geometry(
    metrics: pd.DataFrame,
    geometry: pd.DataFrame,
    extra_columns: list[str],
) -> pd.DataFrame:
    geometry_values = [
        "mean_norm_active",
        "cv_norm_active",
        "effective_rank_active",
        "unit_effective_rank_active",
        "unit_effective_rank_train_normal",
        "unit_effective_rank_test_normal",
        "mean_pairwise_cosine_train_normal",
        "mean_pairwise_cosine_test_normal",
        "cosine_nn_margin_test_normal",
        "spearman_norm_vs_xnorm",
        "val_knn10_agreement",
        "val_local_dist_spearman",
        "target_entropy",
        "offdiag_entropy",
        "offdiag_mass_ratio",
        "effective_neighbors",
        "mean_row_offdiag_mass",
    ]
    available = [
        column
        for column in geometry_values + extra_columns
        if column in geometry.columns and column not in metrics.columns
    ]
    return metrics.merge(
        geometry[["arm", "seed", *available]],
        on=["arm", "seed"],
        how="left",
        validate="many_to_one",
    )


def combined_summary(
    metrics: pd.DataFrame,
    geometry: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    metric_summary = summarize_metrics(metrics, group_columns)
    geometry_summary = summarize_geometry(geometry, group_columns)
    if "n_seeds" in geometry_summary.columns:
        geometry_summary = geometry_summary.rename(
            columns={"n_seeds": "geometry_n_seeds"}
        )
    return metric_summary.merge(
        geometry_summary,
        on=group_columns,
        how="left",
        validate="many_to_one",
    )


def existing_geometry_for_dose(
    direction_geometry_frame: pd.DataFrame,
    bundle: DataBundle,
    *,
    arm: str,
    alpha: float,
    tau_mult: float,
    seeds: list[int],
) -> pd.DataFrame:
    recipe = alpha_recipe(alpha) if tau_mult == 1.0 else tau_recipe(tau_mult)
    rows = []
    for seed in seeds:
        selected = direction_geometry_frame[
            (direction_geometry_frame["arm"] == arm)
            & (direction_geometry_frame["seed"] == seed)
        ]
        if selected.empty:
            raise ValueError(f"Missing checkpoint geometry for {arm} seed={seed}")
        _, target_stats = soft_matrix_for(recipe, bundle.profiles, seed)
        row = selected.iloc[0].to_dict()
        row.update(target_stats)
        row.update({"alpha": alpha, "tau_mult": tau_mult})
        rows.append(row)
    return pd.DataFrame(rows)


def run_s8c_alpha(
    bundle: DataBundle,
    args: argparse.Namespace,
    output_root: Path,
) -> pd.DataFrame:
    direction = reevaluate_s5a_checkpoints(
        bundle,
        args,
        output_root,
        arms=("Hard-Rand", "SoftInst-Rand"),
        seeds=args.seeds,
    )
    old_metrics, _ = load_s5a_tables(
        ("Hard-Rand", "SoftInst-Rand"),
        args.seeds,
    )
    metrics_parts = []
    geometry_parts = []
    for alpha, arm in ((0.0, "Hard-Rand"), (0.5, "SoftInst-Rand")):
        selected = old_metrics[old_metrics["arm"] == arm].copy()
        selected["alpha"] = alpha
        selected["tau_mult"] = 1.0
        selected["dose_arm"] = arm
        metrics_parts.append(selected)
        geometry_parts.append(
            existing_geometry_for_dose(
                direction,
                bundle,
                arm=arm,
                alpha=alpha,
                tau_mult=1.0,
                seeds=args.seeds,
            )
        )
    for alpha in ALPHA_GRID:
        if abs(alpha) <= EPS or abs(alpha - 0.5) <= EPS:
            continue
        recipe = alpha_recipe(alpha)
        for seed in args.seeds:
            result = run_training(
                recipe=recipe,
                seed=seed,
                stage="s8c_alpha",
                collect_trajectory=False,
                output_root=output_root,
                bundle=bundle,
                args=args,
            )
            selected = result["metrics"].copy()
            selected["alpha"] = alpha
            selected["tau_mult"] = 1.0
            selected["dose_arm"] = recipe.name
            metrics_parts.append(selected)
            row = dict(result["geometry"])
            row["alpha"] = alpha
            row["tau_mult"] = 1.0
            geometry_parts.append(pd.DataFrame([row]))
    metrics = pd.concat(metrics_parts, ignore_index=True)
    geometry = pd.concat(geometry_parts, ignore_index=True)
    by_seed = attach_geometry(
        metrics,
        geometry,
        extra_columns=["alpha", "tau_mult"],
    )
    by_seed.to_csv(
        output_root / "dose_response_alpha_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    combined_summary(metrics, geometry, ["alpha", "tau_mult"]).to_csv(
        output_root / "dose_response_alpha_summary.csv",
        index=False,
        encoding="utf-8",
    )
    summarize_geometry(geometry, ["alpha", "tau_mult"]).to_csv(
        output_root / "dose_response_alpha_geometry_summary.csv",
        index=False,
        encoding="utf-8",
    )
    return by_seed


def run_s8c_tau(
    bundle: DataBundle,
    args: argparse.Namespace,
    output_root: Path,
) -> pd.DataFrame:
    direction = reevaluate_s5a_checkpoints(
        bundle,
        args,
        output_root,
        arms=("SoftInst-Rand",),
        seeds=args.seeds,
    )
    old_metrics, _ = load_s5a_tables(("SoftInst-Rand",), args.seeds)
    metrics_parts = []
    geometry_parts = []
    baseline = old_metrics.copy()
    baseline["alpha"] = 0.5
    baseline["tau_mult"] = 1.0
    baseline["dose_arm"] = "SoftInst-Rand"
    metrics_parts.append(baseline)
    geometry_parts.append(
        existing_geometry_for_dose(
            direction,
            bundle,
            arm="SoftInst-Rand",
            alpha=0.5,
            tau_mult=1.0,
            seeds=args.seeds,
        )
    )
    for tau_mult in TAU_MULT_GRID:
        if abs(tau_mult - 1.0) <= EPS:
            continue
        recipe = tau_recipe(tau_mult)
        for seed in args.seeds:
            result = run_training(
                recipe=recipe,
                seed=seed,
                stage="s8c_tau",
                collect_trajectory=False,
                output_root=output_root,
                bundle=bundle,
                args=args,
            )
            selected = result["metrics"].copy()
            selected["alpha"] = 0.5
            selected["tau_mult"] = tau_mult
            selected["dose_arm"] = recipe.name
            metrics_parts.append(selected)
            row = dict(result["geometry"])
            row["alpha"] = 0.5
            row["tau_mult"] = tau_mult
            geometry_parts.append(pd.DataFrame([row]))
    metrics = pd.concat(metrics_parts, ignore_index=True)
    geometry = pd.concat(geometry_parts, ignore_index=True)
    by_seed = attach_geometry(
        metrics,
        geometry,
        extra_columns=["alpha", "tau_mult"],
    )
    by_seed.to_csv(
        output_root / "dose_response_tau_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    combined_summary(metrics, geometry, ["alpha", "tau_mult"]).to_csv(
        output_root / "dose_response_tau_summary.csv",
        index=False,
        encoding="utf-8",
    )
    summarize_geometry(geometry, ["alpha", "tau_mult"]).to_csv(
        output_root / "dose_response_tau_geometry_summary.csv",
        index=False,
        encoding="utf-8",
    )
    return by_seed


def run_cross_tables(
    bundle: DataBundle,
    args: argparse.Namespace,
    output_root: Path,
) -> pd.DataFrame:
    soft_metrics = pd.read_csv(output_root / "soft_sphere_metrics_by_seed.csv")
    soft_geometry = pd.read_csv(output_root / "soft_sphere_geometry_by_seed.csv")
    s5_metrics, _ = load_s5a_tables(DIRECTION_ARMS, args.seeds)
    checkpoint_geometry = reevaluate_s5a_checkpoints(
        bundle,
        args,
        output_root,
        arms=DIRECTION_ARMS,
        seeds=args.seeds,
    )
    metrics = pd.concat([soft_metrics, s5_metrics], ignore_index=True)
    geometry = pd.concat(
        [soft_geometry, checkpoint_geometry],
        ignore_index=True,
    )
    geometry_summary = summarize_geometry(geometry, ["arm"])
    test10 = metrics[metrics["split"] == "test"]
    test10 = test10[test10["k"] == 10]
    wide_rows = []
    for arm in geometry_summary["arm"]:
        row = geometry_summary[geometry_summary["arm"] == arm].iloc[0].to_dict()
        for scorer, suffix in (
            ("plain_l1", "mask"),
            ("bank_euclidean", "euclidean"),
            ("bank_cosine", "cosine"),
            ("pai_a1", "pai_mean"),
            ("pai_official", "pai_official"),
        ):
            selected = test10[
                (test10["arm"] == arm) & (test10["scorer"] == scorer)
            ]
            row[f"{suffix}_pr_auc_mean"] = float(selected["pr_auc"].mean())
            row[f"{suffix}_roc_auc_mean"] = float(selected["roc_auc"].mean())
            row[f"{suffix}_event_recall_k10_mean"] = float(
                selected["event_recall_3d"].mean()
            )
        row["factorial_core"] = arm in {
            "Hard-Rand",
            "Hard-L2Reg-b0.20",
            "Hard-Sphere-t0.1",
            "UniformSoft-Rand",
            "PermSoft-Rand",
            "SoftInst-Rand",
        }
        wide_rows.append(row)
    summary = pd.DataFrame(wide_rows).sort_values(
        ["factorial_core", "mean_norm_active_mean"],
        ascending=[False, True],
    )
    summary.to_csv(
        output_root / "cosine_vs_norm_attribution_summary.csv",
        index=False,
        encoding="utf-8",
    )
    core_names = [
        "Hard-Rand",
        "Hard-L2Reg-b0.20",
        "UniformSoft-Rand",
        "PermSoft-Rand",
        "SoftInst-Rand",
        "Soft-Sphere-t0.1",
    ]
    core_columns = [
        "arm",
        "mean_norm_active_mean",
        "unit_effective_rank_test_normal_mean",
        "mean_pairwise_cosine_test_normal_mean",
        "cosine_pr_auc_mean",
    ]
    summary[summary["arm"].isin(core_names)][core_columns].to_csv(
        output_root / "cosine_vs_norm_core_6x4.csv",
        index=False,
        encoding="utf-8",
    )
    geometry.to_csv(
        output_root / "cosine_vs_norm_geometry_by_seed.csv",
        index=False,
        encoding="utf-8",
    )

    comparisons = []
    for tau in SOFT_SPHERE_TAUS:
        candidate = f"Soft-Sphere-t{tau:g}"
        comparisons.append((candidate, "SoftInst-Rand", f"t{tau:g}_vs_softinst"))
        comparisons.append(
            (
                candidate,
                f"Hard-Sphere-t{tau:g}",
                f"t{tau:g}_vs_hardsphere",
            )
        )
    for reference, label in (
        ("UniformSoft-Rand", "hard_l2_vs_uniform"),
        ("PermSoft-Rand", "hard_l2_vs_permsoft"),
        ("SoftInst-Rand", "hard_l2_vs_softinst"),
    ):
        comparisons.append(("Hard-L2Reg-b0.20", reference, label))
    metric_tests = paired_metric_tests(
        metrics,
        comparisons,
        scorers=("bank_euclidean", "bank_cosine", "pai_a1"),
    )
    geometry = geometry.copy()
    geometry["scorer"] = ""
    geometry_tests = paired_geometry_tests(geometry, comparisons)
    paired = pd.concat([metric_tests, geometry_tests], ignore_index=True)
    paired.to_csv(
        output_root / "cosine_vs_norm_paired_tests.csv",
        index=False,
        encoding="utf-8",
    )
    return summary


def trend_shape(x_values: list[float], y_values: list[float]) -> str:
    pairs = sorted(zip(x_values, y_values))
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if len(y) < 3 or not np.isfinite(y).all():
        return "insufficient"
    diffs = np.diff(y)
    scale = max(float(np.ptp(y)), EPS)
    tolerance = scale * 1e-6
    if np.all(diffs >= -tolerance):
        return "monotonic_increasing"
    if np.all(diffs <= tolerance):
        return "monotonic_decreasing"
    maximum_index = int(np.argmax(y))
    minimum_index = int(np.argmin(y))
    if 0 < maximum_index < len(y) - 1:
        return "inverted_U"
    if 0 < minimum_index < len(y) - 1:
        return "U_shaped"
    return "non_monotonic"


def format_number(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_executive_summary(output_root: Path) -> None:
    attribution = pd.read_csv(
        output_root / "cosine_vs_norm_attribution_summary.csv"
    )
    soft_pairs = pd.read_csv(output_root / "soft_sphere_paired_tests.csv")
    alpha_geometry = pd.read_csv(
        output_root / "dose_response_alpha_geometry_summary.csv"
    )
    tau_geometry = pd.read_csv(
        output_root / "dose_response_tau_geometry_summary.csv"
    )
    alpha_metrics = pd.read_csv(output_root / "dose_response_alpha_summary.csv")
    tau_metrics = pd.read_csv(output_root / "dose_response_tau_summary.csv")
    dynamics = pd.read_csv(output_root / "dynamics_trajectory_summary.csv")
    audit = pd.read_csv(output_root / "s5a_reproduction_audit_summary.csv")

    q1 = soft_pairs[
        (soft_pairs["metric"] == "PR-AUC")
        & (soft_pairs["scorer"].isin(["bank_cosine", "bank_euclidean", "pai_a1"]))
    ].copy()
    q1_table = []
    for row in q1.itertuples(index=False):
        q1_table.append(
            "| "
            + " | ".join(
                [
                    row.candidate_arm,
                    row.comparison_family,
                    row.scorer,
                    format_number(row.delta_mean, 6),
                    row.wins_text,
                    f"{row.p_value:.3g}",
                    f"{row.p_holm:.3g}",
                ]
            )
            + " |"
        )

    q2_arms = [
        "Hard-Rand",
        "Hard-L2Reg-b0.20",
        "UniformSoft-Rand",
        "PermSoft-Rand",
        "SoftInst-Rand",
    ]
    q2 = attribution[attribution["arm"].isin(q2_arms)].copy()
    q2_table = []
    for row in q2.itertuples(index=False):
        q2_table.append(
            "| "
            + " | ".join(
                [
                    row.arm,
                    format_number(row.mean_norm_active_mean),
                    format_number(row.unit_effective_rank_test_normal_mean),
                    format_number(row.mean_pairwise_cosine_test_normal_mean),
                    format_number(row.cosine_nn_margin_test_normal_mean),
                    format_number(row.cosine_pr_auc_mean),
                    format_number(row.euclidean_pr_auc_mean),
                    format_number(row.pai_mean_pr_auc_mean),
                ]
            )
            + " |"
        )

    dynamics_final = dynamics[dynamics["step"] == 620].copy()
    dynamics_initial = dynamics[dynamics["step"] == 0].copy()
    dynamics_table = []
    for arm in DYNAMICS_ARMS:
        initial = dynamics_initial[dynamics_initial["arm"] == arm]
        final = dynamics_final[dynamics_final["arm"] == arm]
        if initial.empty or final.empty:
            continue
        dynamics_table.append(
            "| "
            + " | ".join(
                [
                    arm,
                    format_number(initial["mean_norm_active_mean"].iloc[0]),
                    format_number(final["mean_norm_active_mean"].iloc[0]),
                    format_number(
                        initial["unit_effective_rank_mean"].iloc[0]
                    ),
                    format_number(
                        final["unit_effective_rank_mean"].iloc[0]
                    ),
                    format_number(initial["val_cos_pr_auc_mean"].iloc[0]),
                    format_number(final["val_cos_pr_auc_mean"].iloc[0]),
                ]
            )
            + " |"
        )

    def dose_rows(geometry, metrics, parameter: str) -> list[str]:
        test10 = metrics[
            (metrics["split"] == "test") & (metrics["k"] == 10)
        ]
        rows = []
        for value in sorted(geometry[parameter].unique()):
            geom = geometry[geometry[parameter] == value].iloc[0]
            metrics_row = {}
            for scorer in ("bank_cosine", "bank_euclidean", "pai_a1"):
                selected = test10[
                    (test10[parameter] == value)
                    & (test10["scorer"] == scorer)
                ]
                metrics_row[scorer] = (
                    float(selected["pr_auc_mean"].mean())
                    if not selected.empty
                    else float("nan")
                )
            rows.append(
                "| "
                + " | ".join(
                    [
                        format_number(float(value), 3),
                        format_number(geom["target_entropy_mean"]),
                        format_number(geom["offdiag_mass_ratio_mean"]),
                        format_number(geom["mean_norm_active_mean"]),
                        format_number(geom["unit_effective_rank_active_mean"]),
                        format_number(metrics_row["bank_cosine"]),
                        format_number(metrics_row["bank_euclidean"]),
                        format_number(metrics_row["pai_a1"]),
                    ]
                )
                + " |"
            )
        return rows

    alpha_table = dose_rows(alpha_geometry, alpha_metrics, "alpha")
    tau_table = dose_rows(tau_geometry, tau_metrics, "tau_mult")
    alpha_cos = alpha_metrics[
        (alpha_metrics["split"] == "test")
        & (alpha_metrics["k"] == 10)
        & (alpha_metrics["scorer"] == "bank_cosine")
    ]
    tau_cos = tau_metrics[
        (tau_metrics["split"] == "test")
        & (tau_metrics["k"] == 10)
        & (tau_metrics["scorer"] == "bank_cosine")
    ]
    alpha_shapes = {
        "norm": trend_shape(
            alpha_geometry["alpha"].tolist(),
            alpha_geometry["mean_norm_active_mean"].tolist(),
        ),
        "entropy": trend_shape(
            alpha_geometry["alpha"].tolist(),
            alpha_geometry["target_entropy_mean"].tolist(),
        ),
        "unit_rank": trend_shape(
            alpha_geometry["alpha"].tolist(),
            alpha_geometry["unit_effective_rank_active_mean"].tolist(),
        ),
        "cosine_pr": trend_shape(
            alpha_cos["alpha"].tolist(),
            alpha_cos["pr_auc_mean"].tolist(),
        ),
    }
    tau_shapes = {
        "norm": trend_shape(
            tau_geometry["tau_mult"].tolist(),
            tau_geometry["mean_norm_active_mean"].tolist(),
        ),
        "entropy": trend_shape(
            tau_geometry["tau_mult"].tolist(),
            tau_geometry["target_entropy_mean"].tolist(),
        ),
        "unit_rank": trend_shape(
            tau_geometry["tau_mult"].tolist(),
            tau_geometry["unit_effective_rank_active_mean"].tolist(),
        ),
        "cosine_pr": trend_shape(
            tau_cos["tau_mult"].tolist(),
            tau_cos["pr_auc_mean"].tolist(),
        ),
    }

    soft_vs_soft = q1[q1["reference_arm"] == "SoftInst-Rand"]
    soft_vs_hard = q1[q1["reference_arm"].str.startswith("Hard-Sphere")]
    q1_text = (
        "在三个温度与三类关键评分器上，Soft-Sphere 相对 SoftInst 的 PR-AUC "
        f"平均配对差值为 {format_number(float(soft_vs_soft['delta_mean'].mean()), 6)}；"
        "相对同温度 Hard-Sphere 的平均配对差值为 "
        f"{format_number(float(soft_vs_hard['delta_mean'].mean()), 6)}。"
    )
    if (soft_vs_soft["delta_mean"] <= 0).all():
        q1_text += " 逐项看，切断未归一化模长后相对 SoftInst 的点级增益均未保留。"
    else:
        q1_text += " 逐项看，模长归一化后仍有部分正残差，但必须结合置信区间判断。"

    q2_hard = q2[q2["arm"] == "Hard-L2Reg-b0.20"].iloc[0]
    q2_soft = q2[q2["arm"] == "SoftInst-Rand"].iloc[0]
    q2_text = (
        f"Hard-L2Reg-b0.20 的余弦 PR-AUC 为 {format_number(q2_hard['cosine_pr_auc_mean'])}，"
        f"SoftInst 为 {format_number(q2_soft['cosine_pr_auc_mean'])}；"
        f"二者欧氏 PR-AUC 分别为 {format_number(q2_hard['euclidean_pr_auc_mean'])} "
        f"和 {format_number(q2_soft['euclidean_pr_auc_mean'])}。"
        f"Hard-L2Reg 的单位方向有效秩为 {format_number(q2_hard['unit_effective_rank_test_normal_mean'])}，"
        f"SoftInst 为 {format_number(q2_soft['unit_effective_rank_test_normal_mean'])}；"
        f"单位球平均成对余弦分别为 {format_number(q2_hard['mean_pairwise_cosine_test_normal_mean'])} "
        f"和 {format_number(q2_soft['mean_pairwise_cosine_test_normal_mean'])}。"
    )
    if (
        q2_soft["unit_effective_rank_test_normal_mean"]
        > q2_hard["unit_effective_rank_test_normal_mean"]
    ):
        q2_text += (
            " 这与软权重保留更高方向维数、缓解纯径向收缩后方向坍缩的解释一致。"
        )
    else:
        q2_text += (
            " 本次单位方向谱指标未显示 SoftInst 的方向有效秩更高，需以动力学与近邻边距共同解释。"
        )

    max_audit = audit[[
        "max_abs_delta_pr_auc",
        "max_abs_delta_roc_auc",
        "max_abs_delta_event_recall_3d",
    ]].to_numpy().max()
    exact_all = bool(audit["exact_within_1e-12"].all())
    audit_text = (
        "S5a 重合 arm 的逐种子复现审计"
        + (
            "完全落在 1e-12 内。"
            if exact_all
            else (
                f"最大绝对误差为 {max_audit:.3e}；无轨迹的同命令直跑也出现同类"
                " GPU 重训数值差异，因此最终基准指标使用冻结 S5a，轨迹 run "
                "仅用于训练动力学。"
            )
        )
    )

    markdown = f"""# Exp S8 Executive Summary

Date: 2026-11-23 (Asia/Shanghai). Seeds: `20261116-20261120`.

## 核心答案

### (1) Soft-Sphere 是否使增益消失

{q1_text}

| Candidate | Comparison | Scorer | Delta PR-AUC | Wins | p | Holm p |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(q1_table)}

### (2) 为什么等范数约束在余弦上仍不足

{q2_text}

| Arm | Mean norm | Unit effective rank | Mean pairwise cosine | Cosine NN margin | Cosine PR-AUC | Euclidean PR-AUC | PAI-Mean PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(q2_table)}

| Arm | Norm step 0 | Norm step 620 | Unit ER step 0 | Unit ER step 620 | Val cosine PR step 0 | Val cosine PR step 620 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(dynamics_table)}

### (3) Alpha 与 tau_mult 的剂量-响应形状

- Alpha: norm=`{alpha_shapes['norm']}`, entropy=`{alpha_shapes['entropy']}`, unit_rank=`{alpha_shapes['unit_rank']}`, cosine_PR=`{alpha_shapes['cosine_pr']}`.
- Tau multiplier: norm=`{tau_shapes['norm']}`, entropy=`{tau_shapes['entropy']}`, unit_rank=`{tau_shapes['unit_rank']}`, cosine_PR=`{tau_shapes['cosine_pr']}`.

| Alpha | Target entropy | Offdiag mass ratio | Mean norm | Unit ER | Cosine PR-AUC | Euclidean PR-AUC | PAI-Mean PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(alpha_table)}

| Tau multiplier | Target entropy | Offdiag mass ratio | Mean norm | Unit ER | Cosine PR-AUC | Euclidean PR-AUC | PAI-Mean PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(tau_table)}

## Reproduction and integrity

- {audit_text}
- Soft-Sphere runs assert exactly 620 soft-loss calls, at least 1,240 tensor-gradient hook calls, nonzero off-diagonal target mass, and post-normalization norm 1.0 within `1e-4`.
- Training-trajectory snapshots save/restore Python, NumPy, CPU Torch, and CUDA RNG states.
- The fast trajectory monitor uses all validation positive active days plus 5,000 deterministic negative active days; final performance uses the full test set.

## Interpretation boundary

These results are causal attribution evidence for CERT r4.2 under the frozen 620-step protocol. They do not establish the same mechanism on CLUE-LDS, SMD, or live bank production data. Watchlist is absent from every encoder, memory bank, and score path.
"""
    (output_root / "S8_EXECUTIVE_SUMMARY.md").write_text(
        markdown,
        encoding="utf-8",
    )


def write_run_log(
    output_root: Path,
    command: str,
    wall_seconds: float,
) -> None:
    soft_geometry = pd.read_csv(
        output_root / "soft_sphere_geometry_by_seed.csv"
    )
    audit = pd.read_csv(output_root / "s5a_reproduction_audit_summary.csv")
    checkpoint_repro = pd.read_csv(
        output_root / "s5a_checkpoint_geometry_reproduction.csv"
    )
    dynamic = pd.read_csv(output_root / "dynamics_trajectory_by_seed.csv")
    alpha = pd.read_csv(output_root / "dose_response_alpha_by_seed.csv")
    tau = pd.read_csv(output_root / "dose_response_tau_by_seed.csv")
    hook_calls_ok = bool(
        soft_geometry["hook_calls"].eq(620).all()
        and soft_geometry["gradient_hook_calls"].ge(1240).all()
    )
    max_norm_deviation = float(
        (soft_geometry["mean_post_norm"] - 1.0).abs().max()
    )
    reproduction_max = float(
        audit[[
            "max_abs_delta_pr_auc",
            "max_abs_delta_roc_auc",
            "max_abs_delta_event_recall_3d",
        ]].to_numpy().max()
    )
    checkpoint_norm_delta = float(
        checkpoint_repro["norm_abs_delta"].max()
    )
    checkpoint_rank_delta = float(
        checkpoint_repro["rank_abs_delta"].max()
    )
    text = f"""# Exp S8 Run Log

Date: 2026-11-23 (Asia/Shanghai).

## Fixed protocol

- Dataset: CERT r4.2.
- Seeds: `20261116 20261117 20261118 20261119 20261120`.
- Steps: 620 (10 epochs x 62 steps), batch size 16.
- Scorers: `plain_l1`, `bank_euclidean`, `bank_cosine`, `pai_a1`, `pai_official`.
- Event K: 5, 10, 20 with the canonical 7-day cooldown.
- New writes were restricted to `{output_root}`. S1-S7, `results/canonical/`, `processed_*`, and historical archives were read-only.

## Command

```powershell
{command}
```

## Execution summary

- Wall time this invocation: {wall_seconds / 60.0:.2f} minutes.
- S8a runs: {len(soft_geometry)} (`3` temperatures x `5` seeds).
- S8b trajectory seed-arm rows: {len(dynamic) // 11} x 11 snapshots = {len(dynamic)} rows.
- S8c alpha rows: {len(alpha)} scorer/K rows across `7` alpha values and `5` seeds.
- S8c tau rows: {len(tau)} scorer/K rows across `5` tau_mult values and `5` seeds.

## Hook and gradient assertions

- Exact 620 hierarchy-loss calls on every Soft-Sphere run: {hook_calls_ok}.
- Every Soft-Sphere run has at least 1,240 gradient hook calls.
- Maximum absolute deviation of mean post-normalization norm from 1.0: {max_norm_deviation:.3e}.
- Off-diagonal soft mass must be positive on every Soft-Sphere optimization call.

## S5a reproduction

- Fresh S8b training versus stored S5a metrics: maximum absolute delta = {reproduction_max:.3e}.
- S5a checkpoint geometry reevaluation:
  - maximum absolute mean-norm delta = {checkpoint_norm_delta:.3e};
  - maximum absolute effective-rank delta = {checkpoint_rank_delta:.3e}.

Per-arm audit:

| Arm | Rows | Max PR delta | Max ROC delta | Max recall delta | Exact |
|---|---:|---:|---:|---:|---|
"""
    for _, row in audit.iterrows():
        text += (
            f"| {row['arm']} | {int(row['rows_compared'])} | "
            f"{row['max_abs_delta_pr_auc']:.3e} | "
            f"{row['max_abs_delta_roc_auc']:.3e} | "
            f"{row['max_abs_delta_event_recall_3d']:.3e} | "
            f"{row['exact_within_1e-12']} |\n"
        )
    if reproduction_max > EPS:
        text += (
            "\nReproduction boundary: a no-trajectory direct rerun of the same "
            "S5a command also differs numerically on this GPU. Strict identity "
            "is therefore claimed only for checkpoint geometry; causal benchmark "
            "metrics use the frozen S5a tables.\n"
        )
    text += """

## Trajectory RNG isolation

- Snapshot evaluation saves and restores Python `random`, NumPy, CPU Torch, and all CUDA RNG states.
- Snapshot points are exactly steps 0, 62, 124, ..., 620.
- Final downstream metrics are evaluated only after training, using the canonical full CERTr4.2 encoding and scorers.

## Artifact boundaries

- No file under `research/ccf-b/supplement_results/exp_s1_r4.2` through `exp_s7_10seed_unified` was written.
- No S5a/S5b/S5c/S5d result or checkpoint was overwritten.
- The two historical l2reg directories were read-only references.
"""
    (output_root / "RUN_LOG_S8.md").write_text(text, encoding="utf-8")


def run_smoke(
    bundle: DataBundle,
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    seed = args.seeds[0]
    recipe = recipe_from_name("Hard-Rand")
    result = run_training(
        recipe=recipe,
        seed=seed,
        stage="s8b",
        collect_trajectory=True,
        output_root=output_root,
        bundle=bundle,
        args=args,
    )
    old, _ = load_s5a_tables(("Hard-Rand",), [seed])
    keys = ["seed", "arm", "scorer", "split", "k"]
    merged = old[keys + ["pr_auc", "roc_auc", "event_recall_3d"]].merge(
        result["metrics"][keys + ["pr_auc", "roc_auc", "event_recall_3d"]],
        on=keys,
        suffixes=("_s5a", "_s8"),
        validate="one_to_one",
    )
    max_delta = max(
        float((merged[f"{field}_s5a"] - merged[f"{field}_s8"]).abs().max())
        for field in ("pr_auc", "roc_auc", "event_recall_3d")
    )
    report = {
        "seed": seed,
        "arm": "Hard-Rand",
        "rows_compared": len(merged),
        "max_abs_delta": max_delta,
        "exact_within_1e-12": max_delta <= EPS,
        "trajectory_steps": result["trajectory"]["step"].tolist(),
    }
    (output_root / "smoke_reproduction.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = load_data(args)
    start = time.time()
    experiments = args.experiments
    if "all" in experiments:
        experiments = [
            "s8a",
            "s8b",
            "s8c-alpha",
            "s8c-tau",
            "cross",
            "finalize",
        ]
    for experiment in experiments:
        print(f"\n=== {experiment} ===", flush=True)
        if experiment == "smoke":
            run_smoke(bundle, args, output_root)
        elif experiment == "s8a":
            run_s8a(bundle, args, output_root)
        elif experiment == "s8b":
            run_s8b(bundle, args, output_root)
        elif experiment == "s8c-alpha":
            run_s8c_alpha(bundle, args, output_root)
        elif experiment == "s8c-tau":
            run_s8c_tau(bundle, args, output_root)
        elif experiment == "cross":
            run_cross_tables(bundle, args, output_root)
        elif experiment == "finalize":
            wall_seconds = time.time() - start
            command = (
                "C:\\Users\\Lenovo\\.conda\\envs\\ts2vec\\python.exe "
                "research\\ccf-b\\supplement\\exp_s8_necessity_and_dynamics.py "
                + " ".join(sys.argv[1:])
            )
            write_executive_summary(output_root)
            write_run_log(output_root, command, wall_seconds)
            (output_root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "generated_at": pd.Timestamp.now(
                            tz="Asia/Shanghai"
                        ).isoformat(),
                        "seeds": args.seeds,
                        "scripts": str(Path(__file__).resolve()),
                        "wall_seconds": round(wall_seconds, 2),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    print(f"\nExp S8 output: {output_root}", flush=True)


if __name__ == "__main__":
    main()
