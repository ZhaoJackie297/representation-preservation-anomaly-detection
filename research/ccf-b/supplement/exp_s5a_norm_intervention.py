"""Exp S5a (P0 Core Causal Intervention): Explicit Norm Regularization, Spherical
L2-Normalization + Temperature, and Semantics-Stripped Soft Controls on CERT r4.2.

Place at: <repo_root>/research/ccf-b/supplement/exp_s5a_norm_intervention.py
Run from <repo_root>:
  python research/ccf-b/supplement/exp_s5a_norm_intervention.py

Why this script exists (directly resolves Reviewer Critique I.1):
  Correlational evidence alone (Table 5: norm drops 3.46 -> 1.38 while PR-AUC rises)
  does not prove that implicit norm contraction causes the downstream gain.
  This script runs an exact 5-seed controlled intervention (620 random-batch steps,
  identical 5-layer dilated TS2Vec backbone, identical 5 downstream scorers, plus
  full representation geometry diagnostics) comparing 11 arms:
    1) Hard-Rand              : standard unnormalized hard TS2Vec baseline
    2) Hard-L2Reg-b0.02       : Hard-Rand + 0.02 * E[||z||_2^2]
    3) Hard-L2Reg-b0.10       : Hard-Rand + 0.10 * E[||z||_2^2]
    4) Hard-L2Reg-b0.30       : Hard-Rand + 0.30 * E[||z||_2^2]
    5) Hard-L2Reg-b1.00       : Hard-Rand + 1.00 * E[||z||_2^2]
    6) Hard-Sphere-t0.1       : Hard-Rand with per-level L2-normalized InfoNCE (tau=0.1)
    7) Hard-Sphere-t0.2       : Hard-Rand with per-level L2-normalized InfoNCE (tau=0.2)
    8) Hard-Sphere-t0.5       : Hard-Rand with per-level L2-normalized InfoNCE (tau=0.5)
    9) PermSoft-Rand          : Soft instance weights P W_I P^T (identical weight spectrum,
                                row sums, target entropy & implicit norm contraction,
                                ZERO alignment with true 47D user profiles)
   10) UniformSoft-Rand       : Constant off-diagonal soft weight = mean_{u!=v} W_I[u,v]
                                (pure uniform label smoothing without pairwise structure)
   11) SoftInst-Rand          : True 47D profile soft instance weights (gold reference)
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, ttest_rel
from sklearn.cluster import MiniBatchKMeans


REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_DIR = REPO_ROOT / "runtime" / "cert"
PHASE0_DIR = REPO_ROOT / "research" / "ccf-b" / "phase0"
sys.path.insert(0, str(CERT_DIR))
sys.path.insert(0, str(PHASE0_DIR))

import coverage_soft_ts2vec as cst_mod  # noqa: E402
from coverage_soft_ts2vec import CoverageSoftTS2Vec  # noqa: E402
from conditional_combination_r42 import incident_hit_ids  # noqa: E402
from e0_e1_representation_preservation import (  # noqa: E402
    align_raw_features,
    global_train_z,
)
from e2_softclt_instance import (  # noqa: E402
    PROFILE_FEATURES,
    RAW_FEATURES,
    soft_instance_labels,
)
from e3_multichannel_amplitude import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    aggregate_components,
    build_alerts,
    point_metrics,
)
from e4_joint_instance_pai import bank_scores  # noqa: E402
from feature_contracts import (  # noqa: E402
    build_user_profiles,
    sanitize_user_day_features,
)
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
)


DEFAULT_SEEDS = list(range(20261116, 20261121))  # 5 locked seeds matching S2
K_GRID = (5, 10, 20)
PAI_OFF_W_EUC = 0.6
PAI_OFF_W_G = 0.4
PAI_OFF_W_Q = 0.2

DEFAULT_ARMS = [
    "Hard-Rand",
    "Hard-L2Reg-b0.02",
    "Hard-L2Reg-b0.10",
    "Hard-L2Reg-b0.30",
    "Hard-L2Reg-b1.00",
    "Hard-Sphere-t0.1",
    "Hard-Sphere-t0.2",
    "Hard-Sphere-t0.5",
    "PermSoft-Rand",
    "UniformSoft-Rand",
    "SoftInst-Rand",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp S5a: Norm & Semantics Causal Intervention on CERT r4.2")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def build_random_batch_table(
    user_count: int, batch_size: int, n_steps: int, seed: int
) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    batches: list[tuple[int, ...]] = []
    batches_per_epoch = user_count // batch_size
    while len(batches) < n_steps:
        perm = rng.permutation(user_count)
        for b in range(batches_per_epoch):
            st = b * batch_size
            batches.append(tuple(int(x) for x in perm[st : st + batch_size]))
            if len(batches) >= n_steps:
                break
    return batches


def align_profile_raw(csv_path: Path, users: np.ndarray, dates: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(
        csv_path, usecols=["user_id", "date", *PROFILE_FEATURES], parse_dates=["date"]
    )
    index = pd.MultiIndex.from_product(
        [users.tolist(), pd.to_datetime(dates)], names=["user_id", "date"]
    )
    aligned = frame.set_index(["user_id", "date"]).reindex(index)
    vals = aligned[PROFILE_FEATURES].to_numpy(dtype=np.float64).reshape(
        len(users), len(dates), len(PROFILE_FEATURES)
    )
    return sanitize_user_day_features(vals, count_dimensions=len(RAW_FEATURES))


# ---------------------------------------------------------------------------
# Exact Per-Level Spherical Hierarchical Contrastive Loss (Hard + L2Norm + tau)
# ---------------------------------------------------------------------------

def _inst_cl_spherical(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.0)
    z1_n = F.normalize(z1, dim=-1, eps=1e-8)
    z2_n = F.normalize(z2, dim=-1, eps=1e-8)
    z = torch.cat([z1_n, z2_n], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2)) / tau  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2.0
    return loss


def _temp_cl_spherical(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z1_n = F.normalize(z1, dim=-1, eps=1e-8)
    z2_n = F.normalize(z2, dim=-1, eps=1e-8)
    z = torch.cat([z1_n, z2_n], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2)) / tau  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2.0
    return loss


def spherical_hier_cl(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: float,
    lambda_: float = 0.5,
    temporal_unit: int = 0,
) -> torch.Tensor:
    """Exact TS2Vec hierarchical hard contrastive loss with L2-normalization & temperature
    applied at every pooling scale so cosine logits are strictly bounded in [-1/tau, 1/tau]."""
    loss = z1.new_tensor(0.0)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0.0:
            loss = loss + lambda_ * _inst_cl_spherical(z1, z2, tau)
        if d >= temporal_unit and (1.0 - lambda_) != 0.0:
            loss = loss + (1.0 - lambda_) * _temp_cl_spherical(z1, z2, tau)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if lambda_ != 0.0:
            loss = loss + lambda_ * _inst_cl_spherical(z1, z2, tau)
        d += 1
    return loss / max(d, 1)


@contextlib.contextmanager
def patched_contrastive_loss(
    mode: str, param: float, call_counter: list[int]
):
    """Safely hooks all hierarchical contrastive loss entrypoints used by CoverageSoftTS2Vec.

    Asserts via `call_counter` after fit that the hook fired once per optimizer step,
    preventing any silent no-op if module aliases differ.
    """
    if mode == "none":
        yield
        return

    candidate_names = ("hier_CL_soft", "hierarchical_contrastive_loss")
    targets: list[tuple[object, str, Callable]] = []

    # Check coverage_soft_ts2vec and any imported soft_ts2vec / losses modules
    modules_to_check = [cst_mod]
    for m_name, m_obj in list(sys.modules.items()):
        if m_obj is None:
            continue
        if any(k in m_name for k in ("soft_ts2vec", "coverage_soft", "losses")):
            if m_obj not in modules_to_check:
                modules_to_check.append(m_obj)

    for mod in modules_to_check:
        for attr in candidate_names:
            fn = getattr(mod, attr, None)
            if callable(fn):
                targets.append((mod, attr, fn))

    if not targets:
        raise RuntimeError(
            f"Could not locate hier_CL_soft / hierarchical_contrastive_loss in {[m.__name__ for m in modules_to_check]}. "
            f"Inspect coverage_soft_ts2vec.py to see which loss function fit_coverage_table calls."
        )

    active_depth = [0]

    def make_wrapper(orig_fn: Callable) -> Callable:
        def wrapped(z1: torch.Tensor, z2: torch.Tensor, *args, **kwargs):
            if active_depth[0] > 0:
                return orig_fn(z1, z2, *args, **kwargs)
            active_depth[0] += 1
            try:
                call_counter[0] += 1
                if mode == "l2reg":
                    base_loss = orig_fn(z1, z2, *args, **kwargs)
                    sq_norm = 0.5 * (
                        z1.pow(2).sum(dim=-1).mean() + z2.pow(2).sum(dim=-1).mean()
                    )
                    return base_loss + param * sq_norm
                if mode == "sphere":
                    lam = float(kwargs.get("lambda_", kwargs.get("alpha", 0.5)))
                    t_unit = int(kwargs.get("temporal_unit", 0))
                    return spherical_hier_cl(z1, z2, tau=param, lambda_=lam, temporal_unit=t_unit)
                raise ValueError(f"Unknown mode: {mode}")
            finally:
                active_depth[0] -= 1

        return wrapped

    for mod, attr, orig_fn in targets:
        setattr(mod, attr, make_wrapper(orig_fn))
    try:
        yield
    finally:
        for mod, attr, orig_fn in targets:
            setattr(mod, attr, orig_fn)


# ---------------------------------------------------------------------------
# Representation Geometry + Eq. (7) Radial-Angular Diagnostics
# ---------------------------------------------------------------------------

def effective_rank(mat: np.ndarray) -> float:
    """Roy-Vetterli spectral entropy effective rank of centered active representations."""
    x = mat.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    # Use covariance eigenvalues for speed (64 x 64)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12]
    if len(eigvals) == 0:
        return 0.0
    p = eigvals / eigvals.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def user_geometry_metrics(
    reprs: np.ndarray,
    active: np.ndarray,
    profiles: np.ndarray,
    train_end: int,
    val_end: int,
) -> dict[str, float]:
    """Top-10 neighbor agreement & local distance Spearman on validation active users."""
    val_act = active[:, train_end + 1 : val_end + 1]
    val_rep = reprs[:, train_end + 1 : val_end + 1]
    valid_users = val_act.sum(axis=1) > 0
    u_means = []
    for u in np.where(valid_users)[0]:
        u_means.append(val_rep[u, val_act[u]].mean(axis=0))
    u_means = np.asarray(u_means, dtype=np.float64)

    # Standardize profiles exactly as in soft_instance_labels
    prof_v = profiles[valid_users].astype(np.float64)
    mu = np.nanmean(prof_v, axis=0, keepdims=True)
    sd = np.nanstd(prof_v, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    prof_z = np.nan_to_num((prof_v - mu) / sd, nan=0.0)

    d_prof = cdist(prof_z, prof_z, metric="euclidean")
    d_repr = cdist(u_means, u_means, metric="euclidean")
    n = len(u_means)
    np.fill_diagonal(d_prof, np.inf)
    np.fill_diagonal(d_repr, np.inf)

    k = 10
    agrees = []
    spears = []
    for i in range(n):
        nn_p = set(np.argsort(d_prof[i])[:k].tolist())
        nn_r = set(np.argsort(d_repr[i])[:k].tolist())
        agrees.append(len(nn_p & nn_r) / k)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        r_val = spearmanr(d_prof[i, mask], d_repr[i, mask])[0]
        if np.isfinite(r_val):
            spears.append(float(r_val))

    return {
        "val_knn10_agreement": float(np.mean(agrees)),
        "val_local_dist_spearman": float(np.mean(spears)),
    }


def radial_norm_diagnostics(
    reprs: np.ndarray,
    values: np.ndarray,
    G: np.ndarray,
    active: np.ndarray,
    labels: np.ndarray,
    train_end: int,
    val_end: int,
    device: torch.device,
) -> dict[str, float]:
    tr_repr = reprs[:, : train_end + 1][active[:, : train_end + 1]]
    k_clusters = min(500, max(1, int(round(0.1 * len(tr_repr)))))
    km = MiniBatchKMeans(
        n_clusters=k_clusters,
        init="k-means++",
        random_state=2024,
        batch_size=max(8192, k_clusters),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    ).fit(tr_repr)
    bank = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device)
    bank_r = torch.linalg.norm(bank, dim=1)
    bank_u = F.normalize(bank, dim=1)

    test_mask = np.zeros_like(active, dtype=bool)
    test_mask[:, val_end + 1 :] = active[:, val_end + 1 :]
    q_arr = reprs[test_mask]
    y_test = labels[test_mask]
    g_test = G[test_mask]
    xn_test = np.linalg.norm(values[:, :, :20], axis=-1)[test_mask]
    q_norms = np.linalg.norm(q_arr, axis=-1)

    rad_sq_l, tot_sq_l = [], []
    with torch.inference_mode():
        for st in range(0, len(q_arr), 4096):
            q = torch.from_numpy(q_arr[st : st + 4096]).to(device)
            rq = torch.linalg.norm(q, dim=1)
            qu = F.normalize(q, dim=1)
            d_cos = (1.0 - qu @ bank_u.T).clamp_min_(0.0)
            euc = torch.cdist(q, bank)
            _, top3_idx = euc.topk(3, dim=1, largest=False)
            rc = bank_r[top3_idx]
            dcos_sel = torch.gather(d_cos, 1, top3_idx)
            rad_sq = (rq.unsqueeze(1) - rc) ** 2
            ang_tm = 2.0 * rq.unsqueeze(1) * rc * dcos_sel
            rad_sq_l.append(rad_sq.mean(dim=1).cpu().numpy())
            tot_sq_l.append((rad_sq + ang_tm).mean(dim=1).cpu().numpy())

    rad_share = np.concatenate(rad_sq_l) / np.maximum(np.concatenate(tot_sq_l), 1e-12)
    norm_m = y_test == 0
    anom_m = y_test == 1
    fin_g = np.isfinite(q_norms) & np.isfinite(g_test)
    fin_x = np.isfinite(q_norms) & np.isfinite(xn_test)

    act_norms = np.linalg.norm(reprs[active], axis=-1)
    return {
        "mean_norm_active": float(act_norms.mean()),
        "std_norm_active": float(act_norms.std()),
        "cv_norm_active": float(act_norms.std() / max(act_norms.mean(), 1e-8)),
        "mean_norm_test_normal": float(q_norms[norm_m].mean()),
        "mean_norm_test_anom": float(q_norms[anom_m].mean()),
        "effective_rank_active": effective_rank(reprs[active]),
        "effective_rank_val": effective_rank(reprs[:, train_end + 1 : val_end + 1][active[:, train_end + 1 : val_end + 1]]),
        "spearman_norm_vs_xnorm": float(spearmanr(q_norms[fin_x], xn_test[fin_x])[0]),
        "spearman_norm_vs_magG": float(spearmanr(q_norms[fin_g], g_test[fin_g])[0]),
        "radial_energy_share_normal": float(rad_share[norm_m].mean()),
        "radial_energy_share_anom": float(rad_share[anom_m].mean()),
    }


def evaluate_r42(
    method_name: str,
    signal: np.ndarray,
    data: dict,
    active: np.ndarray,
    train_end: int,
    val_end: int,
    seed: int,
) -> tuple[list[dict], dict[str, list[str]]]:
    rank = signal.astype(np.float32, copy=True)
    rank[~np.isfinite(rank)] = -np.inf
    rows: list[dict] = []
    hit_map: dict[str, list[str]] = {}
    bounds = {
        "validation": (train_end + 1, val_end),
        "test": (val_end + 1, len(data["dates"]) - 1),
    }
    for split, (start, end) in bounds.items():
        pm = point_metrics(signal, data, active, start, end)
        for k in K_GRID:
            alerts = build_alerts(rank, active, k)
            res = evaluate_alert_system(data, alerts, method_name, k, external=False)
            rows.append(
                {
                    "seed": seed,
                    "method": method_name,
                    "split": split,
                    "k": k,
                    **compact_metric(res, split),
                    **pm,
                }
            )
            if split == "test" and k == 10:
                hits = incident_hit_ids(
                    data["users"], data["dates"], alerts, data["incidents"], start, end
                )
                hit_map[f"{method_name}__test_k10"] = sorted(str(x) for x in hits)
    return rows, hit_map


def parse_arm_spec(arm: str) -> tuple[bool, str, float, str]:
    """Returns (use_soft, loss_mode, param, soft_variant)."""
    if arm == "Hard-Rand":
        return False, "none", 0.0, "none"
    if arm.startswith("Hard-L2Reg-b"):
        return False, "l2reg", float(arm.split("Hard-L2Reg-b", 1)[1]), "none"
    if arm.startswith("Hard-Sphere-t"):
        return False, "sphere", float(arm.split("Hard-Sphere-t", 1)[1]), "none"
    if arm == "PermSoft-Rand":
        return True, "none", 0.0, "perm"
    if arm == "UniformSoft-Rand":
        return True, "none", 0.0, "uniform"
    if arm == "SoftInst-Rand":
        return True, "none", 0.0, "profile"
    raise ValueError(f"Unknown arm spec: {arm}")


def build_soft_matrix(
    true_soft: np.ndarray, variant: str, seed: int
) -> np.ndarray | None:
    if variant == "none":
        return None
    if variant == "profile":
        return true_soft
    n = true_soft.shape[0]
    if variant == "perm":
        # Symmetric conjugate permutation P W P^T preserves diagonal (=1), spectrum,
        # row sums, entropy, and off-diagonal distribution while destroying user-profile semantics.
        rng = np.random.default_rng(seed + 77777)
        p = rng.permutation(n)
        return true_soft[np.ix_(p, p)].copy()
    if variant == "uniform":
        off_mask = ~np.eye(n, dtype=bool)
        mean_off = float(true_soft[off_mask].mean())
        mat = np.full_like(true_soft, mean_off)
        np.fill_diagonal(mat, np.diag(true_soft))
        return mat
    raise ValueError(f"Unknown soft_variant: {variant}")


def main() -> None:
    args = parse_args()
    out_root = args.output_dir or (
        REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s5a_norm_intervention"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    processed = CERT_DIR / "processed_r42"
    ds = np.load(processed / "r4.2_sequence_dataset.npz", allow_pickle=False)
    values = ds["values"].astype(np.float32)
    users = ds["users"].astype(str)
    dates = ds["dates"].astype(str)
    active = ds["active_mask"][:, :, 0].astype(bool)
    train_end = int(ds["train_end_index"])
    val_end = int(ds["val_end_index"])
    train_values = values[:, : train_end + 1]
    user_count = values.shape[0]

    data = load_r42()
    labels = data["labels"]
    raw_csv = processed / "r4.2_user_day_features.csv"
    profile_raw = align_profile_raw(raw_csv, users, dates)
    profiles = build_user_profiles(profile_raw, active, train_end)
    true_soft_labels, dist_meta = soft_instance_labels(profiles, alpha=args.alpha)

    raw_20 = align_raw_features(raw_csv, users, dates)
    G, Q = aggregate_components(
        raw_20, active, train_end, user_specific=True, aggregation="channel_mean"
    )
    G_z = global_train_z(G, active, train_end)
    Q_z = global_train_z(Q, active, train_end)

    n_steps = (user_count // args.batch_size) * args.epochs
    device_id = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{device_id}" if isinstance(device_id, int) else "cpu")

    all_rows: list[dict] = []
    geom_rows: list[dict] = []

    for arm in args.arms:
        use_soft, loss_mode, param, soft_variant = parse_arm_spec(arm)
        for seed in args.seeds:
            run_dir = out_root / arm / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_csv = run_dir / "metrics.csv"
            geom_json = run_dir / "geometry.json"

            if metrics_csv.exists() and geom_json.exists() and not args.force:
                all_rows.extend(pd.read_csv(metrics_csv).to_dict(orient="records"))
                geom_rows.append(json.loads(geom_json.read_text(encoding="utf-8")))
                print(f"[SKIP] {arm} seed={seed}", flush=True)
                continue

            t0 = time.time()
            batch_table = build_random_batch_table(user_count, args.batch_size, n_steps, seed)
            soft_mat = build_soft_matrix(true_soft_labels, soft_variant, seed)

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
                batch_size=args.batch_size,
                lambda_=0.5,
                tau_temp=1.5,
                soft_instance=use_soft,
                soft_temporal=False,
            )

            hook_calls = [0]
            with patched_contrastive_loss(loss_mode, param, hook_calls):
                logs = model.fit_coverage_table(
                    train_data=train_values,
                    soft_labels=soft_mat,
                    batch_table=batch_table,
                    n_epochs=1,
                    verbose=False,
                )

            if loss_mode != "none" and hook_calls[0] < n_steps:
                raise RuntimeError(
                    f"Loss hook for arm={arm} fired {hook_calls[0]} times (expected >= {n_steps})! "
                    f"Aborting to prevent silent unpatched training."
                )

            encode_args = {
                "causal": True,
                "sliding_length": 1,
                "sliding_padding": 30,
                "batch_size": 256,
            }
            plain_repr = np.asarray(model.encode(values, **encode_args), dtype=np.float32)
            masked_repr = np.asarray(
                model.encode(values, mask="mask_last", **encode_args), dtype=np.float32
            )
            masked_raw = np.abs(masked_repr - plain_repr).sum(axis=-1)
            cos_raw, euc_raw = bank_scores(plain_repr, plain_repr, active, train_end, device)

            masked_z = global_train_z(masked_raw, active, train_end)
            cos_z = global_train_z(cos_raw, active, train_end)
            euc_z = global_train_z(euc_raw, active, train_end)
            pai_a1 = euc_z + PAI_W_G * G_z + PAI_W_Q * Q_z
            pai_off = PAI_OFF_W_EUC * euc_z + PAI_OFF_W_G * G_z + PAI_OFF_W_Q * Q_z

            # Geometry & Eq. (7) diagnostics
            g_rad = radial_norm_diagnostics(
                plain_repr, values, G, active, labels, train_end, val_end, device
            )
            g_usr = user_geometry_metrics(plain_repr, active, profiles, train_end, val_end)
            g_rec = {
                "arm": arm,
                "seed": seed,
                "n_steps": n_steps,
                "hook_calls": hook_calls[0],
                "final_loss": float(logs[-1]["loss"]) if logs else float("nan"),
                **g_rad,
                **g_usr,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            geom_json.write_text(json.dumps(g_rec, ensure_ascii=False, indent=2), encoding="utf-8")
            torch.save(model.net.state_dict(), run_dir / "model.pth")

            scorers = {
                f"{arm}__plain_l1": masked_z,
                f"{arm}__bank_cosine": cos_z,
                f"{arm}__bank_euclidean": euc_z,
                f"{arm}__pai_a1": pai_a1,
                f"{arm}__pai_official": pai_off,
            }
            seed_rows: list[dict] = []
            seed_hits: dict[str, list[str]] = {}
            for sc_name, sc_arr in scorers.items():
                r_list, h_map = evaluate_r42(
                    sc_name, sc_arr, data, active, train_end, val_end, seed
                )
                seed_rows.extend(r_list)
                seed_hits.update(h_map)

            pd.DataFrame(seed_rows).to_csv(metrics_csv, index=False, encoding="utf-8")
            (run_dir / "hit_ids_k10.json").write_text(
                json.dumps(seed_hits, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            all_rows.extend(seed_rows)
            geom_rows.append(g_rec)
            print(
                f"[DONE] {arm:18s} seed={seed} norm={g_rec['mean_norm_active']:.3f} "
                f"effRank={g_rec['effective_rank_active']:.2f} knn10={g_rec['val_knn10_agreement']:.3f} "
                f"({g_rec['elapsed_sec']:.1f}s)",
                flush=True,
            )

    # Save aggregate tables
    full_df = pd.DataFrame(all_rows)
    full_df["arm"] = full_df["method"].str.split("__").str[0]
    full_df["scorer"] = full_df["method"].str.split("__").str[1]
    full_df.to_csv(out_root / "metrics_by_seed.csv", index=False, encoding="utf-8")

    summary = (
        full_df.groupby(["arm", "scorer", "split", "k"], as_index=False)
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
        .sort_values(["split", "k", "scorer", "arm"])
    )
    summary.to_csv(out_root / "method_summary.csv", index=False, encoding="utf-8")

    gdf = pd.DataFrame(geom_rows)
    gdf.to_csv(out_root / "geometry_by_seed.csv", index=False, encoding="utf-8")
    gsum = gdf.groupby("arm", as_index=False).mean(numeric_only=True)
    gsum.to_csv(out_root / "geometry_summary.csv", index=False, encoding="utf-8")

    # Paired t-tests vs Hard-Rand and vs SoftInst-Rand on test K=10
    test10 = full_df[(full_df["split"] == "test") & (full_df["k"] == 10)]
    paired_rows = []
    for ref_arm in ("Hard-Rand", "SoftInst-Rand"):
        if ref_arm not in set(test10["arm"]):
            continue
        for sc in ("plain_l1", "bank_cosine", "bank_euclidean", "pai_a1"):
            ref_df = test10[(test10["arm"] == ref_arm) & (test10["scorer"] == sc)].set_index("seed")
            for arm in args.arms:
                if arm == ref_arm:
                    continue
                cur_df = test10[(test10["arm"] == arm) & (test10["scorer"] == sc)].set_index("seed")
                common = sorted(set(ref_df.index) & set(cur_df.index))
                if len(common) < 2:
                    continue
                d_pr = cur_df.loc[common, "pr_auc"].to_numpy() - ref_df.loc[common, "pr_auc"].to_numpy()
                d_roc = cur_df.loc[common, "roc_auc"].to_numpy() - ref_df.loc[common, "roc_auc"].to_numpy()
                d_k10 = cur_df.loc[common, "incident_hits"].to_numpy() - ref_df.loc[common, "incident_hits"].to_numpy()
                paired_rows.append(
                    {
                        "arm": arm,
                        "ref_arm": ref_arm,
                        "scorer": sc,
                        "n_seeds": len(common),
                        "delta_pr_mean": float(d_pr.mean()),
                        "delta_pr_wins": f"{int((d_pr > 0).sum())}/{len(common)}",
                        "delta_pr_p": float(ttest_rel(cur_df.loc[common, "pr_auc"], ref_df.loc[common, "pr_auc"]).pvalue),
                        "delta_roc_mean": float(d_roc.mean()),
                        "delta_k10_hits_mean": float(d_k10.mean()),
                    }
                )
    pd.DataFrame(paired_rows).to_csv(
        out_root / "paired_vs_hard_and_soft.csv", index=False, encoding="utf-8"
    )

    print("\n=== S5a GEOMETRY SUMMARY ===")
    print(
        gsum[
            [
                "arm",
                "mean_norm_active",
                "effective_rank_active",
                "val_knn10_agreement",
                "val_local_dist_spearman",
                "radial_energy_share_anom",
            ]
        ].to_string(index=False)
    )
    print("\n=== S5a TEST K=10 (bank_cosine & pai_a1) ===")
    view = summary[
        (summary["split"] == "test")
        & (summary["k"] == 10)
        & (summary["scorer"].isin(["plain_l1", "bank_cosine", "bank_euclidean", "pai_a1"]))
    ]
    print(view[["arm", "scorer", "pr_auc_mean", "roc_auc_mean", "incident_hits_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
