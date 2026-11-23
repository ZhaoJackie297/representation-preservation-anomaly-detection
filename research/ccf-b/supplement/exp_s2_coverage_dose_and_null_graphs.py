"""Exp S2 (P0): Pure-A3 Coverage Dose-Response & Isolated Null-Graph Controls (A5a / A5b).

Place this script at: <repo_root>/research/ccf-b/supplement/exp_s2_coverage_dose_and_null_graphs.py
Run from <repo_root>, e.g.:
  python research/ccf-b/supplement/exp_s2_coverage_dose_and_null_graphs.py --objective soft
  python research/ccf-b/supplement/exp_s2_coverage_dose_and_null_graphs.py --objective hard

Key methodological fixes over the original multi-graph coverage sampler:
  1. Never mixes A5a / A5b random edges into the A3 semantic coverage batch table.
  2. Locks IDENTICAL total training steps (--total-steps 700) across ALL arms:
     dose_rand (~15%), dose_A3_50 (~50%), dose_A3_80 (~80%), dose_A3_90 (~90%),
     ctrl_A5a_only (80% of degree-preserving random graph), and ctrl_A5b_only (80% of permuted graph).
  3. Logs exact realized A3 / A5a / A5b edge coverage over the FULL training batch sequence.
  4. Evaluates plain_l1, bank_cosine, bank_euclidean, and pai_a1 on every trained encoder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances


REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_DIR = REPO_ROOT / "runtime" / "cert"
PHASE0_DIR = REPO_ROOT / "research" / "ccf-b" / "phase0"
CLUE_DIR = REPO_ROOT / "research" / "ccf-b" / "clue_lds"
sys.path.insert(0, str(CERT_DIR))
sys.path.insert(0, str(PHASE0_DIR))
sys.path.insert(0, str(CLUE_DIR))

from build_prior_graphs import (  # noqa: E402
    degree_preserving_random,
    permuted_prior,
    top_m_edges,
)
from common_coverage_sampler import (  # noqa: E402
    build_common_coverage_table,
    normalize_edge,
)
from coverage_soft_ts2vec import CoverageSoftTS2Vec  # noqa: E402
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
    evaluate_variant,
)
from e4_joint_instance_pai import bank_scores  # noqa: E402
from feature_contracts import (  # noqa: E402
    build_user_profiles,
    sanitize_user_day_features,
)
from quota_relationship_sim import load_r42  # noqa: E402


DEFAULT_SEEDS = [20261116, 20261117, 20261118, 20261119, 20261120]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp S2: Pure A3 dose-response & A5a/A5b controls")
    parser.add_argument("--objective", choices=["soft", "hard"], default="soft")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=700,
        help="Locked identical gradient step count across all arms (must exceed A3 90%% coverage batches).",
    )
    parser.add_argument("--top-m", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--designed-slots", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_gaussian_prior(profiles: np.ndarray) -> np.ndarray:
    d = pairwise_distances(profiles, metric="euclidean")
    upper = d[np.triu_indices(len(d), k=1)]
    h = max(float(np.median(upper)), 1e-6)
    P = np.exp(-(d ** 2) / (2.0 * h * h))
    np.fill_diagonal(P, 0.0)
    row_sums = np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    return (P / row_sums).astype(np.float32)


def random_batches(user_count: int, batch_size: int, n_batches: int, seed: int) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[int, ...]] = []
    per_ep = user_count // batch_size
    while len(out) < n_batches:
        perm = rng.permutation(user_count)
        for i in range(per_ep):
            out.append(tuple(int(x) for x in perm[i * batch_size : (i + 1) * batch_size]))
            if len(out) >= n_batches:
                break
    return out


def compute_realized_coverage(
    batches: list[tuple[int, ...]],
    graphs: dict[str, set[tuple[int, int]]],
) -> dict[str, float]:
    seen_pairs: set[tuple[int, int]] = set()
    for batch in batches:
        for u, v in combinations(sorted(batch), 2):
            seen_pairs.add(normalize_edge(u, v))
    return {
        name: float(len(edges & seen_pairs) / max(len(edges), 1))
        for name, edges in graphs.items()
    }


def make_fixed_step_table(
    *,
    arm_name: str,
    user_count: int,
    graphs: dict[str, set[tuple[int, int]]],
    seed: int,
    total_steps: int,
    batch_size: int,
    designed_slots: int,
) -> tuple[list[tuple[int, ...]], dict[str, float]]:
    if arm_name == "dose_rand":
        batches = random_batches(user_count, batch_size, total_steps, seed)
        return batches, compute_realized_coverage(batches, graphs)

    spec = {
        "dose_A3_50": ("A3", 0.50),
        "dose_A3_80": ("A3", 0.80),
        "dose_A3_90": ("A3", 0.90),
        "ctrl_A5a_only": ("A5a", 0.80),
        "ctrl_A5b_only": ("A5b", 0.80),
    }
    graph_key, target_cov = spec[arm_name]
    cov_res = build_common_coverage_table(
        user_count=user_count,
        target_edge_sets={graph_key: graphs[graph_key]},
        seed=seed,
        stage=0,
        batch_size=batch_size,
        designed_slots=designed_slots,
        target_coverage=target_cov,
        min_training_batches=0,
    )
    cov_batches = list(cov_res.coverage_batches)
    if len(cov_batches) > total_steps:
        raise RuntimeError(
            f"{arm_name} seed={seed} needed {len(cov_batches)} batches > total_steps={total_steps}. "
            f"Re-run with --total-steps {len(cov_batches) + 50}."
        )
    # Pad up to exact total_steps by repeating the covered batch sequence cyclically
    # (preserves exact target coverage level without leaking extra uncovered edges).
    if len(cov_batches) < total_steps:
        reps = (total_steps + len(cov_batches) - 1) // len(cov_batches)
        full_batches = (cov_batches * reps)[:total_steps]
    else:
        full_batches = cov_batches[:total_steps]
    # Shuffle batch order deterministically while keeping exact edge coverage invariant
    rng = np.random.default_rng(seed + 991)
    order = rng.permutation(len(full_batches))
    shuffled = [full_batches[int(i)] for i in order]
    return shuffled, compute_realized_coverage(shuffled, graphs)


def main() -> None:
    args = parse_args()
    out_root = args.output_dir or (
        REPO_ROOT / "research" / "ccf-b" / "supplement_results" / f"exp_s2_{args.objective}_r42"
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

    raw_csv = processed / "r4.2_user_day_features.csv"
    frame = pd.read_csv(raw_csv, usecols=["user_id", "date", *PROFILE_FEATURES], parse_dates=["date"])
    idx = pd.MultiIndex.from_product([users.tolist(), pd.to_datetime(dates)], names=["user_id", "date"])
    aligned = frame.set_index(["user_id", "date"]).reindex(idx)
    raw_23 = sanitize_user_day_features(
        aligned[PROFILE_FEATURES].to_numpy(dtype=np.float64).reshape(len(users), len(dates), len(PROFILE_FEATURES)),
        count_dimensions=len(RAW_FEATURES),
    )
    profiles = build_user_profiles(raw_23, active, train_end)
    soft_labels, _ = soft_instance_labels(profiles, alpha=0.5)

    # Construct A3, A5a, A5b cleanly from 47D training-period profile prior
    prior = build_gaussian_prior(profiles)
    edges_A3 = top_m_edges(prior, args.top_m)
    edges_A5a = degree_preserving_random(edges_A3, user_count, seed=20261116 + 2)
    perm_prior, _ = permuted_prior(prior, seed=20261116 + 1)
    edges_A5b = top_m_edges(perm_prior, args.top_m)
    graphs = {"A3": edges_A3, "A5a": edges_A5a, "A5b": edges_A5b}

    raw_20 = align_raw_features(raw_csv, users, dates)
    G, Q = aggregate_components(raw_20, active, train_end, user_specific=True, aggregation="channel_mean")
    G_z = global_train_z(G, active, train_end)
    Q_z = global_train_z(Q, active, train_end)

    arms = [
        "dose_rand",
        "dose_A3_50",
        "dose_A3_80",
        "dose_A3_90",
        "ctrl_A5a_only",
        "ctrl_A5b_only",
    ]
    use_soft = args.objective == "soft"
    device_id = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{device_id}" if isinstance(device_id, int) else "cpu")

    all_rows: list[dict] = []
    cov_rows: list[dict] = []

    for arm in arms:
        for seed in args.seeds:
            run_dir = out_root / arm / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_csv = run_dir / "metrics.csv"
            cov_json = run_dir / "coverage.json"

            if metrics_csv.exists() and cov_json.exists() and not args.force:
                all_rows.extend(pd.read_csv(metrics_csv).to_dict(orient="records"))
                cov_rows.append(json.loads(cov_json.read_text(encoding="utf-8")))
                print(f"[SKIP] {args.objective} {arm} seed={seed}", flush=True)
                continue

            t0 = time.time()
            batches, realized_cov = make_fixed_step_table(
                arm_name=arm,
                user_count=user_count,
                graphs=graphs,
                seed=seed,
                total_steps=args.total_steps,
                batch_size=args.batch_size,
                designed_slots=args.designed_slots,
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
                batch_size=args.batch_size,
                lambda_=0.5,
                tau_temp=1.5,
                soft_instance=use_soft,
                soft_temporal=False,
            )
            model.fit_coverage_table(
                train_data=train_values,
                soft_labels=soft_labels if use_soft else None,
                batch_table=batches,
                n_epochs=1,
                verbose=False,
            )

            encode_args = {"causal": True, "sliding_length": 1, "sliding_padding": 30, "batch_size": 256}
            plain_repr = np.asarray(model.encode(values, **encode_args), dtype=np.float32)
            masked_repr = np.asarray(model.encode(values, mask="mask_last", **encode_args), dtype=np.float32)
            masked_raw = np.abs(masked_repr - plain_repr).sum(axis=-1)
            cos_raw, euc_raw = bank_scores(plain_repr, plain_repr, active, train_end, device)

            masked_z = global_train_z(masked_raw, active, train_end)
            cos_z = global_train_z(cos_raw, active, train_end)
            euc_z = global_train_z(euc_raw, active, train_end)
            pai_a1 = euc_z + PAI_W_G * G_z + PAI_W_Q * Q_z

            seed_rows: list[dict] = []
            for sc_tag, sc_arr in [
                ("plain_l1", masked_z),
                ("bank_cosine", cos_z),
                ("bank_euclidean", euc_z),
                ("pai_a1", pai_a1),
            ]:
                for r in evaluate_variant(
                    f"{args.objective}__{arm}__{sc_tag}", sc_arr, data, active, train_end, val_end, seed
                ):
                    r["arm"] = arm
                    r["scorer"] = sc_tag
                    r["realized_A3_cov"] = realized_cov["A3"]
                    r["realized_A5a_cov"] = realized_cov["A5a"]
                    r["realized_A5b_cov"] = realized_cov["A5b"]
                    seed_rows.append(r)

            cov_rec = {
                "objective": args.objective,
                "arm": arm,
                "seed": seed,
                "total_steps": args.total_steps,
                **{f"realized_{k}_cov": v for k, v in realized_cov.items()},
                "elapsed_sec": round(time.time() - t0, 2),
            }
            pd.DataFrame(seed_rows).to_csv(metrics_csv, index=False, encoding="utf-8")
            cov_json.write_text(json.dumps(cov_rec, ensure_ascii=False, indent=2), encoding="utf-8")
            all_rows.extend(seed_rows)
            cov_rows.append(cov_rec)
            print(
                f"[DONE] {args.objective} {arm} seed={seed} "
                f"A3={realized_cov['A3']:.3f} A5a={realized_cov['A5a']:.3f} A5b={realized_cov['A5b']:.3f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_root / "dose_response_by_seed.csv", index=False, encoding="utf-8")
    pd.DataFrame(cov_rows).to_csv(out_root / "coverage_audit.csv", index=False, encoding="utf-8")
    summary = (
        df.groupby(["arm", "scorer", "split", "k"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            A3_cov=("realized_A3_cov", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            incident_hits_mean=("incident_hits", "mean"),
        )
        .sort_values(["split", "k", "scorer", "arm"])
    )
    summary.to_csv(out_root / "dose_response_summary.csv", index=False, encoding="utf-8")
    print("\n=== DOSE-RESPONSE TEST K=10 (PAI-A1 & BANK_EUCLIDEAN) ===")
    mask = (summary["split"] == "test") & (summary["k"] == 10) & (summary["scorer"].isin(["pai_a1", "bank_euclidean"]))
    print(summary[mask].to_string(index=False))


if __name__ == "__main__":
    main()
