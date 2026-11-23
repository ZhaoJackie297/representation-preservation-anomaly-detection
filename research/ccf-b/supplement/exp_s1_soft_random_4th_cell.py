"""Exp S1 (P0): Missing 4th Cell — Soft Instance Contrast + Random Mini-Batching (10 seeds).

Place this script at: <repo_root>/research/ccf-b/supplement/exp_s1_soft_random_4th_cell.py
Run from <repo_root>, e.g.:
  python research/ccf-b/supplement/exp_s1_soft_random_4th_cell.py --dataset r4.2 --step-mode both
  python research/ccf-b/supplement/exp_s1_soft_random_4th_cell.py --dataset r5.2 --step-mode both

Why this script exists:
  Closes the 2x2 factorial table (Hard/Soft x Random/Coverage) across the exact locked
  10 seeds (20261116-20261125) and evaluates ALL 5 scorers (plain_l1, bank_cosine,
  bank_euclidean, pai_a1, pai_official) on the same encoder so training-end and
  scoring-end attributions are completely decoupled.
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


REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_DIR = REPO_ROOT / "runtime" / "cert"
PHASE0_DIR = REPO_ROOT / "research" / "ccf-b" / "phase0"
sys.path.insert(0, str(CERT_DIR))
sys.path.insert(0, str(PHASE0_DIR))

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
    load_r52,
)
from conditional_combination_r42 import incident_hit_ids  # noqa: E402


LOCKED_10_SEEDS = list(range(20261116, 20261126))
K_GRID = (5, 10, 20)
PAI_OFF_W_EUC = 0.6
PAI_OFF_W_G = 0.4
PAI_OFF_W_Q = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp S1: 10-seed Soft + Random 4th cell")
    parser.add_argument("--dataset", choices=["r4.2", "r5.2"], required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=LOCKED_10_SEEDS,
    )
    parser.add_argument(
        "--step-mode",
        choices=["10ep", "matched", "both"],
        default="both",
        help="10ep = standard 10 epochs (620 / 1250 steps); matched = exact per-seed A1-Cov step count.",
    )
    parser.add_argument(
        "--also-hard-random-matched",
        action="store_true",
        help="Also run Hard + Random at matched step count to separate step-budget from batch-structure.",
    )
    parser.add_argument(
        "--r52-matched-steps",
        type=int,
        default=2300,
        help="Fallback matched step count on r5.2 if per-seed coverage CSV is absent.",
    )
    parser.add_argument(
        "--r52-training-summary",
        type=Path,
        default=None,
        help="r5.2 matrix summary.json containing per-seed A1-Cov optimizer steps.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_matched_step_map(
    dataset: str,
    seeds: list[int],
    r52_default: int,
    r52_training_summary: Path | None = None,
) -> dict[int, int]:
    if dataset == "r4.2":
        csv_path = REPO_ROOT / "research" / "ccf-b" / "r4.2_10seed_coverage_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return {int(r.seed): int(r.batches) for r in df.itertuples(index=False)}
        return {s: 1145 for s in seeds}
    # r5.2: prefer the exact matrix summary, either passed explicitly or stored
    # in the repository.  Its per-run A1-Cov optimizer step count is the
    # matched budget used by this supplement.
    candidates = []
    if r52_training_summary is not None:
        candidates.append(r52_training_summary)
    candidates.append(REPO_ROOT / "research" / "ccf-b" / "r5.2_10seed_training_summary.json")
    for json_path in candidates:
        if not json_path.exists():
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(meta, dict) and "runs" in meta:
            out = {}
            for s in seeds:
                run = meta["runs"].get(str(s), {})
                steps = run.get("models", {}).get("A1-Cov", {}).get("optimizer_steps")
                if steps is None:
                    steps = run.get("coverage", {}).get("training_batch_count")
                if steps is not None:
                    out[int(s)] = int(steps)
            if len(out) == len(seeds):
                return out
        elif isinstance(meta, dict):
            out = {}
            for s in seeds:
                record = meta.get(str(s))
                if isinstance(record, dict):
                    steps = record.get("batches", record.get("optimizer_steps"))
                    if steps is not None:
                        out[int(s)] = int(steps)
            if len(out) == len(seeds):
                return out
    return {s: r52_default for s in seeds}


def build_random_batch_table(
    user_count: int,
    batch_size: int,
    n_steps: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """Standard without-replacement epoch-shuffled random batches truncated to n_steps."""
    rng = np.random.default_rng(seed)
    batches: list[tuple[int, ...]] = []
    batches_per_epoch = user_count // batch_size
    while len(batches) < n_steps:
        perm = rng.permutation(user_count)
        for b in range(batches_per_epoch):
            start = b * batch_size
            batches.append(tuple(int(x) for x in perm[start : start + batch_size]))
            if len(batches) >= n_steps:
                break
    return batches


def align_profile_raw(csv_path: Path, users: np.ndarray, dates: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(
        csv_path,
        usecols=["user_id", "date", *PROFILE_FEATURES],
        parse_dates=["date"],
    )
    index = pd.MultiIndex.from_product(
        [users.tolist(), pd.to_datetime(dates)], names=["user_id", "date"]
    )
    aligned = frame.set_index(["user_id", "date"]).reindex(index)
    vals = aligned[PROFILE_FEATURES].to_numpy(dtype=np.float64).reshape(
        len(users), len(dates), len(PROFILE_FEATURES)
    )
    return sanitize_user_day_features(vals, count_dimensions=len(RAW_FEATURES))


def evaluate_all_splits(
    dataset_name: str,
    method_name: str,
    signal: np.ndarray,
    data: dict,
    active: np.ndarray,
    train_end: int,
    val_end: int | None,
    seed: int,
) -> tuple[list[dict], dict[str, list[str]]]:
    rank = signal.astype(np.float32, copy=True)
    rank[~np.isfinite(rank)] = -np.inf
    rows: list[dict] = []
    hit_map: dict[str, list[str]] = {}

    if dataset_name == "r4.2":
        assert val_end is not None
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
    else:
        test_start = train_end + 1
        end = len(data["dates"]) - 1
        pm = point_metrics(signal, data, active, test_start, end)
        for k in K_GRID:
            alerts = build_alerts(rank, active, k)
            res = evaluate_alert_system(data, alerts, method_name, k, external=True)
            rows.append(
                {
                    "seed": seed,
                    "method": method_name,
                    "split": "test",
                    "k": k,
                    **compact_metric(res, None),
                    **pm,
                }
            )
            if k == 10:
                hits = res["external_test"]["horizons"]["3"]["incident_ids"]
                hit_map[f"{method_name}__test_k10"] = sorted(str(x) for x in hits)
    return rows, hit_map


def main() -> None:
    args = parse_args()
    out_root = args.output_dir or (
        REPO_ROOT / "research" / "ccf-b" / "supplement_results" / f"exp_s1_{args.dataset}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    tag = "r42" if args.dataset == "r4.2" else "r52"
    processed = CERT_DIR / f"processed_{tag}"
    ds = np.load(processed / f"{args.dataset}_sequence_dataset.npz", allow_pickle=False)
    values = ds["values"].astype(np.float32)
    users = ds["users"].astype(str)
    dates = ds["dates"].astype(str)
    active = ds["active_mask"][:, :, 0].astype(bool)
    train_end = int(ds["train_end_index"])
    val_end = int(ds["val_end_index"]) if "val_end_index" in ds else None
    train_values = values[:, : train_end + 1]
    user_count = values.shape[0]

    data = load_r42() if args.dataset == "r4.2" else load_r52()
    raw_csv = processed / f"{args.dataset}_user_day_features.csv"
    profile_raw = align_profile_raw(raw_csv, users, dates)
    profiles = build_user_profiles(profile_raw, active, train_end)
    soft_labels, dist_meta = soft_instance_labels(profiles, alpha=args.alpha)
    (out_root / "distance_metadata.json").write_text(
        json.dumps(dist_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Precompute raw amplitude components once (deterministic across seeds)
    raw_20 = align_raw_features(raw_csv, users, dates)
    G, Q = aggregate_components(
        raw_20, active, train_end, user_specific=True, aggregation="channel_mean"
    )
    G_z = global_train_z(G, active, train_end)
    Q_z = global_train_z(Q, active, train_end)

    steps_10ep = (user_count // args.batch_size) * 10
    matched_map = load_matched_step_map(
        args.dataset,
        args.seeds,
        args.r52_matched_steps,
        args.r52_training_summary,
    )

    arms: list[tuple[str, bool, str]] = []
    if args.step_mode in ("10ep", "both"):
        arms.append(("SoftInst-Rand-10ep", True, "10ep"))
    if args.step_mode in ("matched", "both"):
        arms.append(("SoftInst-Rand-Matched", True, "matched"))
    if args.also_hard_random_matched:
        arms.append(("Hard-Rand-Matched", False, "matched"))

    device_id = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{device_id}" if isinstance(device_id, int) else "cpu")

    all_rows: list[dict] = []
    all_hits: dict[str, dict[str, list[str]]] = {}

    for arm_name, use_soft, step_kind in arms:
        for seed in args.seeds:
            run_dir = out_root / arm_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_csv = run_dir / "metrics.csv"
            hits_json = run_dir / "hit_ids_k10.json"

            if metrics_csv.exists() and hits_json.exists() and not args.force:
                all_rows.extend(pd.read_csv(metrics_csv).to_dict(orient="records"))
                all_hits[f"{arm_name}__seed_{seed}"] = json.loads(
                    hits_json.read_text(encoding="utf-8")
                )
                print(f"[SKIP] {arm_name} seed={seed}", flush=True)
                continue

            t0 = time.time()
            n_steps = steps_10ep if step_kind == "10ep" else matched_map[seed]
            batch_table = build_random_batch_table(
                user_count=user_count,
                batch_size=args.batch_size,
                n_steps=n_steps,
                seed=seed,
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
            logs = model.fit_coverage_table(
                train_data=train_values,
                soft_labels=soft_labels if use_soft else None,
                batch_table=batch_table,
                n_epochs=1,
                verbose=False,
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

            repr_norms = np.linalg.norm(plain_repr, axis=-1).astype(np.float32)

            torch.save(model.net.state_dict(), run_dir / "model.pth")
            np.savez_compressed(
                run_dir / "scores.npz",
                plain_l1=masked_z.astype(np.float32),
                bank_cosine=cos_z.astype(np.float32),
                bank_euclidean=euc_z.astype(np.float32),
                pai_a1=pai_a1.astype(np.float32),
                pai_official=pai_off.astype(np.float32),
                repr_norm=repr_norms,
            )

            scorers = {
                f"{arm_name}__plain_l1": masked_z,
                f"{arm_name}__bank_cosine": cos_z,
                f"{arm_name}__bank_euclidean": euc_z,
                f"{arm_name}__pai_a1": pai_a1,
                f"{arm_name}__pai_official": pai_off,
            }
            seed_rows: list[dict] = []
            seed_hits: dict[str, list[str]] = {}
            for sc_name, sc_arr in scorers.items():
                r_list, h_map = evaluate_all_splits(
                    args.dataset, sc_name, sc_arr, data, active, train_end, val_end, seed
                )
                seed_rows.extend(r_list)
                seed_hits.update(h_map)

            pd.DataFrame(seed_rows).to_csv(metrics_csv, index=False, encoding="utf-8")
            hits_json.write_text(
                json.dumps(seed_hits, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (run_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "arm": arm_name,
                        "seed": seed,
                        "soft_instance": use_soft,
                        "n_steps": n_steps,
                        "final_loss": logs[-1]["loss"] if logs else None,
                        "mean_repr_norm_active": float(repr_norms[active].mean()),
                        "elapsed_sec": round(time.time() - t0, 2),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            all_rows.extend(seed_rows)
            all_hits[f"{arm_name}__seed_{seed}"] = seed_hits
            print(
                f"[DONE] {args.dataset} {arm_name} seed={seed} steps={n_steps} "
                f"time={time.time() - t0:.1f}s",
                flush=True,
            )

    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(out_root / "metrics_by_seed.csv", index=False, encoding="utf-8")
    summary = (
        full_df.groupby(["method", "split", "k"], as_index=False)
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
    summary.to_csv(out_root / "method_summary.csv", index=False, encoding="utf-8")
    (out_root / "all_hit_ids_k10.json").write_text(
        json.dumps(all_hits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== TEST K=10 SUMMARY ===")
    print(summary[(summary["split"] == "test") & (summary["k"] == 10)].to_string(index=False))


if __name__ == "__main__":
    main()
