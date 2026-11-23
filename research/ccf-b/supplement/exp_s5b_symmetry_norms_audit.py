"""Exp S5b (P0 Symmetry & Consistency Closure):
  1) hard-r42-10ep : 10-seed Hard-Rand-10ep on CERT r4.2 (all 5 scorers + 10-seed mean norm & effRank)
                     -> fills cosine/euclidean dashes & replaces 3-seed norm (3.464) in Table 4
  2) hard-r52-full : 10-seed Hard-Rand-10ep & Hard-Rand-Matched on CERT r5.2 (matching exact per-seed
                     A1-Cov steps 2,631-2,654) + 10-seed norms & effRank
                     -> closes Reviewer Critique I.3 (step-budget confounder on r5.2!)
  3) audit-iforest-t0 : CPU-only verification of iForest-27D (500 trees) exact 8-decimal metrics and
                        scan of existing repo CSVs for T0+Mask (0.007440 vs 0.007447).

Place at: <repo_root>/research/ccf-b/supplement/exp_s5b_symmetry_norms_audit.py
Run from <repo_root>:
  python research/ccf-b/supplement/exp_s5b_symmetry_norms_audit.py hard-r42-10ep
  python research/ccf-b/supplement/exp_s5b_symmetry_norms_audit.py hard-r52-full
  python research/ccf-b/supplement/exp_s5b_symmetry_norms_audit.py audit-iforest-t0
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
from sklearn.ensemble import IsolationForest


REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_DIR = REPO_ROOT / "runtime" / "cert"
PHASE0_DIR = REPO_ROOT / "research" / "ccf-b" / "phase0"
sys.path.insert(0, str(CERT_DIR))
sys.path.insert(0, str(PHASE0_DIR))

from coverage_soft_ts2vec import CoverageSoftTS2Vec  # noqa: E402
from conditional_combination_r42 import incident_hit_ids  # noqa: E402
from e0_e1_representation_preservation import (  # noqa: E402
    align_raw_features,
    global_train_z,
)
from e3_multichannel_amplitude import (  # noqa: E402
    PAI_W_G,
    PAI_W_Q,
    aggregate_components,
    build_alerts,
    point_metrics,
)
from e4_joint_instance_pai import bank_scores  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
)


LOCKED_10_SEEDS = list(range(20261116, 20261126))
K_GRID = (5, 10, 20)
PAI_OFF_W_EUC = 0.6
PAI_OFF_W_G = 0.4
PAI_OFF_W_Q = 0.2
OUT_BASE = REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s5b_symmetry_audit"
DEFAULT_R52_SUMMARY = Path(r"E:\MarketRisker\work\cert\external\r52_10seed_matrix_v1\summary.json")


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


def effective_rank(mat: np.ndarray) -> float:
    x = mat.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12]
    if len(eigvals) == 0:
        return 0.0
    p = eigvals / eigvals.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def load_r52_matched_steps(seeds: list[int], summary_json: Path, fallback: int = 2642) -> dict[int, int]:
    """Read exact per-seed A1-Cov optimizer steps from r52_10seed_matrix_v1/summary.json."""
    candidates = [
        summary_json,
        REPO_ROOT / "research" / "ccf-b" / "r5.2_10seed_training_summary.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        meta = json.loads(p.read_text(encoding="utf-8"))
        # Support multiple schema shapes cleanly
        out: dict[int, int] = {}
        if isinstance(meta, dict):
            # Case 1: top-level seed keys
            for s in seeds:
                if str(s) in meta and isinstance(meta[str(s)], dict):
                    for k in ("batches", "n_steps", "a1_cov_batches", "steps"):
                        if k in meta[str(s)]:
                            out[s] = int(meta[str(s)][k])
                            break
            # Case 2: nested under "seeds" or "runs" or "per_seed"
            if len(out) < len(seeds):
                for container_key in ("per_seed", "seeds", "runs", "coverage_batches"):
                    sub = meta.get(container_key)
                    if isinstance(sub, dict):
                        for s in seeds:
                            v = sub.get(str(s))
                            if isinstance(v, (int, float)):
                                out[s] = int(v)
                            elif isinstance(v, dict):
                                for k in ("batches", "n_steps", "a1_cov_batches", "steps"):
                                    if k in v:
                                        out[s] = int(v[k])
                                        break
                    elif isinstance(sub, list):
                        for item in sub:
                            if isinstance(item, dict) and "seed" in item:
                                s = int(item["seed"])
                                if s in seeds:
                                    for k in ("batches", "n_steps", "a1_cov_batches", "steps"):
                                        if k in item:
                                            out[s] = int(item[k])
                                            break
        if len(out) == len(seeds):
            print(f"[INFO] Loaded exact r5.2 per-seed matched steps from {p}: {out}", flush=True)
            return out

    # Fallback: read from already-completed exp_s1_r5.2 SoftInst-Rand-Matched run_meta.json!
    s1_matched_dir = REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s1_r5.2" / "SoftInst-Rand-Matched"
    out = {}
    for s in seeds:
        mfile = s1_matched_dir / f"seed_{s}" / "run_meta.json"
        if mfile.exists():
            m = json.loads(mfile.read_text(encoding="utf-8"))
            if "n_steps" in m:
                out[s] = int(m["n_steps"])
    if len(out) == len(seeds):
        print(f"[INFO] Loaded exact r5.2 per-seed matched steps from exp_s1_r5.2/SoftInst-Rand-Matched: {out}", flush=True)
        return out

    print(f"[WARN] Using fallback r5.2 matched steps={fallback}", flush=True)
    return {s: fallback for s in seeds}


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
                rows.append({"seed": seed, "method": method_name, "split": split, "k": k, **compact_metric(res, split), **pm})
                if split == "test" and k == 10:
                    hits = incident_hit_ids(data["users"], data["dates"], alerts, data["incidents"], start, end)
                    hit_map[f"{method_name}__test_k10"] = sorted(str(x) for x in hits)
    else:
        test_start = train_end + 1
        end = len(data["dates"]) - 1
        pm = point_metrics(signal, data, active, test_start, end)
        for k in K_GRID:
            alerts = build_alerts(rank, active, k)
            res = evaluate_alert_system(data, alerts, method_name, k, external=True)
            rows.append({"seed": seed, "method": method_name, "split": "test", "k": k, **compact_metric(res, None), **pm})
            if k == 10:
                hits = res["external_test"]["horizons"]["3"]["incident_ids"]
                hit_map[f"{method_name}__test_k10"] = sorted(str(x) for x in hits)
    return rows, hit_map


def run_hard_arms(
    dataset: str,
    arms: list[tuple[str, dict[int, int]]],
    seeds: list[int],
    out_dir: Path,
    batch_size: int = 16,
    force: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "r42" if dataset == "r4.2" else "r52"
    processed = CERT_DIR / f"processed_{tag}"
    ds = np.load(processed / f"{dataset}_sequence_dataset.npz", allow_pickle=False)
    values = ds["values"].astype(np.float32)
    users = ds["users"].astype(str)
    dates = ds["dates"].astype(str)
    active = ds["active_mask"][:, :, 0].astype(bool)
    train_end = int(ds["train_end_index"])
    val_end = int(ds["val_end_index"]) if "val_end_index" in ds else None
    train_values = values[:, : train_end + 1]
    user_count = values.shape[0]

    data = load_r42() if dataset == "r4.2" else load_r52()
    raw_20 = align_raw_features(processed / f"{dataset}_user_day_features.csv", users, dates)
    G, Q = aggregate_components(raw_20, active, train_end, user_specific=True, aggregation="channel_mean")
    G_z = global_train_z(G, active, train_end)
    Q_z = global_train_z(Q, active, train_end)

    device_id = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{device_id}" if isinstance(device_id, int) else "cpu")

    all_rows: list[dict] = []
    meta_rows: list[dict] = []

    for arm_name, step_map in arms:
        for seed in seeds:
            run_dir = out_dir / arm_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_csv = run_dir / "metrics.csv"
            meta_json = run_dir / "run_meta.json"

            if metrics_csv.exists() and meta_json.exists() and not force:
                all_rows.extend(pd.read_csv(metrics_csv).to_dict(orient="records"))
                meta_rows.append(json.loads(meta_json.read_text(encoding="utf-8")))
                print(f"[SKIP] {dataset} {arm_name} seed={seed}", flush=True)
                continue

            t0 = time.time()
            n_steps = step_map[seed]
            batch_table = build_random_batch_table(user_count, batch_size, n_steps, seed)

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
                batch_size=batch_size,
                lambda_=0.5,
                tau_temp=1.5,
                soft_instance=False,
                soft_temporal=False,
            )
            logs = model.fit_coverage_table(
                train_data=train_values,
                soft_labels=None,
                batch_table=batch_table,
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
            pai_off = PAI_OFF_W_EUC * euc_z + PAI_OFF_W_G * G_z + PAI_OFF_W_Q * Q_z

            repr_norms = np.linalg.norm(plain_repr, axis=-1).astype(np.float32)
            eff_r = effective_rank(plain_repr[active])

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
                r_list, h_map = evaluate_all_splits(dataset, sc_name, sc_arr, data, active, train_end, val_end, seed)
                seed_rows.extend(r_list)
                seed_hits.update(h_map)

            pd.DataFrame(seed_rows).to_csv(metrics_csv, index=False, encoding="utf-8")
            (run_dir / "hit_ids_k10.json").write_text(json.dumps(seed_hits, ensure_ascii=False, indent=2), encoding="utf-8")
            meta_rec = {
                "dataset": dataset,
                "arm": arm_name,
                "seed": seed,
                "n_steps": n_steps,
                "final_loss": float(logs[-1]["loss"]) if logs else None,
                "mean_repr_norm_active": float(repr_norms[active].mean()),
                "std_repr_norm_active": float(repr_norms[active].std()),
                "effective_rank_active": eff_r,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            meta_json.write_text(json.dumps(meta_rec, ensure_ascii=False, indent=2), encoding="utf-8")
            all_rows.extend(seed_rows)
            meta_rows.append(meta_rec)
            print(
                f"[DONE] {dataset} {arm_name} seed={seed} steps={n_steps} "
                f"norm={meta_rec['mean_repr_norm_active']:.3f} effRank={eff_r:.2f} ({meta_rec['elapsed_sec']:.1f}s)",
                flush=True,
            )

    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(out_dir / "metrics_by_seed.csv", index=False, encoding="utf-8")
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
    summary.to_csv(out_dir / "method_summary.csv", index=False, encoding="utf-8")

    mdf = pd.DataFrame(meta_rows)
    mdf.to_csv(out_dir / "norms_by_seed.csv", index=False, encoding="utf-8")
    msum = mdf.groupby("arm", as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean_steps=("n_steps", "mean"),
        norm_mean=("mean_repr_norm_active", "mean"),
        norm_std=("mean_repr_norm_active", "std"),
        eff_rank_mean=("effective_rank_active", "mean"),
        eff_rank_std=("effective_rank_active", "std"),
    )
    msum.to_csv(out_dir / "norms_summary.csv", index=False, encoding="utf-8")

    print(f"\n=== {dataset} NORMS SUMMARY ===")
    print(msum.to_string(index=False))
    print(f"\n=== {dataset} TEST K=10 SUMMARY ===")
    print(summary[(summary["split"] == "test") & (summary["k"] == 10)].to_string(index=False))


def cmd_audit_iforest_t0() -> None:
    out_dir = OUT_BASE / "iforest_t0_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"repo_matches": [], "recomputed_iforest": {}}

    # 1) Search existing CSVs/JSONs under results/ and research/ for 0.00744 / iforest / T0
    for search_root in (REPO_ROOT / "results", REPO_ROOT / "research"):
        if not search_root.exists():
            continue
        for f in search_root.rglob("*.csv"):
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if "0.00744" in txt or "iforest" in txt.lower() or "isolation" in txt.lower():
                hits = [
                    line.strip()
                    for line in txt.splitlines()
                    if any(k in line.lower() for k in ("0.00744", "iforest", "isolation", "t0", "mask"))
                ]
                report["repo_matches"].append(
                    {"file": str(f.relative_to(REPO_ROOT)), "lines": hits[:25]}
                )

    # 2) Clean reference IsolationForest (500 trees, seeds 42 and 2024) on r4.2 27D active days
    processed = CERT_DIR / "processed_r42"
    ds = np.load(processed / "r4.2_sequence_dataset.npz", allow_pickle=False)
    values = np.nan_to_num(ds["values"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    active = ds["active_mask"][:, :, 0].astype(bool)
    train_end = int(ds["train_end_index"])
    val_end = int(ds["val_end_index"])
    data = load_r42()

    x_tr = values[:, : train_end + 1][active[:, : train_end + 1]]
    x_all = values.reshape(-1, values.shape[-1])
    for rs in (42, 0, 2024):
        clf = IsolationForest(n_estimators=500, random_state=rs, n_jobs=-1)
        clf.fit(x_tr)
        scores = (-clf.score_samples(x_all)).reshape(values.shape[0], values.shape[1]).astype(np.float32)
        pm_test = point_metrics(scores, data, active, val_end + 1, len(data["dates"]) - 1)
        rank = scores.copy()
        rank[~active] = -np.inf
        alerts = build_alerts(rank, active, 10)
        res = evaluate_alert_system(data, alerts, f"iforest_rs{rs}", 10, external=False)
        report["recomputed_iforest"][f"random_state_{rs}"] = {
            **pm_test,
            **compact_metric(res, "test"),
        }

    out_path = out_dir / "iforest_t0_exact_audit.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Exp S5b: Symmetry & Norm Consistency Closure")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("hard-r42-10ep")
    p1.add_argument("--seeds", nargs="+", type=int, default=LOCKED_10_SEEDS)
    p1.add_argument("--force", action="store_true")

    p2 = sub.add_parser("hard-r52-full")
    p2.add_argument("--seeds", nargs="+", type=int, default=LOCKED_10_SEEDS)
    p2.add_argument("--r52-summary-json", type=Path, default=DEFAULT_R52_SUMMARY)
    p2.add_argument("--force", action="store_true")

    sub.add_parser("audit-iforest-t0")

    args = p.parse_args()
    if args.cmd == "hard-r42-10ep":
        steps = {s: (1000 // 16) * 10 for s in args.seeds}  # 620 steps
        run_hard_arms("r4.2", [("Hard-Rand-10ep", steps)], args.seeds, OUT_BASE / "r42_hard_10ep", force=args.force)
    elif args.cmd == "hard-r52-full":
        steps_10ep = {s: (2000 // 16) * 10 for s in args.seeds}  # 1250 steps
        steps_matched = load_r52_matched_steps(args.seeds, args.r52_summary_json)
        run_hard_arms(
            "r5.2",
            [("Hard-Rand-10ep", steps_10ep), ("Hard-Rand-Matched", steps_matched)],
            args.seeds,
            OUT_BASE / "r52_hard_full",
            force=args.force,
        )
    elif args.cmd == "audit-iforest-t0":
        cmd_audit_iforest_t0()


if __name__ == "__main__":
    main()
