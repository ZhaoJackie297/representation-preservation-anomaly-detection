"""Exp S3 (P0, CPU-only, ~5 min): Zero-Retrain Reviewer Diagnostics Bundle.

Place this script at: <repo_root>/research/ccf-b/supplement/exp_s3_cpu_diagnostics_bootstrap_norm.py
Subcommands (run from <repo_root>):
  1) CERT r4.2 200-seed random ranker baseline (K=5,10,20; 7d cooldown; 3d tolerance):
     python research/ccf-b/supplement/exp_s3_cpu_diagnostics_bootstrap_norm.py random-r42

  2) Encoder-free standalone raw-amplitude baselines (magG-only, T2-only, magG+T2) on r4.2 & r5.2:
     python research/ccf-b/supplement/exp_s3_cpu_diagnostics_bootstrap_norm.py amp-standalone --dataset r4.2
     python research/ccf-b/supplement/exp_s3_cpu_diagnostics_bootstrap_norm.py amp-standalone --dataset r5.2

  3) Paired Event-Level Bootstrap (24 / 99 events) & User-Cluster Bootstrap (1000 / 2000 users):
     python research/ccf-b/supplement/exp_s3_cpu_diagnostics_bootstrap_norm.py bootstrap \
       --dataset r4.2 --dir-a <dir_with_seed_scores_A> --key-a pai_a1 \
       --dir-b <dir_with_seed_scores_B> --key-b plain_l1

  4) Representation Norm vs. Input Amplitude Correlation + Eq. (7) Radial-Angular Decomposition:
     python research/ccf-b/supplement/exp_s3_cpu_diagnostics_bootstrap_norm.py norm-corr \
       --dataset r4.2 --checkpoints T0=<path/to/t0_model.pth> T4=<path/to/t4_model.pth>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_DIR = REPO_ROOT / "runtime" / "cert"
sys.path.insert(0, str(CERT_DIR))

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
from e4_joint_instance_pai import load_soft_model  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    compact_metric,
    evaluate_alert_system,
    load_r42,
    load_r52,
)


OUT_ROOT = REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s3_diagnostics"


def cmd_random_r42(args: argparse.Namespace) -> None:
    out_dir = OUT_ROOT / "random_r42"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_r42()
    active = data["active_mask"]
    rng = np.random.default_rng(args.base_seed)

    rows = []
    for s in range(args.n_seeds):
        scores = rng.random(active.shape, dtype=np.float32)
        scores[~active] = -np.inf
        for k in (5, 10, 20):
            alerts = build_alerts(scores, active, k)
            res = evaluate_alert_system(data, alerts, "random", k, external=False)
            for split in ("validation", "test"):
                m = compact_metric(res, split)
                rows.append({"seed": s, "split": split, "k": k, **m})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "r4.2_random_baseline_seeds.csv", index=False, encoding="utf-8")
    summary = (
        df.groupby(["split", "k"], as_index=False)
        .agg(
            event_recall_3d_mean=("event_recall_3d", "mean"),
            event_recall_3d_std=("event_recall_3d", "std"),
            event_recall_3d_q025=("event_recall_3d", lambda x: float(np.quantile(x, 0.025))),
            event_recall_3d_q975=("event_recall_3d", lambda x: float(np.quantile(x, 0.975))),
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_std=("incident_hits", "std"),
        )
    )
    summary.to_csv(out_dir / "r4.2_random_baseline_summary.csv", index=False, encoding="utf-8")
    print("=== CERT r4.2 RANDOM BASELINE (200 seeds) ===")
    print(summary.to_string(index=False))


def cmd_amp_standalone(args: argparse.Namespace) -> None:
    out_dir = OUT_ROOT / f"amp_standalone_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "r42" if args.dataset == "r4.2" else "r52"
    processed = CERT_DIR / f"processed_{tag}"
    ds = np.load(processed / f"{args.dataset}_sequence_dataset.npz", allow_pickle=False)
    users = ds["users"].astype(str)
    dates = ds["dates"].astype(str)
    active = ds["active_mask"][:, :, 0].astype(bool)
    train_end = int(ds["train_end_index"])
    val_end = int(ds["val_end_index"]) if "val_end_index" in ds else None
    data = load_r42() if args.dataset == "r4.2" else load_r52()
    raw = align_raw_features(processed / f"{args.dataset}_user_day_features.csv", users, dates)

    rows = []
    for cfg_name, user_spec, agg in [
        ("A1_user_channel_mean", True, "channel_mean"),
        ("A0_global_channel_mean", False, "channel_mean"),
        ("A3_user_channel_top2", True, "channel_top2"),
    ]:
        G, Q = aggregate_components(
            raw, active, train_end, user_specific=user_spec, aggregation=agg
        )
        G_z = global_train_z(G, active, train_end)
        Q_z = global_train_z(Q, active, train_end)
        comb = PAI_W_G * G_z + PAI_W_Q * Q_z

        for comp_name, sig in [
            (f"{cfg_name}__magG_only", G_z),
            (f"{cfg_name}__T2_only", Q_z),
            (f"{cfg_name}__magG_plus_T2", comb),
        ]:
            rank = sig.astype(np.float32, copy=True)
            rank[~np.isfinite(rank)] = -np.inf
            if args.dataset == "r4.2":
                assert val_end is not None
                for split, (st, ed) in [
                    ("validation", (train_end + 1, val_end)),
                    ("test", (val_end + 1, len(dates) - 1)),
                ]:
                    pm = point_metrics(sig, data, active, st, ed)
                    for k in (5, 10, 20):
                        alerts = build_alerts(rank, active, k)
                        res = evaluate_alert_system(data, alerts, comp_name, k, external=False)
                        rows.append({"method": comp_name, "split": split, "k": k, **compact_metric(res, split), **pm})
            else:
                pm = point_metrics(sig, data, active, train_end + 1, len(dates) - 1)
                for k in (5, 10, 20):
                    alerts = build_alerts(rank, active, k)
                    res = evaluate_alert_system(data, alerts, comp_name, k, external=True)
                    rows.append({"method": comp_name, "split": "test", "k": k, **compact_metric(res, None), **pm})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "amp_standalone_summary.csv", index=False, encoding="utf-8")
    print(f"=== STANDALONE AMPLITUDE BASELINES ({args.dataset}, test K=10) ===")
    print(df[(df["split"] == "test") & (df["k"] == 10)].to_string(index=False))


def _load_seed_scores(
    score_dir: Path, key: str, file_name: str | None = None
) -> dict[str, np.ndarray]:
    """Load score arrays keyed by seed folder name or filename from score_dir.

    The local score archive stores all three methods in the same ``seed_*``
    directory.  Matching by key alone would therefore overwrite one method
    with another, so callers should pass the exact method filename.
    """
    if file_name:
        candidates = sorted(score_dir.glob(f"seed_*/{file_name}"))
        if not candidates and (score_dir / file_name).is_file():
            candidates = [score_dir / file_name]
    else:
        candidates = sorted(score_dir.rglob("*.npz"))

    found: dict[str, np.ndarray] = {}
    for p in candidates:
        with np.load(p, allow_pickle=False) as z:
            if key in z:
                if file_name:
                    tag = p.parent.name
                else:
                    tag = p.parent.name if p.parent != score_dir else p.stem
                if tag in found:
                    raise RuntimeError(
                        f"Duplicate seed tag '{tag}' for key '{key}'. "
                        "Pass --file-a/--file-b to select a method explicitly."
                    )
                found[tag] = z[key].astype(np.float32)
    if not found:
        suffix = f" matching '{file_name}'" if file_name else ""
        raise FileNotFoundError(
            f"No .npz{suffix} with key '{key}' found under {score_dir}"
        )
    return found


def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Paired Event-Level Bootstrap (over 24 / 99 events) + User-Cluster Bootstrap (PR/ROC)."""
    out_dir = OUT_ROOT / f"bootstrap_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_r42() if args.dataset == "r4.2" else load_r52()
    active = data["active_mask"]
    labels = data["labels"]
    if args.dataset == "r4.2":
        test_start = data["val_end"] + 1
    else:
        test_start = data["test_start"]
    test_end = len(data["dates"]) - 1

    scores_a = _load_seed_scores(args.dir_a, args.key_a, args.file_a)
    scores_b = _load_seed_scores(args.dir_b, args.key_b, args.file_b)
    common_seeds = sorted(set(scores_a) & set(scores_b))
    if not common_seeds:
        raise RuntimeError(
            f"No overlapping seed tags between {args.dir_a} ({list(scores_a)[:3]}) "
            f"and {args.dir_b} ({list(scores_b)[:3]})."
        )

    # Identify canonical test incident list
    inc = data["incidents"]
    if args.dataset == "r4.2":
        test_start_date = data["dates"][test_start]
        test_end_date = data["dates"][test_end]
        test_inc = inc[(inc["start_date"] >= test_start_date) & (inc["start_date"] <= test_end_date)]
        inc_ids = sorted(test_inc["incident_id"].astype(str).tolist())
    else:
        inc_ids = sorted(inc["incident_id"].astype(str).tolist())
    n_events = len(inc_ids)

    # Per-event hit probability averaged over matched seeds (independent unit = incident)
    hits_a_mat = np.zeros((len(common_seeds), n_events), dtype=np.float64)
    hits_b_mat = np.zeros((len(common_seeds), n_events), dtype=np.float64)
    for si, seed_tag in enumerate(common_seeds):
        for mat, sc in [(hits_a_mat, scores_a[seed_tag]), (hits_b_mat, scores_b[seed_tag])]:
            rk = sc.astype(np.float32, copy=True)
            rk[~np.isfinite(rk)] = -np.inf
            alerts = build_alerts(rk, active, args.k)
            if args.dataset == "r4.2":
                hit_set = {
                    str(x)
                    for x in incident_hit_ids(
                        data["users"], data["dates"], alerts, inc, test_start, test_end
                    )
                }
            else:
                res = evaluate_alert_system(data, alerts, "boot", args.k, external=True)
                hit_set = {str(x) for x in res["external_test"]["horizons"]["3"]["incident_ids"]}
            for ei, eid in enumerate(inc_ids):
                mat[si, ei] = 1.0 if eid in hit_set else 0.0

    ev_a = hits_a_mat.mean(axis=0)  # shape (n_events,)
    ev_b = hits_b_mat.mean(axis=0)
    ev_diff = ev_a - ev_b

    rng = np.random.default_rng(args.boot_seed)
    boot_ev = np.empty(args.n_boot, dtype=np.float64)
    for b in range(args.n_boot):
        idx = rng.integers(0, n_events, size=n_events)
        boot_ev[b] = ev_diff[idx].mean()

    nonzero_ev = ev_diff[ev_diff != 0]
    ev_wilcoxon_p = float(wilcoxon(nonzero_ev).pvalue) if len(nonzero_ev) >= 2 else 1.0

    # User-cluster bootstrap for PR-AUC & ROC-AUC on seed-averaged test scores
    mean_a = np.nanmean(np.stack([scores_a[s] for s in common_seeds], axis=0), axis=0)
    mean_b = np.nanmean(np.stack([scores_b[s] for s in common_seeds], axis=0), axis=0)
    act_t = active[:, test_start : test_end + 1]
    lab_t = labels[:, test_start : test_end + 1]
    sa_t = mean_a[:, test_start : test_end + 1]
    sb_t = mean_b[:, test_start : test_end + 1]

    # Pre-extract per-user active vectors for fast user-cluster resampling
    n_users = act_t.shape[0]
    u_lab = [lab_t[u][act_t[u]] for u in range(n_users)]
    u_sa = [sa_t[u][act_t[u]] for u in range(n_users)]
    u_sb = [sb_t[u][act_t[u]] for u in range(n_users)]

    obs_y = np.concatenate(u_lab)
    obs_a = np.concatenate(u_sa)
    obs_b = np.concatenate(u_sb)
    obs_dpr = float(average_precision_score(obs_y, obs_a) - average_precision_score(obs_y, obs_b))
    obs_droc = float(roc_auc_score(obs_y, obs_a) - roc_auc_score(obs_y, obs_b))

    boot_dpr = []
    boot_droc = []
    for _ in range(args.n_user_boot):
        uidx = rng.integers(0, n_users, size=n_users)
        yb = np.concatenate([u_lab[u] for u in uidx])
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ab = np.concatenate([u_sa[u] for u in uidx])
        bb = np.concatenate([u_sb[u] for u in uidx])
        boot_dpr.append(float(average_precision_score(yb, ab) - average_precision_score(yb, bb)))
        boot_droc.append(float(roc_auc_score(yb, ab) - roc_auc_score(yb, bb)))

    report = {
        "dataset": args.dataset,
        "n_common_seeds": len(common_seeds),
        "n_events": n_events,
        "k": args.k,
        "event_level": {
            "mean_recall_A": float(ev_a.mean()),
            "mean_recall_B": float(ev_b.mean()),
            "delta_recall_A_minus_B": float(ev_diff.mean()),
            "boot_95ci": [float(np.quantile(boot_ev, 0.025)), float(np.quantile(boot_ev, 0.975))],
            "events_improved": int((ev_diff > 0).sum()),
            "events_worsened": int((ev_diff < 0).sum()),
            "events_tied": int((ev_diff == 0).sum()),
            "wilcoxon_p_across_events": ev_wilcoxon_p,
        },
        "user_cluster_bootstrap": {
            "n_valid_resamples": len(boot_dpr),
            "obs_delta_pr_auc": obs_dpr,
            "delta_pr_auc_95ci": [float(np.quantile(boot_dpr, 0.025)), float(np.quantile(boot_dpr, 0.975))],
            "obs_delta_roc_auc": obs_droc,
            "delta_roc_auc_95ci": [float(np.quantile(boot_droc, 0.025)), float(np.quantile(boot_droc, 0.975))],
        },
    }
    out_file = out_dir / f"{args.tag}_bootstrap.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_norm_corr(args: argparse.Namespace) -> None:
    """Audit ||r(u,t)||_2 vs. raw input amplitude & Eq. (7) radial-vs-angular decomposition."""
    out_dir = OUT_ROOT / f"norm_corr_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "r42" if args.dataset == "r4.2" else "r52"
    processed = CERT_DIR / f"processed_{tag}"
    ds = np.load(processed / f"{args.dataset}_sequence_dataset.npz", allow_pickle=False)
    values = ds["values"].astype(np.float32)
    users = ds["users"].astype(str)
    dates = ds["dates"].astype(str)
    active = ds["active_mask"][:, :, 0].astype(bool)
    train_end = int(ds["train_end_index"])
    val_end = int(ds["val_end_index"]) if "val_end_index" in ds else train_end
    test_start = (val_end + 1) if args.dataset == "r4.2" else (train_end + 1)
    data = load_r42() if args.dataset == "r4.2" else load_r52()
    labels = data["labels"]

    raw_20 = align_raw_features(processed / f"{args.dataset}_user_day_features.csv", users, dates)
    G, Q = aggregate_components(raw_20, active, train_end, user_specific=True, aggregation="channel_mean")
    x_norm = np.linalg.norm(values[:, :, :20], axis=-1)

    test_mask = np.zeros_like(active, dtype=bool)
    test_mask[:, test_start:] = active[:, test_start:]
    y_test = labels[test_mask]
    g_test = G[test_mask]
    q_test = Q[test_mask]
    xn_test = x_norm[test_mask]

    rows = []
    for spec in args.checkpoints:
        label_name, ckpt_str = spec.split("=", 1)
        model, device = load_soft_model(Path(ckpt_str))
        reprs = np.asarray(
            model.encode(
                values,
                causal=True,
                sliding_length=1,
                sliding_padding=30,
                batch_size=256,
            ),
            dtype=np.float32,
        )
        # Fit identical MiniBatchKMeans normal bank on training-period active representations
        tr_repr = reprs[:, : train_end + 1][active[:, : train_end + 1]]
        k_clusters = min(500, max(1, int(round(0.1 * len(tr_repr)))))
        km = MiniBatchKMeans(
            n_clusters=k_clusters, init="k-means++", random_state=2024,
            batch_size=max(8192, k_clusters), max_iter=50, n_init=1, reassignment_ratio=0.01,
        ).fit(tr_repr)
        bank = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device)
        bank_r = torch.linalg.norm(bank, dim=1)  # (K,)
        bank_u = torch.nn.functional.normalize(bank, dim=1)

        q_arr = reprs[test_mask]  # (N_test, 64)
        q_norms = np.linalg.norm(q_arr, axis=-1)

        radial_sq_list, angular_term_list, euc_sq_list = [], [], []
        radial_only_list, cos_only_list, euc_only_list = [], [], []
        with torch.inference_mode():
            for st in range(0, len(q_arr), 4096):
                q = torch.from_numpy(q_arr[st : st + 4096]).to(device)
                rq = torch.linalg.norm(q, dim=1)  # (B,)
                qu = torch.nn.functional.normalize(q, dim=1)
                d_cos = (1.0 - qu @ bank_u.T).clamp_min_(0.0)
                euc = torch.cdist(q, bank)
                # Top-3 nearest centers under Euclidean distance -> exact Eq. (7) decomposition
                top3_euc, top3_idx = euc.topk(3, dim=1, largest=False)
                rc = bank_r[top3_idx]  # (B, 3)
                dcos_sel = torch.gather(d_cos, 1, top3_idx)
                rad_sq = (rq.unsqueeze(1) - rc) ** 2
                ang_tm = 2.0 * rq.unsqueeze(1) * rc * dcos_sel
                tot_sq = rad_sq + ang_tm

                radial_sq_list.append(rad_sq.mean(dim=1).cpu().numpy())
                angular_term_list.append(ang_tm.mean(dim=1).cpu().numpy())
                euc_sq_list.append(tot_sq.mean(dim=1).cpu().numpy())
                radial_only_list.append(torch.abs(rq.unsqueeze(1) - rc).mean(dim=1).cpu().numpy())
                euc_only_list.append(top3_euc.mean(dim=1).cpu().numpy())
                cos_only_list.append(d_cos.topk(3, dim=1, largest=False).values.mean(dim=1).cpu().numpy())

        rad_sq_v = np.concatenate(radial_sq_list)
        ang_tm_v = np.concatenate(angular_term_list)
        tot_sq_v = np.maximum(np.concatenate(euc_sq_list), 1e-12)
        rad_share = rad_sq_v / tot_sq_v
        rad_only = np.concatenate(radial_only_list)
        cos_only = np.concatenate(cos_only_list)
        euc_only = np.concatenate(euc_only_list)

        norm_m = y_test == 0
        anom_m = y_test == 1
        fin_g = np.isfinite(q_norms) & np.isfinite(g_test)
        fin_q = np.isfinite(q_norms) & np.isfinite(q_test)
        fin_x = np.isfinite(q_norms) & np.isfinite(xn_test)
        rows.append(
            {
                "model": label_name,
                "norm_mean_normal": float(q_norms[norm_m].mean()),
                "norm_std_normal": float(q_norms[norm_m].std()),
                "norm_cv_normal": float(q_norms[norm_m].std() / max(q_norms[norm_m].mean(), 1e-8)),
                "norm_mean_anom": float(q_norms[anom_m].mean()),
                "spearman_norm_vs_magG": float(spearmanr(q_norms[fin_g], g_test[fin_g])[0]),
                "pearson_norm_vs_magG": float(pearsonr(q_norms[fin_g], g_test[fin_g])[0]),
                "spearman_norm_vs_T2": float(spearmanr(q_norms[fin_q], q_test[fin_q])[0]),
                "spearman_norm_vs_xnorm": float(spearmanr(q_norms[fin_x], xn_test[fin_x])[0]),
                "radial_energy_share_normal": float(rad_share[norm_m].mean()),
                "radial_energy_share_anom": float(rad_share[anom_m].mean()),
                "pr_auc_radial_only": float(average_precision_score(y_test, rad_only)),
                "roc_auc_radial_only": float(roc_auc_score(y_test, rad_only)),
                "pr_auc_cosine_bank": float(average_precision_score(y_test, cos_only)),
                "roc_auc_cosine_bank": float(roc_auc_score(y_test, cos_only)),
                "pr_auc_euclidean_bank": float(average_precision_score(y_test, euc_only)),
                "roc_auc_euclidean_bank": float(roc_auc_score(y_test, euc_only)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "norm_amplitude_radial_audit.csv", index=False, encoding="utf-8")
    print(df.T.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp S3: Zero-retrain CPU diagnostics bundle")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("random-r42")
    p1.add_argument("--n-seeds", type=int, default=200)
    p1.add_argument("--base-seed", type=int, default=20261110)

    p2 = sub.add_parser("amp-standalone")
    p2.add_argument("--dataset", choices=["r4.2", "r5.2"], required=True)

    p3 = sub.add_parser("bootstrap")
    p3.add_argument("--dataset", choices=["r4.2", "r5.2"], required=True)
    p3.add_argument("--dir-a", type=Path, required=True)
    p3.add_argument(
        "--file-a",
        type=str,
        help="Exact .npz filename below each seed_* directory, e.g. A1-Cov_scores.npz",
    )
    p3.add_argument("--key-a", type=str, required=True)
    p3.add_argument("--dir-b", type=Path, required=True)
    p3.add_argument(
        "--file-b",
        type=str,
        help="Exact .npz filename below each seed_* directory, e.g. A0-official_scores.npz",
    )
    p3.add_argument("--key-b", type=str, required=True)
    p3.add_argument("--tag", type=str, default="A_vs_B")
    p3.add_argument("--k", type=int, default=10)
    p3.add_argument("--n-boot", type=int, default=2000)
    p3.add_argument("--n-user-boot", type=int, default=1000)
    p3.add_argument("--boot-seed", type=int, default=20261121)

    p4 = sub.add_parser("norm-corr")
    p4.add_argument("--dataset", choices=["r4.2", "r5.2"], required=True)
    p4.add_argument("--checkpoints", nargs="+", required=True, help="Pairs like T0=path/model.pth T4=path/model.pth")

    args = parser.parse_args()
    if args.cmd == "random-r42":
        cmd_random_r42(args)
    elif args.cmd == "amp-standalone":
        cmd_amp_standalone(args)
    elif args.cmd == "bootstrap":
        cmd_bootstrap(args)
    elif args.cmd == "norm-corr":
        cmd_norm_corr(args)


if __name__ == "__main__":
    main()
