"""Exp S5c (P1 Recommended Consistency Upgrade):
  1) cert-3seed : 3-seed (20261116-20261118) Eq. (7) Radial-Angular & Norm-Amplitude Audit on
                  CERT r4.2 using Hard-Rand (T0), SoftInst-Rand (T3), and optional T4 checkpoints.
                  -> Upgrades Table 7 CERT from 1-seed to 3-seed mean +/- std!
  2) smd-3seed  : 3-seed (20261201-20261203) Eq. (7) Radial-Angular & Norm-Amplitude Audit on
                  SMD (28 servers) for T0_hard, T3_inst_soft (= SoftInst-Rand!), and T4_soft.
                  -> Upgrades Table 7 SMD from 1-seed to 3-seed mean +/- std AND aligns main
                     variant T3_inst_soft directly onto SMD!

Place at: <repo_root>/research/ccf-b/supplement/exp_s5c_norm_corr_3seed.py
Run from <repo_root>:
  python research/ccf-b/supplement/exp_s5c_norm_corr_3seed.py cert-3seed
  python research/ccf-b/supplement/exp_s5c_norm_corr_3seed.py smd-3seed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_DIR = REPO_ROOT / "runtime" / "cert"
S2_DIR = REPO_ROOT / "code" / "by_layer" / "S2_dual_end"
SOFTCLT_DIR = REPO_ROOT / "references" / "softclt_repo" / "softclt_ts2vec"
OUT_BASE = REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s5c_norm_corr_3seed"


def audit_cert_model(
    ckpt_path: Path,
    values: np.ndarray,
    active: np.ndarray,
    labels: np.ndarray,
    G: np.ndarray,
    Q: np.ndarray,
    x_norm: np.ndarray,
    train_end: int,
    test_mask: np.ndarray,
) -> dict[str, float]:
    sys.path.insert(0, str(CERT_DIR))
    from e4_joint_instance_pai import load_soft_model

    model, device = load_soft_model(ckpt_path)
    # Try correct spelling first, fall back to upstream spelling if needed
    try:
        reprs = np.asarray(
            model.encode(values, causal=True, sliding_length=1, sliding_padding=30, batch_size=256),
            dtype=np.float32,
        )
    except TypeError:
        reprs = np.asarray(
            model.encode(values, casual=True, sliding_length=1, sliding_padding=30, batch_size=256),
            dtype=np.float32,
        )

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
    bank_u = torch.nn.functional.normalize(bank, dim=1)

    q_arr = reprs[test_mask]
    y_test = labels[test_mask]
    g_test = G[test_mask]
    q_test = Q[test_mask]
    xn_test = x_norm[test_mask]
    q_norms = np.linalg.norm(q_arr, axis=-1)

    rad_sq_l, tot_sq_l, rad_only_l, cos_l, euc_l = [], [], [], [], []
    with torch.inference_mode():
        for st in range(0, len(q_arr), 4096):
            q = torch.from_numpy(q_arr[st : st + 4096]).to(device)
            rq = torch.linalg.norm(q, dim=1)
            qu = torch.nn.functional.normalize(q, dim=1)
            d_cos = (1.0 - qu @ bank_u.T).clamp_min_(0.0)
            euc = torch.cdist(q, bank)
            top3_euc, top3_idx = euc.topk(3, dim=1, largest=False)
            rc = bank_r[top3_idx]
            dcos_sel = torch.gather(d_cos, 1, top3_idx)
            rad_sq = (rq.unsqueeze(1) - rc) ** 2
            ang_tm = 2.0 * rq.unsqueeze(1) * rc * dcos_sel
            rad_sq_l.append(rad_sq.mean(dim=1).cpu().numpy())
            tot_sq_l.append((rad_sq + ang_tm).mean(dim=1).cpu().numpy())
            rad_only_l.append(torch.abs(rq.unsqueeze(1) - rc).mean(dim=1).cpu().numpy())
            euc_l.append(top3_euc.mean(dim=1).cpu().numpy())
            cos_l.append(d_cos.topk(3, dim=1, largest=False).values.mean(dim=1).cpu().numpy())

    rad_share = np.concatenate(rad_sq_l) / np.maximum(np.concatenate(tot_sq_l), 1e-12)
    rad_only = np.concatenate(rad_only_l)
    cos_only = np.concatenate(cos_l)
    euc_only = np.concatenate(euc_l)

    norm_m = y_test == 0
    anom_m = y_test == 1
    fin_g = np.isfinite(q_norms) & np.isfinite(g_test)
    fin_q = np.isfinite(q_norms) & np.isfinite(q_test)
    fin_x = np.isfinite(q_norms) & np.isfinite(xn_test)
    return {
        "norm_mean_normal": float(q_norms[norm_m].mean()),
        "norm_cv_normal": float(q_norms[norm_m].std() / max(q_norms[norm_m].mean(), 1e-8)),
        "norm_mean_anom": float(q_norms[anom_m].mean()),
        "spearman_norm_vs_xnorm": float(spearmanr(q_norms[fin_x], xn_test[fin_x])[0]),
        "spearman_norm_vs_magG": float(spearmanr(q_norms[fin_g], g_test[fin_g])[0]),
        "spearman_norm_vs_T2": float(spearmanr(q_norms[fin_q], q_test[fin_q])[0]),
        "radial_energy_share_normal": float(rad_share[norm_m].mean()),
        "radial_energy_share_anom": float(rad_share[anom_m].mean()),
        "pr_auc_radial_only": float(average_precision_score(y_test, rad_only)),
        "pr_auc_cosine_bank": float(average_precision_score(y_test, cos_only)),
        "roc_auc_cosine_bank": float(roc_auc_score(y_test, cos_only)),
        "pr_auc_euclidean_bank": float(average_precision_score(y_test, euc_only)),
        "roc_auc_euclidean_bank": float(roc_auc_score(y_test, euc_only)),
    }


def cmd_cert_3seed(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(CERT_DIR))
    from e0_e1_representation_preservation import align_raw_features
    from e3_multichannel_amplitude import aggregate_components
    from quota_relationship_sim import load_r42

    out_dir = OUT_BASE / "cert_r42_3seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = CERT_DIR / "processed_r42"
    ds = np.load(processed / "r4.2_sequence_dataset.npz", allow_pickle=False)
    values = ds["values"].astype(np.float32)
    users = ds["users"].astype(str)
    dates = ds["dates"].astype(str)
    active = ds["active_mask"][:, :, 0].astype(bool)
    train_end = int(ds["train_end_index"])
    val_end = int(ds["val_end_index"])
    data = load_r42()
    labels = data["labels"]

    raw_20 = align_raw_features(processed / "r4.2_user_day_features.csv", users, dates)
    G, Q = aggregate_components(raw_20, active, train_end, user_specific=True, aggregation="channel_mean")
    x_norm = np.linalg.norm(values[:, :, :20], axis=-1)
    test_mask = np.zeros_like(active, dtype=bool)
    test_mask[:, val_end + 1 :] = active[:, val_end + 1 :]

    s5a_dir = args.s5a_dir
    s1_dir = REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s1_r4.2"

    rows = []
    for seed in args.seeds:
        ckpt_candidates = {
            "Hard-Rand (T0)": [
                s5a_dir / "Hard-Rand" / f"seed_{seed}" / "model.pth",
                REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s5b_symmetry_audit" / "r42_hard_10ep" / "Hard-Rand-10ep" / f"seed_{seed}" / "model.pth",
            ],
            "SoftInst-Rand (T3)": [
                s5a_dir / "SoftInst-Rand" / f"seed_{seed}" / "model.pth",
                s1_dir / "SoftInst-Rand-10ep" / f"seed_{seed}" / "model.pth",
            ],
        }
        # Include T4 if present in local external root
        if args.t4_root is not None:
            ckpt_candidates["T4 (Full Soft)"] = [args.t4_root / f"seed_{seed}" / "model.pth"]

        for model_label, paths in ckpt_candidates.items():
            ckpt = next((p for p in paths if p.exists()), None)
            if ckpt is None:
                print(f"[WARN] Missing checkpoint for {model_label} seed={seed}", flush=True)
                continue
            m = audit_cert_model(ckpt, values, active, labels, G, Q, x_norm, train_end, test_mask)
            rows.append({"model": model_label, "seed": seed, **m})
            print(f"[DONE] CERT {model_label} seed={seed} rad_anom={m['radial_energy_share_anom']:.4f} cos_pr={m['pr_auc_cosine_bank']:.5f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cert_norm_radial_by_seed.csv", index=False, encoding="utf-8")
    summary = df.groupby("model", as_index=False).mean(numeric_only=True)
    summary.to_csv(out_dir / "cert_norm_radial_3seed_summary.csv", index=False, encoding="utf-8")
    print("\n=== CERT r4.2 3-SEED NORM-AMPLITUDE SUMMARY ===")
    print(summary.T.to_string())


def cmd_smd_3seed(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(S2_DIR))
    sys.path.insert(0, str(SOFTCLT_DIR))
    from e8_smd_softclt import (
        DATA as DEFAULT_SMD_DATA,
        K_BANK,
        TAU_TEMP,
        TOP_K,
        amplitude_scores,
        load_machine,
        machine_names,
        make_windows,
        soft_instance_labels,
    )
    from soft_ts2vec import TS2Vec as SoftTS2Vec

    data_dir = args.data_dir
    if data_dir is None or not data_dir.exists():
        for cand in (DEFAULT_SMD_DATA, Path(r"E:\MarketRisker\work\cert\OmniAnomaly\ServerMachineDataset")):
            if cand.exists():
                data_dir = cand
                break
    assert data_dir is not None and data_dir.exists(), f"SMD data dir not found: {data_dir}"

    out_dir = OUT_BASE / "smd_3seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = machine_names(data_dir)
    device_id = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{device_id}" if isinstance(device_id, int) else "cpu")
    encode_args = {"casual": True, "sliding_length": 1, "sliding_padding": 30, "batch_size": 32}

    variants = [("T0_hard", False, False), ("T3_inst_soft", True, False), ("T4_soft", True, True)]
    all_rows = []

    for seed in args.seeds:
        seed_csv = out_dir / f"smd_by_machine_seed_{seed}.csv"
        if seed_csv.exists() and not args.force:
            all_rows.extend(pd.read_csv(seed_csv).to_dict(orient="records"))
            print(f"[SKIP] SMD seed={seed}", flush=True)
            continue

        seed_rows = []
        for name in selected:
            t0 = time.time()
            train, test, label = load_machine(name, data_dir)
            mean = train.mean(axis=0)
            std = np.where(train.std(axis=0) < 1e-8, 1.0, train.std(axis=0))
            train_norm = ((train - mean) / std).astype(np.float32)
            test_norm = ((test - mean) / std).astype(np.float32)
            windows, profile = make_windows(train_norm)
            soft_labels, _ = soft_instance_labels(profile)
            _, test_amp = amplitude_scores(train, test)

            for variant, s_inst, s_temp in variants:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                model = SoftTS2Vec(
                    input_dims=train_norm.shape[1],
                    output_dims=64,
                    hidden_dims=64,
                    depth=5,
                    device=device_id,
                    batch_size=16,
                    lr=0.001,
                    lambda_=0.5,
                    tau_temp=TAU_TEMP,
                    max_train_length=None,
                    soft_instance=s_inst,
                    soft_temporal=s_temp,
                )
                model.fit(
                    windows,
                    0,
                    0,
                    0,
                    soft_labels if s_inst else None,
                    str(out_dir / "tmp_runs" / f"s{seed}_{name}_{variant}"),
                    n_epochs=10,
                    verbose=False,
                )
                tr_r = np.asarray(model.encode(train_norm[None, :, :], **encode_args)).reshape(-1, 64)
                te_r = np.asarray(model.encode(test_norm[None, :, :], **encode_args)).reshape(-1, 64)

                # Audit Eq. (7)
                clusters = min(K_BANK, max(1, int(round(0.1 * len(tr_r)))), len(tr_r) - 1)
                km = MiniBatchKMeans(
                    n_clusters=clusters,
                    init="k-means++",
                    random_state=2024,
                    batch_size=max(8192, clusters),
                    max_iter=50,
                    n_init=1,
                    reassignment_ratio=0.01,
                ).fit(tr_r)
                bank = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device)
                bank_r = torch.linalg.norm(bank, dim=1)
                bank_u = torch.nn.functional.normalize(bank, dim=1)

                q_norms = np.linalg.norm(te_r, axis=-1)
                rad_sq_l, tot_sq_l, rad_only_l, cos_l, euc_l = [], [], [], [], []
                with torch.inference_mode():
                    for st in range(0, len(te_r), 4096):
                        q = torch.from_numpy(te_r[st : st + 4096].astype(np.float32)).to(device)
                        rq = torch.linalg.norm(q, dim=1)
                        qu = torch.nn.functional.normalize(q, dim=1)
                        d_cos = (1.0 - qu @ bank_u.T).clamp_min_(0.0)
                        euc = torch.cdist(q, bank)
                        top_euc, top_idx = euc.topk(TOP_K, dim=1, largest=False)
                        rc = bank_r[top_idx]
                        dcos_sel = torch.gather(d_cos, 1, top_idx)
                        rad_sq = (rq.unsqueeze(1) - rc) ** 2
                        ang_tm = 2.0 * rq.unsqueeze(1) * rc * dcos_sel
                        rad_sq_l.append(rad_sq.mean(dim=1).cpu().numpy())
                        tot_sq_l.append((rad_sq + ang_tm).mean(dim=1).cpu().numpy())
                        rad_only_l.append(torch.abs(rq.unsqueeze(1) - rc).mean(dim=1).cpu().numpy())
                        euc_l.append(top_euc.mean(dim=1).cpu().numpy())
                        cos_l.append(d_cos.topk(TOP_K, dim=1, largest=False).values.mean(dim=1).cpu().numpy())

                rad_share = np.concatenate(rad_sq_l) / np.maximum(np.concatenate(tot_sq_l), 1e-12)
                g_test = test_amp[:, 0]
                xn = np.linalg.norm(test_norm, axis=-1)
                norm_m = label == 0
                anom_m = label == 1
                seed_rows.append(
                    {
                        "seed": seed,
                        "machine": name,
                        "variant": variant,
                        "norm_mean_normal": float(q_norms[norm_m].mean()),
                        "norm_cv_normal": float(q_norms[norm_m].std() / max(q_norms[norm_m].mean(), 1e-8)),
                        "norm_mean_anom": float(q_norms[anom_m].mean()) if anom_m.any() else float("nan"),
                        "spearman_norm_vs_xnorm": float(spearmanr(q_norms, xn)[0]),
                        "spearman_norm_vs_magG": float(spearmanr(q_norms, g_test)[0]),
                        "radial_energy_share_normal": float(rad_share[norm_m].mean()),
                        "radial_energy_share_anom": float(rad_share[anom_m].mean()) if anom_m.any() else float("nan"),
                        "pr_auc_radial_only": float(average_precision_score(label, np.concatenate(rad_only_l))),
                        "pr_auc_cosine_bank": float(average_precision_score(label, np.concatenate(cos_l))),
                        "roc_auc_cosine_bank": float(roc_auc_score(label, np.concatenate(cos_l))),
                        "pr_auc_euclidean_bank": float(average_precision_score(label, np.concatenate(euc_l))),
                        "roc_auc_euclidean_bank": float(roc_auc_score(label, np.concatenate(euc_l))),
                    }
                )
            print(f"[DONE] SMD seed={seed} {name} ({time.time() - t0:.1f}s)", flush=True)

        pd.DataFrame(seed_rows).to_csv(seed_csv, index=False, encoding="utf-8")
        all_rows.extend(seed_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "smd_norm_radial_all_machines_3seed.csv", index=False, encoding="utf-8")
    # Macro-average across 28 machines per seed, then mean +/- std across 3 seeds
    by_seed = df.groupby(["variant", "seed"], as_index=False).mean(numeric_only=True)
    by_seed.to_csv(out_dir / "smd_norm_radial_by_seed.csv", index=False, encoding="utf-8")
    summary = by_seed.groupby("variant", as_index=False).mean(numeric_only=True)
    summary.to_csv(out_dir / "smd_norm_radial_3seed_summary.csv", index=False, encoding="utf-8")
    print("\n=== SMD 3-SEED MACRO SUMMARY ===")
    print(summary.T.to_string())


def main() -> None:
    p = argparse.ArgumentParser(description="Exp S5c: 3-Seed Eq. (7) Norm-Amplitude Audit (CERT & SMD)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("cert-3seed")
    p1.add_argument("--seeds", nargs="+", type=int, default=[20261116, 20261117, 20261118])
    p1.add_argument(
        "--s5a-dir",
        type=Path,
        default=REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s5a_norm_intervention",
    )
    p1.add_argument("--t4-root", type=Path, default=None)

    p2 = sub.add_parser("smd-3seed")
    p2.add_argument("--seeds", nargs="+", type=int, default=[20261201, 20261202, 20261203])
    p2.add_argument("--data-dir", type=Path, default=None)
    p2.add_argument("--force", action="store_true")

    args = p.parse_args()
    if args.cmd == "cert-3seed":
        cmd_cert_3seed(args)
    elif args.cmd == "smd-3seed":
        cmd_smd_3seed(args)


if __name__ == "__main__":
    main()
