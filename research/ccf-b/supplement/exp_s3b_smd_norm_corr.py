"""Exp S3b (P0/P1, SMD ~5 min on GPU): Representation Norm vs. Input Amplitude & Eq. (7)
Radial-Angular Decomposition on SMD (28 servers, T0-Hard vs T4-Soft, seed 20261201).

Place this script at: <repo_root>/research/ccf-b/supplement/exp_s3b_smd_norm_corr.py
Run from <repo_root>:
  python research/ccf-b/supplement/exp_s3b_smd_norm_corr.py --seed 20261201

Directly compares SMD radial energy share & norm-amplitude correlation against CERT
(exp_s3 norm-corr) to explain quantitatively why Euclidean beats Cosine by +0.322 on SMD
while Cosine slightly leads Euclidean under T4 on CERT r4.2/r5.2.
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
S2_DIR = REPO_ROOT / "code" / "by_layer" / "S2_dual_end"
SOFTCLT_DIR = REPO_ROOT / "references" / "softclt_repo" / "softclt_ts2vec"
sys.path.insert(0, str(S2_DIR))
sys.path.insert(0, str(SOFTCLT_DIR))

from e8_smd_softclt import (  # noqa: E402
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
from soft_ts2vec import TS2Vec as SoftTS2Vec  # noqa: E402


OUT_ROOT = REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s3_diagnostics" / "norm_corr_smd"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp S3b: SMD Norm-Amplitude & Eq. (7) Radial-Angular Audit")
    p.add_argument("--seed", type=int, default=20261201)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_SMD_DATA)
    p.add_argument("--machines", nargs="*", default=[])
    p.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    return p.parse_args()


def audit_repr(
    train_repr: np.ndarray,
    test_repr: np.ndarray,
    label: np.ndarray,
    test_amp: np.ndarray,
    test_norm_x: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    clusters = min(K_BANK, max(1, int(round(0.1 * len(train_repr)))))
    clusters = min(clusters, len(train_repr) - 1)
    km = MiniBatchKMeans(
        n_clusters=clusters, init="k-means++", random_state=2024,
        batch_size=max(8192, clusters), max_iter=50, n_init=1, reassignment_ratio=0.01,
    ).fit(train_repr)
    bank = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device)
    bank_r = torch.linalg.norm(bank, dim=1)
    bank_u = torch.nn.functional.normalize(bank, dim=1)

    q_norms = np.linalg.norm(test_repr, axis=-1)
    rad_sq_l, ang_l, tot_sq_l, rad_only_l, cos_l, euc_l = [], [], [], [], [], []
    with torch.inference_mode():
        for st in range(0, len(test_repr), 4096):
            q = torch.from_numpy(test_repr[st : st + 4096].astype(np.float32)).to(device)
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
            ang_l.append(ang_tm.mean(dim=1).cpu().numpy())
            tot_sq_l.append((rad_sq + ang_tm).mean(dim=1).cpu().numpy())
            rad_only_l.append(torch.abs(rq.unsqueeze(1) - rc).mean(dim=1).cpu().numpy())
            euc_l.append(top_euc.mean(dim=1).cpu().numpy())
            cos_l.append(d_cos.topk(TOP_K, dim=1, largest=False).values.mean(dim=1).cpu().numpy())

    rad_sq_v = np.concatenate(rad_sq_l)
    tot_sq_v = np.maximum(np.concatenate(tot_sq_l), 1e-12)
    rad_share = rad_sq_v / tot_sq_v
    rad_only = np.concatenate(rad_only_l)
    cos_only = np.concatenate(cos_l)
    euc_only = np.concatenate(euc_l)

    g_test = test_amp[:, 0]
    q_test = test_amp[:, 1]
    xn = np.linalg.norm(test_norm_x, axis=-1)
    norm_m = label == 0
    anom_m = label == 1

    return {
        "norm_mean_normal": float(q_norms[norm_m].mean()),
        "norm_std_normal": float(q_norms[norm_m].std()),
        "norm_cv_normal": float(q_norms[norm_m].std() / max(q_norms[norm_m].mean(), 1e-8)),
        "norm_mean_anom": float(q_norms[anom_m].mean()) if anom_m.any() else float("nan"),
        "spearman_norm_vs_magG": float(spearmanr(q_norms, g_test)[0]),
        "pearson_norm_vs_magG": float(pearsonr(q_norms, g_test)[0]),
        "spearman_norm_vs_T2": float(spearmanr(q_norms, q_test)[0]),
        "spearman_norm_vs_xnorm": float(spearmanr(q_norms, xn)[0]),
        "radial_energy_share_normal": float(rad_share[norm_m].mean()),
        "radial_energy_share_anom": float(rad_share[anom_m].mean()) if anom_m.any() else float("nan"),
        "pr_auc_radial_only": float(average_precision_score(label, rad_only)),
        "roc_auc_radial_only": float(roc_auc_score(label, rad_only)),
        "pr_auc_cosine_bank": float(average_precision_score(label, cos_only)),
        "roc_auc_cosine_bank": float(roc_auc_score(label, cos_only)),
        "pr_auc_euclidean_bank": float(average_precision_score(label, euc_only)),
        "roc_auc_euclidean_bank": float(roc_auc_score(label, euc_only)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.machines or machine_names(args.data_dir)
    device_id = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{device_id}" if isinstance(device_id, int) else "cpu")
    # This script imports the upstream SoftCLT TS2Vec implementation, whose
    # historical API name is `casual` (not the local wrapper's `causal`).
    encode_args = {"casual": True, "sliding_length": 1, "sliding_padding": 30, "batch_size": 32}

    rows = []
    for name in selected:
        t0 = time.time()
        train, test, label = load_machine(name, args.data_dir)
        mean = train.mean(axis=0)
        std = np.where(train.std(axis=0) < 1e-8, 1.0, train.std(axis=0))
        train_norm = ((train - mean) / std).astype(np.float32)
        test_norm = ((test - mean) / std).astype(np.float32)
        windows, profile = make_windows(train_norm)
        soft_labels, _ = soft_instance_labels(profile)
        _, test_amp = amplitude_scores(train, test)

        for variant, s_inst, s_temp in [("T0_hard", False, False), ("T4_soft", True, True)]:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            model = SoftTS2Vec(
                input_dims=train_norm.shape[1], output_dims=64, hidden_dims=64, depth=5,
                device=device_id, batch_size=16, lr=0.001, lambda_=0.5, tau_temp=TAU_TEMP,
                max_train_length=None, soft_instance=s_inst, soft_temporal=s_temp,
            )
            model.fit(
                windows, 0, 0, 0,
                soft_labels if s_inst else None,
                str(args.output_dir / "tmp_runs" / f"{name}_{variant}"),
                n_epochs=10, verbose=False,
            )
            tr_r = np.asarray(model.encode(train_norm[None, :, :], **encode_args)).reshape(-1, 64)
            te_r = np.asarray(model.encode(test_norm[None, :, :], **encode_args)).reshape(-1, 64)
            metrics = audit_repr(tr_r, te_r, label, test_amp, test_norm, device)
            rows.append({"machine": name, "seed": args.seed, "variant": variant, **metrics})
        print(f"[DONE] {name} ({time.time() - t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "smd_norm_amplitude_by_machine.csv", index=False, encoding="utf-8")
    summary = df.groupby("variant", as_index=False).mean(numeric_only=True)
    summary.to_csv(args.output_dir / "smd_norm_amplitude_summary.csv", index=False, encoding="utf-8")
    print("=== SMD MACRO-MEAN NORM-AMPLITUDE & RADIAL DECOMPOSITION ===")
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
