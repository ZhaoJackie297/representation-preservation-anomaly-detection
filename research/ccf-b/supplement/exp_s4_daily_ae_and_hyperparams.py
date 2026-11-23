"""Exp S4 (P1, Optional High-Value Strengthening): Daily Deep Baselines (MLP-AE / LSTM-AE)
and Hyperparameter Sensitivity (Memory Bank k_bank, top-q, Soft-weight alpha).

Place this script at: <repo_root>/research/ccf-b/supplement/exp_s4_daily_ae_and_hyperparams.py
Run from <repo_root>, e.g.:
  python research/ccf-b/supplement/exp_s4_daily_ae_and_hyperparams.py ae-baseline --dataset r4.2
  python research/ccf-b/supplement/exp_s4_daily_ae_and_hyperparams.py ae-baseline --dataset r5.2
  python research/ccf-b/supplement/exp_s4_daily_ae_and_hyperparams.py alpha-sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
    evaluate_variant,
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


OUT_ROOT = REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s4_baselines_hp"


class DailyMLPAE(nn.Module):
    def __init__(self, in_dim: int = 27, latent_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalLSTMAE(nn.Module):
    """Causal 30-day sequence LSTM autoencoder predicting/reconstructing the sequence."""

    def __init__(self, in_dim: int = 27, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.LSTM(in_dim, hidden_dim, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        return self.proj(h)


def _eval_signal(dataset: str, name: str, sig: np.ndarray, data: dict, active: np.ndarray, train_end: int, val_end: int | None, seed: int) -> list[dict]:
    sig_z = global_train_z(sig, active, train_end)
    rank = sig_z.astype(np.float32, copy=True)
    rank[~np.isfinite(rank)] = -np.inf
    rows = []
    if dataset == "r4.2":
        assert val_end is not None
        for split, (st, ed) in [("validation", (train_end + 1, val_end)), ("test", (val_end + 1, len(data["dates"]) - 1))]:
            pm = point_metrics(sig_z, data, active, st, ed)
            for k in (5, 10, 20):
                alerts = build_alerts(rank, active, k)
                res = evaluate_alert_system(data, alerts, name, k, external=False)
                rows.append({"seed": seed, "method": name, "split": split, "k": k, **compact_metric(res, split), **pm})
    else:
        pm = point_metrics(sig_z, data, active, train_end + 1, len(data["dates"]) - 1)
        for k in (5, 10, 20):
            alerts = build_alerts(rank, active, k)
            res = evaluate_alert_system(data, alerts, name, k, external=True)
            rows.append({"seed": seed, "method": name, "split": "test", "k": k, **compact_metric(res, None), **pm})
    return rows


def cmd_ae_baseline(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir).resolve() if args.output_dir else OUT_ROOT
    out_dir = output_root / f"ae_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "r42" if args.dataset == "r4.2" else "r52"
    processed = CERT_DIR / f"processed_{tag}"
    ds = np.load(processed / f"{args.dataset}_sequence_dataset.npz", allow_pickle=False)
    values = ds["values"].astype(np.float32)
    active = ds["active_mask"][:, :, 0].astype(bool)
    train_end = int(ds["train_end_index"])
    val_end = int(ds["val_end_index"]) if "val_end_index" in ds else None
    data = load_r42() if args.dataset == "r4.2" else load_r52()

    # Replace any NaNs with 0 for neural AE input (already standardized in sequence_dataset.npz)
    clean_x = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    tr_act = active[:, : train_end + 1]
    tr_days = clean_x[:, : train_end + 1][tr_act]  # (M_train, 27)
    tr_seq = clean_x[:, : train_end + 1]           # (N, T_tr, 27)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_rows: list[dict] = []

    for seed in args.seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1) Daily MLP-AE
        mlp = DailyMLPAE(in_dim=clean_x.shape[-1], latent_dim=16).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(torch.from_numpy(tr_days)), batch_size=512, shuffle=True)
        mlp.train()
        for _ in range(args.epochs):
            for (bx,) in loader:
                bx = bx.to(device)
                opt.zero_grad(set_to_none=True)
                loss = ((mlp(bx) - bx) ** 2).mean()
                loss.backward()
                opt.step()
        mlp.eval()
        with torch.inference_mode():
            flat = torch.from_numpy(clean_x.reshape(-1, clean_x.shape[-1]))
            errs = []
            for st in range(0, len(flat), 8192):
                bx = flat[st : st + 8192].to(device)
                errs.append(((mlp(bx) - bx) ** 2).mean(dim=-1).cpu().numpy())
        mlp_score = np.concatenate(errs).reshape(active.shape).astype(np.float32)
        mlp_score[~active] = np.nan
        all_rows.extend(_eval_signal(args.dataset, "MLP-AE-27D", mlp_score, data, active, train_end, val_end, seed))

        # 2) Causal Sequence LSTM-AE (predict/reconstruct active days causally along each user's timeline)
        torch.manual_seed(seed)
        lstm = CausalLSTMAE(in_dim=clean_x.shape[-1], hidden_dim=64).to(device)
        opt2 = torch.optim.AdamW(lstm.parameters(), lr=1e-3, weight_decay=1e-4)
        seq_loader = DataLoader(
            TensorDataset(torch.from_numpy(tr_seq), torch.from_numpy(tr_act.astype(np.float32))),
            batch_size=32, shuffle=True,
        )
        lstm.train()
        for _ in range(args.epochs):
            for bx, bm in seq_loader:
                bx, bm = bx.to(device), bm.to(device)
                opt2.zero_grad(set_to_none=True)
                pred = lstm(bx)
                # Next-step causal prediction error + active mask to prevent trivial identity leakage
                diff = ((pred[:, :-1, :] - bx[:, 1:, :]) ** 2).mean(dim=-1)
                m = bm[:, 1:]
                loss = (diff * m).sum() / m.sum().clamp_min(1.0)
                loss.backward()
                opt2.step()
        lstm.eval()
        with torch.inference_mode():
            full_t = torch.from_numpy(clean_x)
            seq_errs = []
            for st in range(0, len(full_t), 64):
                bx = full_t[st : st + 64].to(device)
                pred = lstm(bx)
                err = torch.zeros(bx.shape[:2], device=device)
                err[:, 1:] = ((pred[:, :-1, :] - bx[:, 1:, :]) ** 2).mean(dim=-1)
                seq_errs.append(err.cpu().numpy())
        lstm_score = np.concatenate(seq_errs, axis=0).astype(np.float32)
        lstm_score[~active] = np.nan
        all_rows.extend(_eval_signal(args.dataset, "LSTM-Pred-27D", lstm_score, data, active, train_end, val_end, seed))
        print(f"[DONE] {args.dataset} AE/LSTM seed={seed} ({time.time() - t0:.1f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "ae_baselines_by_seed.csv", index=False, encoding="utf-8")
    summary = (
        df.groupby(["method", "split", "k"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            incident_hits_mean=("incident_hits", "mean"),
        )
    )
    summary.to_csv(out_dir / "ae_baselines_summary.csv", index=False, encoding="utf-8")
    print(summary[(summary["split"] == "test") & (summary["k"] == 10)].to_string(index=False))


def cmd_alpha_sweep(args: argparse.Namespace) -> None:
    """Hyperparameter sensitivity for soft-instance weight cap alpha in {0.25, 0.50, 0.75} on r4.2."""
    out_dir = OUT_ROOT / "alpha_sweep_r42"
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = CERT_DIR / "processed_r42"
    ds = np.load(processed / "r4.2_sequence_dataset.npz", allow_pickle=False)
    values = ds["values"].astype(np.float32)
    users = ds["users"].astype(str)
    dates = ds["dates"].astype(str)
    active = ds["active_mask"][:, :, 0].astype(bool)
    train_end = int(ds["train_end_index"])
    val_end = int(ds["val_end_index"])
    train_values = values[:, : train_end + 1]
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
    raw_20 = align_raw_features(raw_csv, users, dates)
    G, Q = aggregate_components(raw_20, active, train_end, user_specific=True, aggregation="channel_mean")
    G_z = global_train_z(G, active, train_end)
    Q_z = global_train_z(Q, active, train_end)

    device_id = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{device_id}" if isinstance(device_id, int) else "cpu")
    n_steps = (values.shape[0] // 16) * 10
    rows = []

    for alpha in args.alphas:
        soft_labels, _ = soft_instance_labels(profiles, alpha=alpha)
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            batches = []
            per_ep = values.shape[0] // 16
            while len(batches) < n_steps:
                perm = rng.permutation(values.shape[0])
                for i in range(per_ep):
                    batches.append(tuple(int(x) for x in perm[i * 16 : (i + 1) * 16]))
                    if len(batches) >= n_steps:
                        break
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = CoverageSoftTS2Vec(
                input_dims=values.shape[-1], output_dims=64, hidden_dims=64, depth=5,
                device=device_id, lr=1e-3, batch_size=16, lambda_=0.5, tau_temp=1.5,
                soft_instance=True, soft_temporal=False,
            )
            model.fit_coverage_table(train_data=train_values, soft_labels=soft_labels, batch_table=batches, n_epochs=1)
            enc = {"causal": True, "sliding_length": 1, "sliding_padding": 30, "batch_size": 256}
            plain_r = np.asarray(model.encode(values, **enc), dtype=np.float32)
            cos_raw, euc_raw = bank_scores(plain_r, plain_r, active, train_end, device)
            cos_z = global_train_z(cos_raw, active, train_end)
            euc_z = global_train_z(euc_raw, active, train_end)
            pai = euc_z + PAI_W_G * G_z + PAI_W_Q * Q_z
            for sc_tag, arr in [("bank_cosine", cos_z), ("bank_euclidean", euc_z), ("pai_a1", pai)]:
                for r in evaluate_variant(f"alpha_{alpha}_{sc_tag}", arr, data, active, train_end, val_end, seed):
                    r["alpha"] = alpha
                    r["scorer"] = sc_tag
                    rows.append(r)
            print(f"[DONE] alpha={alpha} seed={seed}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "alpha_sweep_by_seed.csv", index=False, encoding="utf-8")
    summary = df.groupby(["alpha", "scorer", "split", "k"], as_index=False).agg(
        pr_auc_mean=("pr_auc", "mean"), roc_auc_mean=("roc_auc", "mean"), incident_hits_mean=("incident_hits", "mean")
    )
    summary.to_csv(out_dir / "alpha_sweep_summary.csv", index=False, encoding="utf-8")
    print(summary[(summary["split"] == "test") & (summary["k"] == 10)].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp S4: Daily AE/LSTM baselines & alpha sweep")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("ae-baseline")
    p1.add_argument("--dataset", choices=["r4.2", "r5.2"], required=True)
    p1.add_argument("--seeds", nargs="+", type=int, default=[20261116, 20261117, 20261118])
    p1.add_argument("--epochs", type=int, default=20)
    p1.add_argument(
        "--output-dir",
        default=None,
        help="Output root for ae_<dataset>; defaults to the historical S4 directory.",
    )

    p2 = sub.add_parser("alpha-sweep")
    p2.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p2.add_argument("--seeds", nargs="+", type=int, default=[20261116, 20261117, 20261118])

    args = parser.parse_args()
    if args.cmd == "ae-baseline":
        cmd_ae_baseline(args)
    else:
        cmd_alpha_sweep(args)


if __name__ == "__main__":
    main()
