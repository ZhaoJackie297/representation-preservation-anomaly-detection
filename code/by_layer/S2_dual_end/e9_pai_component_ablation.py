from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
OUT = (
    ROOT
    / "experiments"
    / "representation_preservation"
    / "e9"
)
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [20261116, 20261117, 20261118]
K_GRID = (5, 10, 20)

sys.path.insert(0, str(ROOT))
from e0_e1_representation_preservation import global_train_z  # noqa: E402
from e3_multichannel_amplitude import evaluate_variant  # noqa: E402
from e7_smd_pai import (  # noqa: E402
    amplitude_scores,
    evaluate_machine,
    z_from_train,
)
from quota_relationship_sim import load_r42  # noqa: E402
from smd_pipeline import load_machine, machine_names  # noqa: E402


PAI_W_G = 0.6
PAI_W_Q = 0.4


def certified_ablation() -> pd.DataFrame:
    dataset = np.load(
        ROOT / "processed_r42" / "r4.2_sequence_dataset.npz",
        allow_pickle=False,
    )
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    data = load_r42()
    components = np.load(
        ROOT
        / "experiments"
        / "representation_preservation"
        / "e3"
        / "a3_components.npz",
        allow_pickle=False,
    )
    rows = []
    for seed in SEEDS:
        scores = np.load(
            ROOT
            / "experiments"
            / "representation_preservation"
            / "e0_e1"
            / "r42"
            / f"seed_{seed}"
            / "scores.npz",
            allow_pickle=False,
        )
        euc = global_train_z(scores["euc_score"], active, train_end)
        g = global_train_z(
            components[f"seed_{seed}_G"],
            active,
            train_end,
        )
        q = global_train_z(
            components[f"seed_{seed}_Q"],
            active,
            train_end,
        )
        variants = {
            "euc": euc,
            "euc_plus_0.6G": euc + 0.6 * g,
            "euc_plus_0.4T2": euc + 0.4 * q,
            "euc_plus_PAI": euc + PAI_W_G * g + PAI_W_Q * q,
        }
        for name, signal in variants.items():
            rows.extend(
                evaluate_variant(
                    name,
                    signal,
                    data,
                    active,
                    train_end,
                    val_end,
                    seed,
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "cert_r42_metrics_by_seed.csv",
        index=False,
        encoding="utf-8",
    )
    summary = (
        frame.groupby(["method", "split", "k"], as_index=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            incident_hits_mean=("incident_hits", "mean"),
        )
        .sort_values(["split", "k", "method"])
    )
    summary.to_csv(
        OUT / "cert_r42_metric_summary.csv",
        index=False,
        encoding="utf-8",
    )
    del users, dates
    return frame


def smd_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    score_rows = []
    baseline_root = (
        ROOT
        / "experiments"
        / "representation_preservation"
        / "e7"
        / "smd"
        / "machines"
    )
    for name in machine_names():
        path = baseline_root / f"{name}.npz"
        if not path.exists():
            continue
        train, test, label = load_machine(name)
        score_arrays = np.load(path, allow_pickle=False)
        train_euc = score_arrays["euc_train_score"]
        test_euc = score_arrays["euc_test_score"]
        train_euc_z, test_euc_z = z_from_train(train_euc, test_euc)
        train_amp, test_amp = amplitude_scores(train, test)
        train_g_z, test_g_z = z_from_train(
            train_amp[:, 0],
            test_amp[:, 0],
        )
        train_q_z, test_q_z = z_from_train(
            train_amp[:, 1],
            test_amp[:, 1],
        )
        variants = {
            "euc": (train_euc_z, test_euc_z),
            "euc_plus_0.6G": (
                train_euc_z + 0.6 * train_g_z,
                test_euc_z + 0.6 * test_g_z,
            ),
            "euc_plus_0.4T2": (
                train_euc_z + 0.4 * train_q_z,
                test_euc_z + 0.4 * test_q_z,
            ),
            "euc_plus_PAI": (
                train_euc_z + PAI_W_G * train_g_z + PAI_W_Q * train_q_z,
                test_euc_z + PAI_W_G * test_g_z + PAI_W_Q * test_q_z,
            ),
        }
        for variant, (train_score, test_score) in variants.items():
            metric = evaluate_machine(
                train,
                test,
                label,
                train_score,
                test_score,
            )
            score_rows.append(
                {
                    "machine": name,
                    "method": variant,
                    **metric,
                }
            )
    frame = pd.DataFrame(score_rows)
    frame.to_csv(
        OUT / "smd_metrics_by_machine.csv",
        index=False,
        encoding="utf-8",
    )
    paired = []
    for comparator in (
        "euc_plus_0.6G",
        "euc_plus_0.4T2",
        "euc_plus_PAI",
    ):
        for metric in ("roc_auc", "pr_auc", "pa_f1_top5"):
            left = frame[frame["method"] == comparator].set_index("machine")
            right = frame[frame["method"] == "euc"].set_index("machine")
            delta = left[metric] - right[metric]
            paired.append(
                {
                    "comparison": f"{comparator}_vs_euc",
                    "metric": metric,
                    "mean_difference": float(delta.mean()),
                    "positive_machines": int((delta > 0).sum()),
                    "negative_machines": int((delta < 0).sum()),
                    "wilcoxon_p": (
                        float(wilcoxon(delta).pvalue)
                        if np.any(delta.to_numpy() != 0)
                        else None
                    ),
                }
            )
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(
        OUT / "smd_component_tests.csv",
        index=False,
        encoding="utf-8",
    )
    del rows
    return frame, paired_frame


def main() -> None:
    cert = certified_ablation()
    smd, smd_tests = smd_ablation()
    payload = {
        "cert_rows": len(cert),
        "smd_machines": int(smd["machine"].nunique()),
        "weights": {"magG": PAI_W_G, "T2": PAI_W_Q},
        "interpretation_boundary": (
            "The individual-component ablation tests whether magG and T2 "
            "add value beyond the Euclidean representation-bank score; "
            "a non-significant result must not be reported as proven gain."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(smd_tests.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
