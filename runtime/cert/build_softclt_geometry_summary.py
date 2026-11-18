from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "experiments" / "representation_preservation"
CANONICAL = BASE / "canonical"


def main() -> None:
    e2_mechanism = pd.read_csv(
        BASE / "e2" / "mechanism_metrics_summary.csv"
    ).rename(columns={"variant": "variant_id"})
    instance_mechanism = pd.read_csv(
        BASE
        / "e2_instance"
        / "missing_safe_v2_20261117"
        / "mechanism_summary.csv"
    ).rename(columns={"variant": "variant_id"})
    mechanism = pd.concat(
        [e2_mechanism, instance_mechanism],
        ignore_index=True,
    )

    configuration = {
        "T0": ("hard", "hard", 0.5),
        "T1": ("none", "hard", 0.0),
        "T2_tau1.5": ("none", "soft", 0.0),
        "T3_instance_soft": ("soft", "hard", 0.5),
        "T4_full_soft": ("soft", "soft", 0.5),
    }
    mechanism["instance_term"] = mechanism["variant_id"].map(
        {key: value[0] for key, value in configuration.items()}
    )
    mechanism["temporal_term"] = mechanism["variant_id"].map(
        {key: value[1] for key, value in configuration.items()}
    )
    mechanism["lambda_instance"] = mechanism["variant_id"].map(
        {key: value[2] for key, value in configuration.items()}
    )

    metric_map = {
        "T0": "T0_T0_global_z",
        "T1": "T1_T1_global_z",
        "T2_tau1.5": "T2_tau1.5_T2_tau1.5_global_z",
        "T3_instance_soft": "T3_instance_soft_global_z",
        "T4_full_soft": "T4_full_soft_global_z",
    }
    metrics = pd.read_csv(CANONICAL / "canonical_metric_summary.csv")
    instance_metrics = pd.read_csv(
        BASE
        / "e2_instance"
        / "missing_safe_v2_20261117"
        / "metrics_by_seed.csv"
    )
    rows = []
    for variant_id, method_id in metric_map.items():
        if variant_id in {"T3_instance_soft", "T4_full_soft"}:
            selected = (
                instance_metrics[
                    (instance_metrics["method"] == method_id)
                    & (instance_metrics["k"] == 10)
                ]
                .groupby("split", as_index=False)
                .agg(
                    pr_auc_mean=("pr_auc", "mean"),
                    roc_auc_mean=("roc_auc", "mean"),
                    incident_hits_mean=("incident_hits", "mean"),
                )
            )
        else:
            selected = metrics[
                (metrics["dataset"] == "r4.2")
                & (metrics["method_id"] == method_id)
                & (metrics["k"] == 10)
            ]
        if len(selected) != 2:
            raise RuntimeError(
                f"Expected validation/test rows for {method_id}"
            )
        row = {"variant_id": variant_id}
        for _, metric_row in selected.iterrows():
            split = metric_row["split"]
            row[f"{split}_pr_auc"] = metric_row["pr_auc_mean"]
            row[f"{split}_roc_auc"] = metric_row["roc_auc_mean"]
            row[f"{split}_incident_hits_k10"] = metric_row[
                "incident_hits_mean"
            ]
        rows.append(row)
    performance = pd.DataFrame(rows)

    result = mechanism.merge(
        performance,
        on="variant_id",
        validate="one_to_one",
    )
    order = ["T0", "T1", "T2_tau1.5", "T3_instance_soft", "T4_full_soft"]
    result["variant_order"] = result["variant_id"].map(
        {name: index for index, name in enumerate(order)}
    )
    result = result.sort_values("variant_order").drop(
        columns="variant_order"
    )
    result.to_csv(
        CANONICAL / "softclt_mechanism_geometry.csv",
        index=False,
    )

    t3 = result[result["variant_id"] == "T3_instance_soft"].iloc[0]
    t4 = result[result["variant_id"] == "T4_full_soft"].iloc[0]
    delta = {
        "variant_id": "T4_minus_T3",
        "knn_overlap_mean": (
            t4["knn_overlap_mean"] - t3["knn_overlap_mean"]
        ),
        "local_distance_spearman_mean": (
            t4["local_distance_spearman_mean"]
            - t3["local_distance_spearman_mean"]
        ),
        "effective_rank_mean": (
            t4["effective_rank_mean"] - t3["effective_rank_mean"]
        ),
        "mean_norm_mean": t4["mean_norm_mean"] - t3["mean_norm_mean"],
        "validation_pr_auc": (
            t4["validation_pr_auc"] - t3["validation_pr_auc"]
        ),
        "test_pr_auc": t4["test_pr_auc"] - t3["test_pr_auc"],
        "validation_roc_auc": (
            t4["validation_roc_auc"] - t3["validation_roc_auc"]
        ),
        "test_roc_auc": t4["test_roc_auc"] - t3["test_roc_auc"],
        "test_incident_hits_k10": (
            t4["test_incident_hits_k10"] - t3["test_incident_hits_k10"]
        ),
    }
    pd.DataFrame([delta]).to_csv(
        CANONICAL / "softclt_geometry_T4_minus_T3.csv",
        index=False,
    )

    spectral_path = (
        BASE
        / "s2_graph_spectral"
        / "three_seed_20261117"
        / "spectral_metrics_by_seed.csv"
    )
    spectral = pd.read_csv(spectral_path)
    metric_columns = [
        "instance_dirichlet_mean",
        "temporal_dirichlet_mean",
        "instance_highpass_roc",
        "temporal_highpass_roc",
        "instance_highpass_pr",
        "temporal_highpass_pr",
    ]
    contrasts = [
        ("T2_minus_T1", "T1", "T2_tau1.5"),
        ("T3_minus_T0", "T0", "T3_instance_soft"),
        ("T4_minus_T3", "T3_instance_soft", "T4_full_soft"),
    ]
    contrast_rows = []
    for label, control, treatment in contrasts:
        left = spectral[
            spectral["variant"] == treatment
        ].set_index("seed")
        right = spectral[
            spectral["variant"] == control
        ].set_index("seed")
        for metric in metric_columns:
            difference = (left[metric] - right[metric]).dropna()
            contrast_rows.append(
                {
                    "contrast": label,
                    "metric": metric,
                    "mean_difference": float(difference.mean()),
                    "positive": int((difference > 0).sum()),
                    "negative": int((difference < 0).sum()),
                    "zero": int((difference == 0).sum()),
                    "n_seeds": int(len(difference)),
                }
            )
    pd.DataFrame(contrast_rows).to_csv(
        CANONICAL / "softclt_spectral_contrasts.csv",
        index=False,
    )
    print(result.to_string(index=False))
    print(pd.DataFrame([delta]).to_string(index=False))
    print(
        pd.DataFrame(contrast_rows).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
