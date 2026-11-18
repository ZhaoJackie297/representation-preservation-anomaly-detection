from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "experiments" / "representation_preservation"
CANONICAL = BASE / "canonical"
OUTPUT_TAG = "missing_safe_v2_20261117"
SEEDS = (20261116, 20261117, 20261118)

sys.path.insert(0, str(ROOT))
from e0_e1_representation_preservation import (  # noqa: E402
    align_raw_features,
    global_train_z,
)
from e3_multichannel_amplitude import (  # noqa: E402
    aggregate_components,
    evaluate_variant,
)
from quota_relationship_sim import load_r42  # noqa: E402


METRIC_COLUMNS = [
    "dataset",
    "protocol_id",
    "method_id",
    "split",
    "k",
    "seed",
    "incident_hits",
    "incident_total",
    "event_recall_3d",
    "event_recall_1d",
    "precision_at_k",
    "false_alerts_per_day",
    "scenario_macro_recall_3d",
    "pr_auc",
    "roc_auc",
    "positive_days",
    "source_file",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_metrics(
    frame: pd.DataFrame,
    *,
    dataset: str,
    protocol_id: str,
    method_id: str | None,
    source_file: Path,
) -> pd.DataFrame:
    result = frame.copy()
    result["dataset"] = dataset
    result["protocol_id"] = protocol_id
    if method_id is not None:
        result["method_id"] = method_id
    elif "method_id" not in result:
        result["method_id"] = np.nan
    result["source_file"] = str(source_file)
    for column in METRIC_COLUMNS:
        if column not in result:
            result[column] = np.nan
    numeric_columns = [
        "k",
        "seed",
        "incident_hits",
        "incident_total",
        "event_recall_3d",
        "event_recall_1d",
        "precision_at_k",
        "false_alerts_per_day",
        "scenario_macro_recall_3d",
        "pr_auc",
        "roc_auc",
        "positive_days",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[METRIC_COLUMNS]


def read_metrics(
    path: Path,
    *,
    dataset: str,
    protocol_id: str,
    method_id: str,
    method_filter: set[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if method_filter is not None:
        if "method" not in frame:
            raise ValueError(f"Missing method column: {path}")
        frame = frame[frame["method"].isin(method_filter)].copy()
    return normalized_metrics(
        frame,
        dataset=dataset,
        protocol_id=protocol_id,
        method_id=method_id,
        source_file=path,
    )


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dataset",
        "protocol_id",
        "method_id",
        "split",
        "k",
    ]
    summary = (
        frame.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            event_recall_3d_mean=("event_recall_3d", "mean"),
            event_recall_3d_std=("event_recall_3d", "std"),
            precision_at_k_mean=("precision_at_k", "mean"),
            precision_at_k_std=("precision_at_k", "std"),
            incident_hits_mean=("incident_hits", "mean"),
            incident_hits_std=("incident_hits", "std"),
        )
        .sort_values(group_columns)
    )
    return summary


def paired_difference(
    frame: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    treatment_method: str,
    control_method: str,
    treatment_label: str,
    control_label: str,
) -> dict:
    keys = ["seed", "k"]
    treatment = frame[
        (frame["dataset"] == dataset)
        & (frame["split"] == split)
        & (frame["method_id"] == treatment_method)
    ][keys + ["pr_auc", "roc_auc", "incident_hits"]]
    control = frame[
        (frame["dataset"] == dataset)
        & (frame["split"] == split)
        & (frame["method_id"] == control_method)
    ][keys + ["pr_auc", "roc_auc", "incident_hits"]]
    paired = treatment.merge(
        control,
        on=keys,
        suffixes=("_treatment", "_control"),
        validate="one_to_one",
    )

    def diff_summary(values: pd.Series) -> dict:
        values = values.dropna()
        nonzero = values[np.abs(values) > 1e-15]
        p_value = None
        if len(nonzero) > 0:
            p_value = float(wilcoxon(nonzero).pvalue)
        return {
            "mean_difference": float(values.mean()),
            "median_difference": float(values.median()),
            "positive": int((values > 1e-15).sum()),
            "negative": int((values < -1e-15).sum()),
            "zero": int((np.abs(values) <= 1e-15).sum()),
            "wilcoxon_p": p_value,
        }

    result = {
        "dataset": dataset,
        "split": split,
        "treatment": treatment_label,
        "control": control_label,
        "n_paired_seeds": int(paired["seed"].nunique()),
        "point_metrics": {},
        "incident_hits_by_k": {},
    }
    for metric in ("pr_auc", "roc_auc"):
        point = paired.drop_duplicates(["seed"])
        result["point_metrics"][metric] = diff_summary(
            point[f"{metric}_treatment"]
            - point[f"{metric}_control"]
        )
    for k in sorted(paired["k"].dropna().unique()):
        subset = paired[paired["k"] == k]
        result["incident_hits_by_k"][str(int(k))] = diff_summary(
            subset["incident_hits_treatment"]
            - subset["incident_hits_control"]
        )
    return result


def build_protocol_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protocol_id": "P0_legacy_production_3_8",
                "experiment": "AI.txt section 3.8",
                "score_path": "S2 masked score -> per-user training quantile -> legacy S3",
                "postprocessing": "legacy dynamic risk and top-K queue",
                "comparable_with": "Only other runs of the same legacy production protocol",
                "allowed_use": "Historical production-protocol evidence",
                "forbidden_use": "Do not mix with E0-E10 raw-score tables",
            },
            {
                "protocol_id": "P1_e0e1_raw_s2",
                "experiment": "E0/E1",
                "score_path": "Native TS2Vec raw masked score, training-period global z",
                "postprocessing": "Direct point metrics and fixed daily K with cooldown=7",
                "comparable_with": "E3 component audit and same raw-score runs",
                "allowed_use": "Canonical S2 baseline for PAI component comparisons",
                "forbidden_use": "Do not substitute 3.8 production numbers",
            },
            {
                "protocol_id": "P2_e2_softclt_internal",
                "experiment": "E2",
                "score_path": "SoftTS2Vec internal T0 reference, training-period global z",
                "postprocessing": "Same K grid and cooldown within E2",
                "comparable_with": "E2 T1/T2 only",
                "allowed_use": "Canonical temporal-SoftCLT comparison",
                "forbidden_use": "Do not treat as bit-identical to E0/E1 T0",
            },
            {
                "protocol_id": "P3_e3_component_audit",
                "experiment": "E3",
                "score_path": "E1 Euclidean bank score plus project amplitude components",
                "postprocessing": "Training-period global z and fixed daily K",
                "comparable_with": "E0/E1 S2 and E4 joint T4 using the same component",
                "allowed_use": "Canonical magG/T2/PAI component audit",
                "forbidden_use": "Do not call the project 1.0:0.6:0.4 weights official",
            },
            {
                "protocol_id": "P4_e4_joint_corrected",
                "experiment": "E4 corrected",
                "score_path": "Corrected T3/T4 representation bank plus A1 components",
                "postprocessing": "Training-period global z and fixed daily K",
                "comparable_with": "P3 T0 A1/A3 and matched T0 runs",
                "allowed_use": "Canonical dual-end joint result after missing-value repair",
                "forbidden_use": "Do not combine with superseded 47D profiles",
            },
            {
                "protocol_id": "P5_e5_r52_raw",
                "experiment": "E5",
                "score_path": "Native r5.2 representation bank and components",
                "postprocessing": "Training-period global z and fixed daily K",
                "comparable_with": "Corrected r5.2 E4 T4 joint run",
                "allowed_use": "Canonical r5.2 T0 control",
                "forbidden_use": "Do not substitute r4.2 baselines",
            },
            {
                "protocol_id": "P6_e4_r62_matched_t0",
                "experiment": "E4 r6.2 matched T0",
                "score_path": "Native r6.2 model on the same fixed 1000-user subset",
                "postprocessing": "Training-period global z and fixed daily K",
                "comparable_with": "Corrected r6.2 E4 T4 joint run",
                "allowed_use": "Canonical r6.2 matched control",
                "forbidden_use": "Do not mix with full-user r6.2 T0",
            },
            {
                "protocol_id": "P7_e11_iforest_27d",
                "experiment": "E11",
                "score_path": "IsolationForest on active CERT user-day rows",
                "postprocessing": "Training-only fit; score negation and fixed daily K",
                "comparable_with": "P1/P3 T0 and P4 T4 under the same alerts",
                "allowed_use": "External classical tabular baseline",
                "forbidden_use": "Do not call it a temporal model",
            },
        ]
    )


def build_pai_variant_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_id": "A1_user_channel_mean_project_weights",
                "user_statistics": "Per-user robust median/MAD with global fallback",
                "channel_aggregation": "Five-channel max, then mean across channels",
                "temporal_component": "32-day causal mean shift",
                "weights_eu_magG_T2": "1.0:0.6:0.4",
                "official_weight_equivalent": False,
                "source_implementation": "e3_multichannel_amplitude.aggregate_components",
                "notes": "This is PAI-A1 in the joint experiment, not E0/E1 S5.",
            },
            {
                "variant_id": "A3_user_channel_top2_project_weights",
                "user_statistics": "Per-user robust median/MAD with global fallback",
                "channel_aggregation": "Five-channel max, then top-2 mean",
                "temporal_component": "32-day causal mean shift",
                "weights_eu_magG_T2": "1.0:0.6:0.4",
                "official_weight_equivalent": False,
                "source_implementation": "e0_e1_representation_preservation.amplitude_components",
                "notes": "E0/E1 S5_pai_full_global is this variant; E3 A3 reproduces it.",
            },
            {
                "variant_id": "A1_user_channel_mean_official_weights",
                "user_statistics": "Per-user robust median/MAD with global fallback",
                "channel_aggregation": "Five-channel max, then mean across channels",
                "temporal_component": "32-day causal mean shift",
                "weights_eu_magG_T2": "0.6:0.4:0.2",
                "official_weight_equivalent": True,
                "source_implementation": "PAI official scalar weights applied to project components",
                "notes": "Scalar-weight sensitivity only; component implementation is still local.",
            },
            {
                "variant_id": "PAI_official_repository",
                "user_statistics": "Defined by official PAI aggregation pipeline",
                "channel_aggregation": "Defined by official PAI aggregation pipeline",
                "temporal_component": "Official T2 component",
                "weights_eu_magG_T2": "0.6:0.4:0.2",
                "official_weight_equivalent": True,
                "source_implementation": "references/PAI_audit/aggregators/build_full_table.py",
                "notes": "Not byte-identical to the local CERT component implementation.",
            },
        ]
    )


def build_component_and_weight_metrics() -> tuple[pd.DataFrame, list[Path]]:
    dataset_path = ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
    raw_path = ROOT / "processed_r42" / "r4.2_user_day_features.csv"
    dataset = np.load(dataset_path, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = dataset["dates"].astype(str)
    active = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    data = load_r42()
    raw = align_raw_features(raw_path, users, dates)

    a1_g, a1_q = aggregate_components(
        raw,
        active,
        train_end,
        user_specific=True,
        aggregation="channel_mean",
    )
    a3_g, a3_q = aggregate_components(
        raw,
        active,
        train_end,
        user_specific=True,
        aggregation="channel_top2",
    )
    a1_g_z = global_train_z(a1_g, active, train_end)
    a1_q_z = global_train_z(a1_q, active, train_end)
    a3_g_z = global_train_z(a3_g, active, train_end)
    a3_q_z = global_train_z(a3_q, active, train_end)

    rows: list[dict] = []
    sources: list[Path] = [dataset_path, raw_path]
    score_root = BASE / "e0_e1" / "r42"
    for seed in SEEDS:
        score_path = score_root / f"seed_{seed}" / "scores.npz"
        scores = np.load(score_path, allow_pickle=False)
        sources.append(score_path)
        masked_z = global_train_z(
            scores["masked_raw"],
            active,
            train_end,
        )
        cos_z = global_train_z(
            scores["cos_score"],
            active,
            train_end,
        )
        euc_z = global_train_z(scores["euc_score"], active, train_end)
        project_a1 = euc_z + 0.6 * a1_g_z + 0.4 * a1_q_z
        official_a1 = 0.6 * euc_z + 0.4 * a1_g_z + 0.2 * a1_q_z
        project_a3 = euc_z + 0.6 * a3_g_z + 0.4 * a3_q_z
        official_a3 = 0.6 * euc_z + 0.4 * a3_g_z + 0.2 * a3_q_z
        variants = {
            "T0_mask_original": masked_z,
            "T0_bank_cosine": cos_z,
            "T0_bank_euclidean": euc_z,
            "T0_pai_mag": euc_z + 0.6 * a3_g_z,
            "T0_pai_t2": euc_z + 0.4 * a3_q_z,
            "T0_A1_project_weights": project_a1,
            "T0_A1_official_weights": official_a1,
            "T0_A3_project_weights": project_a3,
            "T0_A3_official_weights": official_a3,
        }
        for method_id, signal in variants.items():
            method_rows = evaluate_variant(
                method_id,
                signal,
                data,
                active,
                train_end,
                val_end,
                seed,
            )
            rows.extend(method_rows)

        joint_path = (
            BASE
            / "e4_joint"
            / OUTPUT_TAG
            / f"T4_full_soft_seed_{seed}_scores.npz"
        )
        if joint_path.exists():
            joint_scores = np.load(joint_path, allow_pickle=False)
            sources.append(joint_path)
            euclidean_key = (
                "euclidean"
                if "euclidean" in joint_scores
                else "T4_bank_euclidean"
            )
            euc_z = joint_scores[euclidean_key]
            variants = {
                "T4_A1_project_weights": (
                    euc_z + 0.6 * a1_g_z + 0.4 * a1_q_z
                ),
                "T4_A1_official_weights": (
                    0.6 * euc_z + 0.4 * a1_g_z + 0.2 * a1_q_z
                ),
                "T4_A3_project_weights": (
                    euc_z + 0.6 * a3_g_z + 0.4 * a3_q_z
                ),
                "T4_A3_official_weights": (
                    0.6 * euc_z + 0.4 * a3_g_z + 0.2 * a3_q_z
                ),
            }
            for method_id, signal in variants.items():
                method_rows = evaluate_variant(
                    method_id,
                    signal,
                    data,
                    active,
                    train_end,
                    val_end,
                    seed,
                )
                rows.extend(method_rows)

    frame = pd.DataFrame(rows)
    frame["dataset"] = "r4.2"
    frame["protocol_id"] = "P3_e3_component_audit"
    frame["method_id"] = frame["method"]
    frame["source_file"] = "recomputed from E0/E1 scores and corrected components"
    return frame, sources


def rebuild_e2_results() -> tuple[pd.DataFrame, list[Path]]:
    root = BASE / "e2" / "results"
    paths = sorted(root.glob("*/seed_*/metrics.csv"))
    frames = []
    for path in paths:
        method = path.parents[1].name
        seed = int(path.parent.name.removeprefix("seed_"))
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frame["method_id"] = method + "_" + frame["method"].astype(str)
        frame["dataset"] = "r4.2"
        frame["protocol_id"] = "P2_e2_softclt_internal"
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), paths
    combined = pd.concat(frames, ignore_index=True)
    source_files = combined["source_file"].copy()
    combined = normalized_metrics(
        combined,
        dataset="r4.2",
        protocol_id="P2_e2_softclt_internal",
        method_id=None,
        source_file=root,
    )
    combined["source_file"] = source_files
    return combined, paths


def main() -> None:
    CANONICAL.mkdir(parents=True, exist_ok=True)
    e2_dir = CANONICAL / "e2"
    e2_dir.mkdir(parents=True, exist_ok=True)

    e2_metrics, e2_sources = rebuild_e2_results()
    e2_metrics.to_csv(e2_dir / "metrics_by_seed.csv", index=False)
    summarize_metrics(e2_metrics).to_csv(
        e2_dir / "metric_summary.csv",
        index=False,
    )
    e2_seed_matrix = (
        e2_metrics.groupby("method_id", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            seeds=(
                "seed",
                lambda values: ",".join(
                    str(value) for value in sorted(set(values))
                ),
            ),
        )
        .sort_values("method_id")
    )
    e2_seed_matrix.to_csv(e2_dir / "available_seed_matrix.csv", index=False)
    legacy_e2_files = [
        BASE / "e2" / "metrics_by_seed.csv",
        BASE / "e2" / "metric_summary.csv",
        BASE / "e2" / "run_summaries.json",
    ]
    (e2_dir / "top_level_overwrite_audit.json").write_text(
        json.dumps(
            {
                "issue": (
                    "The legacy e2 script rewrote top-level aggregate files "
                    "on every partial run."
                ),
                "resolution": (
                    "Canonical aggregates are rebuilt from every "
                    "results/<method>/seed_*/metrics.csv file. New runs write "
                    "to e2/aggregates/<run-tag>/."
                ),
                "per_seed_source_files": len(e2_sources),
                "methods": int(e2_seed_matrix["method_id"].nunique()),
                "legacy_top_level_files": [
                    {
                        "path": str(path),
                        "exists": path.exists(),
                        "bytes": path.stat().st_size if path.exists() else None,
                        "modified_utc": (
                            datetime.fromtimestamp(
                                path.stat().st_mtime,
                                timezone.utc,
                            ).isoformat()
                            if path.exists()
                            else None
                        ),
                        "sha256": sha256(path) if path.exists() else None,
                    }
                    for path in legacy_e2_files
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    component_metrics, component_sources = (
        build_component_and_weight_metrics()
    )
    component_metrics = normalized_metrics(
        component_metrics,
        dataset="r4.2",
        protocol_id="P3_e3_component_audit",
        method_id=None,
        source_file=CANONICAL / "pai_weight_sensitivity_by_seed.csv",
    )
    component_metrics.to_csv(
        CANONICAL / "pai_weight_sensitivity_by_seed.csv",
        index=False,
    )
    summarize_metrics(component_metrics).to_csv(
        CANONICAL / "pai_weight_sensitivity_summary.csv",
        index=False,
    )

    e4_r42 = pd.read_csv(
        BASE / "e4_joint" / OUTPUT_TAG / "metrics_by_seed.csv"
    )
    e4_r42["method_id"] = e4_r42["method"]
    e4_r42 = normalized_metrics(
        e4_r42,
        dataset="r4.2",
        protocol_id="P4_e4_joint_corrected",
        method_id=None,
        source_file=BASE / "e4_joint" / OUTPUT_TAG / "metrics_by_seed.csv",
    )

    e4_r52 = pd.read_csv(
        BASE / "e4_joint" / "r52" / OUTPUT_TAG / "metrics_by_seed.csv"
    )
    e4_r52["split"] = "test"
    e4_r52["method_id"] = e4_r52["method"]
    e4_r52 = normalized_metrics(
        e4_r52,
        dataset="r5.2",
        protocol_id="P4_e4_joint_corrected",
        method_id=None,
        source_file=(
            BASE / "e4_joint" / "r52" / OUTPUT_TAG / "metrics_by_seed.csv"
        ),
    )

    e4_r62 = pd.read_csv(
        BASE / "e4_joint" / "r62" / OUTPUT_TAG / "metrics_by_seed.csv"
    )
    e4_r62["split"] = "test"
    e4_r62["method_id"] = e4_r62["method"]
    e4_r62 = normalized_metrics(
        e4_r62,
        dataset="r6.2",
        protocol_id="P4_e4_joint_corrected",
        method_id=None,
        source_file=(
            BASE / "e4_joint" / "r62" / OUTPUT_TAG / "metrics_by_seed.csv"
        ),
    )

    e5_r52_path = BASE / "e5" / "r52" / "metrics_by_seed.csv"
    e5_r52 = pd.read_csv(e5_r52_path)
    e5_r52["split"] = "test"
    e5_r52["method_id"] = e5_r52["method"]
    e5_r52 = normalized_metrics(
        e5_r52,
        dataset="r5.2",
        protocol_id="P5_e5_r52_raw",
        method_id=None,
        source_file=e5_r52_path,
    )

    r62_t0_path = (
        BASE / "e4_joint" / "r62_matched_t0" / "metrics_by_seed.csv"
    )
    r62_t0 = pd.read_csv(r62_t0_path)
    r62_t0["split"] = "test"
    r62_t0["method_id"] = r62_t0["method"]
    r62_t0 = normalized_metrics(
        r62_t0,
        dataset="r6.2",
        protocol_id="P6_e4_r62_matched_t0",
        method_id=None,
        source_file=r62_t0_path,
    )

    e11_path = (
        BASE / "e11_iforest" / "metrics_by_seed.csv"
    )
    if e11_path.exists():
        e11 = pd.read_csv(e11_path)
        e11["method_id"] = e11["method"]
        e11 = normalized_metrics(
            e11,
            dataset="r4.2",
            protocol_id="P7_e11_iforest_27d",
            method_id=None,
            source_file=e11_path,
        )
    else:
        e11 = pd.DataFrame(columns=METRIC_COLUMNS)

    canonical_metrics = pd.concat(
        [
            e2_metrics,
            component_metrics,
            e4_r42,
            e4_r52,
            e4_r62,
            e5_r52,
            r62_t0,
            e11,
        ],
        ignore_index=True,
    )
    canonical_metrics.to_csv(
        CANONICAL / "canonical_metrics_by_seed.csv",
        index=False,
    )
    summarize_metrics(canonical_metrics).to_csv(
        CANONICAL / "canonical_metric_summary.csv",
        index=False,
    )

    e3_a3 = pd.read_csv(BASE / "e3" / "metrics_by_seed.csv")
    e3_a3 = e3_a3[
        e3_a3["method"] == "A3_user_channel_top2_pai"
    ].copy()
    e3_a3["method_id"] = "T0_A3_project_weights"
    canonical_a3 = canonical_metrics[
        (canonical_metrics["dataset"] == "r4.2")
        & (
            canonical_metrics["method_id"]
            == "T0_A3_project_weights"
        )
    ][
        [
            "seed",
            "split",
            "k",
            "pr_auc",
            "roc_auc",
            "incident_hits",
        ]
    ]
    reconciliation = e3_a3.merge(
        canonical_a3,
        on=["seed", "split", "k"],
        suffixes=("_e3_a3", "_recomputed_a3"),
        validate="one_to_one",
    )
    for metric in ("pr_auc", "roc_auc", "incident_hits"):
        reconciliation[f"{metric}_difference"] = (
            reconciliation[f"{metric}_recomputed_a3"]
            - reconciliation[f"{metric}_e3_a3"]
        )
    reconciliation[
        [
            "seed",
            "split",
            "k",
            "pr_auc_e3_a3",
            "pr_auc_recomputed_a3",
            "pr_auc_difference",
            "roc_auc_e3_a3",
            "roc_auc_recomputed_a3",
            "roc_auc_difference",
            "incident_hits_e3_a3",
            "incident_hits_recomputed_a3",
            "incident_hits_difference",
        ]
    ].to_csv(CANONICAL / "d2_a3_reconciliation.csv", index=False)

    protocol_map = build_protocol_map()
    protocol_map.to_csv(
        CANONICAL / "baseline_protocol_map.csv",
        index=False,
    )
    pai_map = build_pai_variant_map()
    pai_map.to_csv(CANONICAL / "pai_variant_mapping.csv", index=False)

    comparisons = []
    for split in ("validation", "test"):
        comparisons.extend(
            [
                paired_difference(
                    canonical_metrics,
                    dataset="r4.2",
                    split=split,
                    treatment_method="T4_full_soft_original",
                    control_method="T0_mask_original",
                    treatment_label="T4 original",
                    control_label="T0 original",
                ),
                paired_difference(
                    canonical_metrics,
                    dataset="r4.2",
                    split=split,
                    treatment_method="T4_full_soft_pai_a1",
                    control_method="T0_A1_project_weights",
                    treatment_label="T4 project A1 weights",
                    control_label="T0 project A1 weights",
                ),
                paired_difference(
                    canonical_metrics,
                    dataset="r4.2",
                    split=split,
                    treatment_method="T4_full_soft_euclidean",
                    control_method="T0_bank_euclidean",
                    treatment_label="T4 Euclidean bank",
                    control_label="T0 Euclidean bank",
                ),
                paired_difference(
                    canonical_metrics,
                    dataset="r4.2",
                    split=split,
                    treatment_method="T4_full_soft_cosine",
                    control_method="T0_bank_cosine",
                    treatment_label="T4 cosine bank",
                    control_label="T0 cosine bank",
                ),
            ]
        )
    comparisons.extend(
        [
            paired_difference(
                canonical_metrics,
                dataset="r5.2",
                split="test",
                treatment_method="T4_pai_a1",
                control_method="pai_a1_channel_mean",
                treatment_label="T4 project A1 weights",
                control_label="T0 project A1 weights",
            ),
            paired_difference(
                canonical_metrics,
                dataset="r5.2",
                split="test",
                treatment_method="T4_bank_euclidean",
                control_method="bank_euclidean_global",
                treatment_label="T4 Euclidean bank",
                control_label="T0 Euclidean bank",
            ),
            paired_difference(
                canonical_metrics,
                dataset="r5.2",
                split="test",
                treatment_method="T4_bank_cosine",
                control_method="bank_cosine_global",
                treatment_label="T4 cosine bank",
                control_label="T0 cosine bank",
            ),
            paired_difference(
                canonical_metrics,
                dataset="r6.2",
                split="test",
                treatment_method="T4_pai_a1",
                control_method="T0m_pai_a1",
                treatment_label="T4 project A1 weights",
                control_label="Matched T0 project A1 weights",
            ),
            paired_difference(
                canonical_metrics,
                dataset="r6.2",
                split="test",
                treatment_method="T4_bank_euclidean",
                control_method="T0m_bank_euclidean",
                treatment_label="T4 Euclidean bank",
                control_label="Matched T0 Euclidean bank",
            ),
            paired_difference(
                canonical_metrics,
                dataset="r6.2",
                split="test",
                treatment_method="T4_original",
                control_method="T0m_original",
                treatment_label="T4 original",
                control_label="Matched T0 original",
            ),
        ]
    )
    (CANONICAL / "joint_paired_comparisons.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_paths = sorted(
        set(
            [
                *e2_sources,
                *component_sources,
                BASE / "e3" / "metrics_by_seed.csv",
                BASE / "e4_joint" / OUTPUT_TAG / "metrics_by_seed.csv",
                BASE / "e4_joint" / "r52" / OUTPUT_TAG / "metrics_by_seed.csv",
                BASE / "e4_joint" / "r62" / OUTPUT_TAG / "metrics_by_seed.csv",
                e5_r52_path,
                r62_t0_path,
                e11_path,
            ]
        )
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_tag": OUTPUT_TAG,
        "policy": [
            "Original files are not overwritten.",
            "Test data are not used to select weights or thresholds.",
            "Missing timing sentinel -1 is treated as missing, not zero.",
            "Project PAI-A1 and E0/E1 S5 are separate variants.",
            "Official PAI scalar weights are a fixed sensitivity comparison.",
        ],
        "sources": [
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.is_file() else None,
            }
            for path in source_paths
        ],
    }
    (CANONICAL / "data_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"canonical rows: {len(canonical_metrics)}")
    print(f"e2 per-seed files: {len(e2_sources)}")
    print(f"output: {CANONICAL}")


if __name__ == "__main__":
    main()
