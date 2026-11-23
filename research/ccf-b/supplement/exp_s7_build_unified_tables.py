"""Build the final Exp S7 CERT r4.2 10-seed summary and paired tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_rel


REPO_ROOT = Path(__file__).resolve().parents[3]
S7_ROOT = REPO_ROOT / "research" / "ccf-b" / "supplement_results" / "exp_s7_10seed_unified"
SEEDS = list(range(20261116, 20261126))
OVERLAP_SEEDS = SEEDS[:3]
EPS = 1e-12

DEEP_BY_SEED = S7_ROOT / "deep_baselines" / "cert_deep_baselines_by_seed.csv"
T4_BY_SEED = S7_ROOT / "t4_10seed_by_seed.csv"
AE_R42_BY_SEED = S7_ROOT / "ae_r4.2" / "ae_baselines_by_seed.csv"
AE_R52_BY_SEED = S7_ROOT / "ae_r5.2" / "ae_baselines_by_seed.csv"
T3_REFERENCE_BY_SEED = (
    REPO_ROOT
    / "research"
    / "ccf-b"
    / "supplement_results"
    / "exp_s1_r4.2"
    / "metrics_by_seed.csv"
)
OLD_AE_R42_BY_SEED = (
    REPO_ROOT
    / "research"
    / "ccf-b"
    / "supplement_results"
    / "exp_s4_baselines_hp"
    / "ae_r4.2"
    / "ae_baselines_by_seed.csv"
)
OLD_AE_R52_BY_SEED = (
    REPO_ROOT
    / "research"
    / "ccf-b"
    / "supplement_results"
    / "exp_s4_baselines_hp"
    / "ae_r5.2"
    / "ae_baselines_by_seed.csv"
)
OLD_T4_BY_SEED = (
    REPO_ROOT
    / "runtime"
    / "cert"
    / "experiments"
    / "representation_preservation"
    / "e4_joint"
    / "paired_pilot_20261201_03_v1"
    / "metrics_by_seed.csv"
)

T3_METHODS = {
    "mask": "SoftInst-Rand-10ep__plain_l1",
    "cosine": "SoftInst-Rand-10ep__bank_cosine",
    "euclidean": "SoftInst-Rand-10ep__bank_euclidean",
    "pai_mean": "SoftInst-Rand-10ep__pai_a1",
}
T4_METHODS = {
    "mask": "T4_full_soft__mask_original",
    "cosine": "T4_full_soft__bank_cosine",
    "euclidean": "T4_full_soft__bank_euclidean",
    "pai_mean": "T4_full_soft__pai_a1",
}
SCORER_LABELS = {
    "mask": "Mask-L1",
    "cosine": "Bank-Cosine",
    "euclidean": "Bank-Euclidean",
    "pai_mean": "PAI-Mean",
    "native": "Native",
}
METRICS = [
    ("pr_auc", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("event_recall_k5", "Event Recall@K=5"),
    ("event_recall_k10", "Event Recall@K=10"),
    ("event_recall_k20", "Event Recall@K=20"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=S7_ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def assert_seed_coverage(frame: pd.DataFrame, label: str) -> None:
    observed = {int(value) for value in frame["seed"].dropna().unique()}
    missing = set(SEEDS) - observed
    if missing:
        raise ValueError(f"{label}: missing seeds {sorted(missing)}")


def standard_frame(frame: pd.DataFrame, method: str, label: str) -> pd.DataFrame:
    subset = frame[
        (frame["method"] == method)
        & (frame["split"] == "test")
    ].copy()
    assert_seed_coverage(subset, label)
    subset = subset[subset["seed"].isin(SEEDS)].copy()

    points = (
        subset.groupby("seed", as_index=False)
        .agg(
            pr_auc=("pr_auc", "first"),
            roc_auc=("roc_auc", "first"),
        )
    )
    if len(points) != len(SEEDS):
        raise ValueError(f"{label}: expected one point row per seed")
    recalls = subset.pivot(index="seed", columns="k", values="event_recall_3d")
    missing_k = {5, 10, 20} - set(recalls.columns)
    if missing_k:
        raise ValueError(f"{label}: missing recall K values {sorted(missing_k)}")
    recalls = recalls[[5, 10, 20]].rename(
        columns={
            5: "event_recall_k5",
            10: "event_recall_k10",
            20: "event_recall_k20",
        }
    )
    result = points.set_index("seed").join(recalls)
    result = result.reindex(SEEDS)
    if result.isna().any().any():
        raise ValueError(f"{label}: incomplete metrics after seed alignment")
    return result


def deep_frame(frame: pd.DataFrame, method: str, label: str) -> pd.DataFrame:
    subset = frame[frame["method"] == method].copy()
    assert_seed_coverage(subset, label)
    subset = subset[subset["seed"].isin(SEEDS)].copy()
    if subset["seed"].duplicated().any():
        raise ValueError(f"{label}: duplicate deep-baseline seed rows")
    result = (
        subset.set_index("seed")[
            [
                "pr_auc",
                "roc_auc",
                "event_recall_k5",
                "event_recall_k10",
                "event_recall_k20",
            ]
        ]
        .reindex(SEEDS)
    )
    if result.isna().any().any():
        raise ValueError(f"{label}: incomplete deep-baseline metrics")
    return result


def mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.6f} +/- {values.std(ddof=1):.6f}"


def build_specs() -> list[dict]:
    specs = [
        {
            "key": "anomaly_transformer",
            "method": "Anomaly Transformer-adapted",
            "scorer": "Native",
            "source": "deep",
            "source_method": "Anomaly Transformer",
            "kind": "external",
            "order": 0,
        },
        {
            "key": "tnnclr",
            "method": "t-NNCLR + memory bank",
            "scorer": "Native",
            "source": "deep",
            "source_method": "t-NNCLR",
            "kind": "external",
            "order": 1,
        },
        {
            "key": "mlp_ae",
            "method": "MLP-AE-27D",
            "scorer": "Native",
            "source": "ae42",
            "source_method": "MLP-AE-27D",
            "kind": "external",
            "order": 2,
        },
        {
            "key": "lstm_pred",
            "method": "LSTM-Pred-27D",
            "scorer": "Native",
            "source": "ae42",
            "source_method": "LSTM-Pred-27D",
            "kind": "external",
            "order": 3,
        },
    ]
    next_order = 4
    for scorer in ("mask", "euclidean", "cosine", "pai_mean"):
        specs.append(
            {
                "key": f"t4_{scorer}",
                "method": "T4 (SoftCLT)",
                "scorer": SCORER_LABELS[scorer],
                "source": "t4",
                "source_method": T4_METHODS[scorer],
                "kind": "candidate",
                "order": next_order,
            }
        )
        next_order += 1
    for scorer in ("mask", "euclidean", "cosine", "pai_mean"):
        specs.append(
            {
                "key": f"t3_{scorer}",
                "method": "SoftInst-Rand (T3)",
                "scorer": SCORER_LABELS[scorer],
                "source": "t3",
                "source_method": T3_METHODS[scorer],
                "kind": "reference",
                "order": next_order,
            }
        )
        next_order += 1
    return specs


def build_frames() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    source_frames = {
        "deep": read_csv(DEEP_BY_SEED),
        "t4": read_csv(T4_BY_SEED),
        "ae42": read_csv(AE_R42_BY_SEED),
        "t3": read_csv(T3_REFERENCE_BY_SEED),
    }
    specs = build_specs()
    frames: dict[str, pd.DataFrame] = {}
    for spec in specs:
        raw = source_frames[spec["source"]]
        if spec["source"] == "deep":
            frames[spec["key"]] = deep_frame(
                raw,
                spec["source_method"],
                spec["key"],
            )
        else:
            frames[spec["key"]] = standard_frame(
                raw,
                spec["source_method"],
                spec["key"],
            )
    return source_frames, frames


def build_summary(specs: list[dict], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in specs:
        frame = frames[spec["key"]]
        row = {
            "dataset": "CERT r4.2",
            "split": "test",
            "method": spec["method"],
            "scorer": spec["scorer"],
            "n_seeds": len(SEEDS),
            "seeds": "20261116-20261125",
            "_order": spec["order"],
        }
        for metric, _ in METRICS:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
            row[f"{metric}_mean_sd"] = mean_sd(frame[metric])
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    if count == 0:
        return []
    adjusted = [math.nan] * count
    ranked = sorted(
        (
            (value, index)
            for index, value in enumerate(p_values)
            if math.isfinite(value)
        ),
        key=lambda item: item[0],
    )
    running = 0.0
    for rank, (value, index) in enumerate(ranked):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_statistics(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    metric: str,
) -> dict:
    aligned = pd.concat(
        [
            candidate[metric].rename("candidate"),
            reference[metric].rename("reference"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if list(aligned.index) != SEEDS:
        raise ValueError(
            f"Paired metric {metric} has seed set {list(aligned.index)}, "
            f"expected {SEEDS}"
        )
    delta = aligned["candidate"] - aligned["reference"]
    n = len(delta)
    mean = float(delta.mean())
    std = float(delta.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=n - 1))
    half_width = critical * std / math.sqrt(n)
    wins = int((delta > EPS).sum())
    losses = int((delta < -EPS).sum())
    ties = n - wins - losses
    note = ""

    if np.all(np.abs(delta.to_numpy()) <= EPS):
        p_value = 1.0
        note = "all_paired_differences_zero"
    elif std <= EPS:
        p_value = 0.0
        note = "constant_nonzero_delta_due_to_zero_sample_variance"
    else:
        p_value = float(ttest_rel(aligned["candidate"], aligned["reference"]).pvalue)

    return {
        "n_pairs": n,
        "delta_mean": mean,
        "delta_std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "positive_wins": wins,
        "losses": losses,
        "ties": ties,
        "wins_text": f"{wins}/{n}",
        "p_value": p_value,
        "test_note": note,
    }


def build_paired_tests(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    comparisons = [
        ("Anomaly Transformer-adapted", "anomaly_transformer", "Native"),
        ("t-NNCLR + memory bank", "tnnclr", "Native"),
        ("MLP-AE-27D", "mlp_ae", "Native"),
        ("LSTM-Pred-27D", "lstm_pred", "Native"),
        ("T4 (SoftCLT)", "t4_mask", "Mask-L1"),
        ("T4 (SoftCLT)", "t4_euclidean", "Bank-Euclidean"),
        ("T4 (SoftCLT)", "t4_cosine", "Bank-Cosine"),
        ("T4 (SoftCLT)", "t4_pai_mean", "PAI-Mean"),
    ]
    t3_key_by_scorer = {
        "Native": "t3_pai_mean",
        "Mask-L1": "t3_mask",
        "Bank-Euclidean": "t3_euclidean",
        "Bank-Cosine": "t3_cosine",
        "PAI-Mean": "t3_pai_mean",
    }
    rows: list[dict] = []
    for metric_index, (metric, metric_label) in enumerate(METRICS):
        metric_rows: list[dict] = []
        for candidate_order, (method, key, scorer) in enumerate(comparisons):
            ref_key = t3_key_by_scorer[scorer]
            stats = paired_statistics(frames[key], frames[ref_key], metric)
            metric_rows.append(
                {
                    "metric_order": metric_index,
                    "comparison_order": candidate_order,
                    "dataset": "CERT r4.2",
                    "split": "test",
                    "metric": metric_label,
                    "candidate": method,
                    "candidate_scorer": scorer,
                    "reference": "SoftInst-Rand (T3)",
                    "reference_scorer": "PAI-Mean"
                    if scorer == "Native"
                    else scorer,
                    "comparison_design": "external_vs_T3_PAI-Mean"
                    if scorer == "Native"
                    else "T4_vs_T3_same_scorer",
                    "delta_definition": "candidate - reference",
                    **stats,
                    "holm_family": (
                        f"CERT r4.2 test / {metric_label} / "
                        "all 8 planned S7 comparisons"
                    ),
                }
            )
        adjusted = holm_adjust([row["p_value"] for row in metric_rows])
        for row, adjusted_value in zip(metric_rows, adjusted):
            row["p_holm"] = adjusted_value
        rows.extend(metric_rows)
    result = pd.DataFrame(rows).sort_values(["metric_order", "comparison_order"])
    return result.drop(columns=["metric_order", "comparison_order"])


def compare_exact_rows(
    older: pd.DataFrame,
    newer: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
    seeds: list[int],
    label: str,
) -> dict:
    left = older[older["seed"].isin(seeds)].copy()
    right = newer[newer["seed"].isin(seeds)].copy()
    merged = left[keys + metrics].merge(
        right[keys + metrics],
        on=keys,
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(f"{label}: no overlapping rows")
    max_abs: dict[str, float] = {}
    mean_abs: dict[str, float] = {}
    for metric in metrics:
        delta = (
            merged[f"{metric}_old"] - merged[f"{metric}_new"]
        ).abs()
        max_abs[metric] = float(delta.max())
        mean_abs[metric] = float(delta.mean())
    return {
        "status": "compared",
        "overlap_seeds": seeds,
        "rows_compared": len(merged),
        "exact_within_1e-12": max(max_abs.values()) <= EPS,
        "max_abs_delta": max_abs,
        "mean_abs_delta": mean_abs,
    }


def build_reproduction_checks(
    source_frames: dict[str, pd.DataFrame],
) -> dict:
    t3_existing = source_frames["t3"]
    t4_new = source_frames["t4"]
    reverse_t3_methods = {value: key for key, value in T3_METHODS.items()}
    reverse_t4_methods = {value: key for key, value in T4_METHODS.items()}

    old_t3 = t3_existing[
        t3_existing["method"].isin(T3_METHODS.values())
        & t3_existing["seed"].isin(OVERLAP_SEEDS)
    ].copy()
    old_t3["method"] = old_t3["method"].map(
        {
            T3_METHODS[scorer]: T4_METHODS[scorer]
            for scorer in T3_METHODS
        }
    )
    new_t3 = t4_new[
        t4_new["method"].isin(T4_METHODS.values())
        & t4_new["seed"].isin(OVERLAP_SEEDS)
    ].copy()
    common_keys = ["seed", "split", "k"]
    t3_repro = compare_exact_rows(
        old_t3,
        new_t3,
        common_keys + ["method"],
        ["pr_auc", "roc_auc", "event_recall_3d"],
        OVERLAP_SEEDS,
        "T3 retraining",
    )
    t3_repro["test_only"] = compare_exact_rows(
        old_t3[old_t3["split"] == "test"],
        new_t3[new_t3["split"] == "test"],
        common_keys + ["method"],
        ["pr_auc", "roc_auc", "event_recall_3d"],
        OVERLAP_SEEDS,
        "T3 retraining test-only",
    )
    t3_repro["sources"] = {
        "existing_reference": str(T3_REFERENCE_BY_SEED),
        "retrained_unified": str(T4_BY_SEED),
        "retrained_methods": "T3_instance_soft__*",
    }
    t3_repro["note"] = (
        "The unified e2 trainer is not bitwise identical to the historical S1 "
        "batch-table training implementation; preserve the measured differences."
    )

    ae_checks: dict[str, dict] = {}
    for dataset, old_path, new_path, frame_key in (
        ("r4.2", OLD_AE_R42_BY_SEED, AE_R42_BY_SEED, "ae42"),
        ("r5.2", OLD_AE_R52_BY_SEED, AE_R52_BY_SEED, "ae52"),
    ):
        old = pd.read_csv(old_path)
        new = pd.read_csv(new_path)
        if dataset == "r4.2":
            new = source_frames[frame_key]
        ae_checks[dataset] = compare_exact_rows(
            old,
            new,
            ["seed", "method", "split", "k"],
            ["pr_auc", "roc_auc", "event_recall_3d"],
            OVERLAP_SEEDS,
            f"AE {dataset}",
        )
        ae_checks[dataset]["old_path"] = str(old_path)
        ae_checks[dataset]["new_path"] = str(new_path)
        ae_checks[dataset]["test_only"] = compare_exact_rows(
            old[old["split"] == "test"],
            new[new["split"] == "test"],
            ["seed", "method", "split", "k"],
            ["pr_auc", "roc_auc", "event_recall_3d"],
            OVERLAP_SEEDS,
            f"AE {dataset} test-only",
        )

    old_t4 = pd.read_csv(OLD_T4_BY_SEED)
    old_t4_seeds = sorted(int(value) for value in old_t4["seed"].unique())
    t4_overlap = sorted(set(old_t4_seeds) & set(OVERLAP_SEEDS))

    return {
        "generated_with_seeds": SEEDS,
        "t3_retraining_first_three": t3_repro,
        "old_s4_ae_first_three": ae_checks,
        "old_t4_archive": {
            "status": "not_comparable",
            "path": str(OLD_T4_BY_SEED),
            "historical_seeds": old_t4_seeds,
            "requested_overlap_seeds": OVERLAP_SEEDS,
            "overlap_seeds": t4_overlap,
            "note": (
                "The historical T4 archive contains 20261201-20261203, "
                "so it has no seed overlap with the locked S7 seeds."
            ),
        },
        "old_deep_baseline_archives": {
            "status": "not_comparable_on_first_three",
            "historical_seeds": [20261201, 20261202, 20261203],
            "requested_overlap_seeds": OVERLAP_SEEDS,
            "overlap_seeds": [],
            "note": (
                "The historical Anomaly Transformer and t-NNCLR archives "
                "use 20261201-20261203; exact first-three reproduction against "
                "the S7 seeds is impossible."
            ),
        },
    }


def build_r52_ae_summary() -> pd.DataFrame:
    frame = read_csv(AE_R52_BY_SEED)
    subset = frame[(frame["split"] == "test") & (frame["seed"].isin(SEEDS))]
    rows: list[dict] = []
    for method in ("MLP-AE-27D", "LSTM-Pred-27D"):
        method_frame = standard_frame(subset, method, f"r5.2 {method}")
        row = {
            "dataset": "CERT r5.2",
            "split": "test",
            "method": method,
            "n_seeds": len(SEEDS),
            "seeds": "20261116-20261125",
        }
        for metric, _ in METRICS:
            row[f"{metric}_mean"] = float(method_frame[metric].mean())
            row[f"{metric}_std"] = float(method_frame[metric].std(ddof=1))
            row[f"{metric}_mean_sd"] = mean_sd(method_frame[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs()
    source_frames, frames = build_frames()

    summary = build_summary(specs, frames)
    paired = build_paired_tests(frames)
    reproduction = build_reproduction_checks(source_frames)
    r52_summary = build_r52_ae_summary()

    summary.to_csv(
        output_dir / "table3_10seed_summary.csv",
        index=False,
        encoding="utf-8",
    )
    paired.to_csv(
        output_dir / "table3_10seed_paired_tests.csv",
        index=False,
        encoding="utf-8",
    )
    (output_dir / "reproduction_checks.json").write_text(
        json.dumps(reproduction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    r52_summary.to_csv(
        output_dir / "r5.2_ae_lstm_10seed_summary.csv",
        index=False,
        encoding="utf-8",
    )
    print("Wrote table3_10seed_summary.csv")
    print("Wrote table3_10seed_paired_tests.csv")
    print("Wrote reproduction_checks.json")
    print("Wrote r5.2_ae_lstm_10seed_summary.csv")
    print(summary[["method", "scorer", "pr_auc_mean_sd", "roc_auc_mean_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
