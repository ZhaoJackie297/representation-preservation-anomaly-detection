from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
ZSCORE_DAILY = PROCESSED / "r4.2_zscore_daily_scores.csv"
GROUP_SCORES = PROCESSED / "r4.2_group_baseline_daily_scores.csv"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"

SELECTION_FILE = PROCESSED / "r4.2_dual_percentile_selection.csv"
RESULTS_FILE = PROCESSED / "r4.2_dual_percentile_results.json"
SUMMARY_FILE = PROCESSED / "r4.2_dual_percentile_summary.json"

sys.path.insert(0, str(ROOT))
from fusion_r42 import daily_percentile, matrix_from_csv  # noqa: E402
from production_protocol_r42 import (  # noqa: E402
    COOLDOWN_DAYS,
    PRIMARY_K,
    REPORT_K,
    build_alerts,
    evaluate,
    label_matrix,
)
from risk_topk_r42 import compute_risk  # noqa: E402


def main() -> None:
    with np.load(DATASET, allow_pickle=False) as dataset:
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active_mask = dataset["active_mask"][:, :, 0].astype(bool)
        train_end = int(dataset["train_end_index"])
        val_end = int(dataset["val_end_index"])

    personal = matrix_from_csv(
        ZSCORE_DAILY,
        users,
        dates,
        "robust_z_score",
    )
    personal = compute_risk(
        users, dates, personal, active_mask
    )
    personal = daily_percentile(personal, active_mask)
    group = daily_percentile(
        matrix_from_csv(
            GROUP_SCORES,
            users,
            dates,
            "group_score",
        ),
        active_mask,
    )

    combined = {
        "percentile_mean": (personal + group) / 2.0,
        "percentile_max": np.maximum(personal, group),
    }
    variants = {}
    for name, score in combined.items():
        variants[f"{name}_raw"] = score.astype(np.float32)
        variants[f"{name}_dynamic"] = compute_risk(
            users, dates, score.astype(np.float32), active_mask
        )

    labels = label_matrix(users, dates)
    incidents = pd.read_csv(INCIDENTS)
    validation_rows = []
    selected = {}
    for family in combined:
        candidates = []
        for aggregation in ("raw", "dynamic"):
            variant = f"{family}_{aggregation}"
            alerts = build_alerts(
                variants[variant],
                active_mask,
                PRIMARY_K,
                COOLDOWN_DAYS,
            )
            result = evaluate(
                users,
                dates,
                active_mask,
                alerts,
                labels,
                incidents,
                train_end,
                val_end,
                variant,
                PRIMARY_K,
            )
            validation = result["splits"]["validation"]
            candidates.append(
                {
                    "family": family,
                    "variant": variant,
                    "validation_recall_3d": validation["horizons"][
                        "3"
                    ]["event_recall"],
                    "validation_recall_1d": validation["horizons"][
                        "1"
                    ]["event_recall"],
                    "validation_precision": validation[
                        "precision_at_k"
                    ],
                    "validation_false_per_day": validation[
                        "false_alerts_per_day"
                    ],
                }
            )
        candidates = sorted(
            candidates,
            key=lambda row: (
                -row["validation_recall_3d"],
                -row["validation_recall_1d"],
                -row["validation_precision"],
                row["validation_false_per_day"],
            ),
        )
        selected[family] = candidates[0]["variant"]
        validation_rows.extend(candidates)

    selection = pd.DataFrame(validation_rows)
    selection.to_csv(
        SELECTION_FILE, index=False, encoding="utf-8"
    )

    results = []
    for family, variant in selected.items():
        for k in REPORT_K:
            alerts = build_alerts(
                variants[variant],
                active_mask,
                k,
                COOLDOWN_DAYS,
            )
            result = evaluate(
                users,
                dates,
                active_mask,
                alerts,
                labels,
                incidents,
                train_end,
                val_end,
                variant,
                k,
            )
            result["family"] = family
            result["base_score"] = variant
            results.append(result)
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "purpose": (
            "Scale-corrected diagnostic for the raw-score dual baseline. "
            "Development test was already seen and is not a final holdout."
        ),
        "normalization": "daily active-user percentile",
        "selected_variants": selected,
        "selection_file": str(SELECTION_FILE),
        "results_file": str(RESULTS_FILE),
        "primary_k": PRIMARY_K,
        "cooldown_days": COOLDOWN_DAYS,
        "labels_used_for_signal": False,
        "labels_used_for_variant_selection": "validation only",
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(
        selection.sort_values(
            ["validation_recall_3d", "validation_recall_1d"],
            ascending=False,
        ).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
