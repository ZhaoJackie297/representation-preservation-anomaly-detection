from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
CALIBRATED = PROCESSED / "r4.2_calibrated_method_scores.npz"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"

SELECTION_FILE = PROCESSED / "r4.2_fuse_then_dynamic_selection.csv"
RESULTS_FILE = PROCESSED / "r4.2_fuse_then_dynamic_results.json"
SUMMARY_FILE = PROCESSED / "r4.2_fuse_then_dynamic_summary.json"
HITS_FILE = PROCESSED / "r4.2_fuse_then_dynamic_hits.csv"
SCENARIO_FILE = PROCESSED / "r4.2_fuse_then_dynamic_scenarios.csv"

PRIMARY_K = 10
REPORT_K = [5, 10, 20]
COOLDOWN_DAYS = 7

sys.path.insert(0, str(ROOT))
from risk_topk_r42 import compute_risk  # noqa: E402
from production_protocol_r42 import (  # noqa: E402
    build_alerts,
    evaluate,
    label_matrix,
)
from conditional_combination_r42 import (  # noqa: E402
    daily_percentile,
    incident_hit_ids,
    scenario_macro,
)


def build_base_fusion(
    ts_base: np.ndarray,
    if_base: np.ndarray,
    operator: str,
) -> np.ndarray:
    if operator == "mean":
        return (ts_base + if_base) / 2.0
    if operator == "if70_ts30":
        return 0.70 * if_base + 0.30 * ts_base
    if operator == "if30_ts70":
        return 0.30 * if_base + 0.70 * ts_base
    if operator == "max":
        return np.maximum(ts_base, if_base)
    if operator == "min":
        return np.minimum(ts_base, if_base)
    raise ValueError(f"Unknown operator: {operator}")


def evaluate_signal(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    signal: np.ndarray,
    labels: np.ndarray,
    incidents: pd.DataFrame,
    train_end: int,
    val_end: int,
    method: str,
    k: int,
) -> tuple[dict, dict[str, set[str]]]:
    alerts = build_alerts(
        signal, active_mask, k, COOLDOWN_DAYS
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
        method,
        k,
    )
    bounds = {
        "validation": (train_end + 1, val_end),
        "test": (val_end + 1, len(dates) - 1),
    }
    hit_sets = {}
    for split, (start_index, end_index) in bounds.items():
        hit_sets[split] = incident_hit_ids(
            users,
            dates,
            alerts,
            incidents,
            start_index,
            end_index,
        )
        macro, scenario_rows = scenario_macro(
            hit_sets[split],
            incidents,
            dates,
            start_index,
            end_index,
        )
        result["splits"][split]["scenario_macro_recall_3d"] = macro
        result["splits"][split]["scenario_recalls"] = scenario_rows
    return result, hit_sets


def main() -> None:
    dataset = np.load(DATASET, allow_pickle=False)
    calibrated = np.load(CALIBRATED, allow_pickle=False)
    users = dataset["users"].astype(str)
    dates = pd.to_datetime(dataset["dates"].astype(str))
    active_mask = dataset["active_mask"][:, :, 0].astype(bool)
    train_end = int(dataset["train_end_index"])
    val_end = int(dataset["val_end_index"])
    labels = label_matrix(users, dates)
    incidents = pd.read_csv(INCIDENTS)
    incidents["start_date"] = pd.to_datetime(
        incidents["start"]
    ).dt.normalize()
    incidents["end_date"] = pd.to_datetime(
        incidents["end"]
    ).dt.normalize()

    ts_base = daily_percentile(
        calibrated["ts2vec_quantile"].astype(np.float32),
        active_mask,
    )
    if_base = daily_percentile(
        calibrated["iforest_quantile"].astype(np.float32),
        active_mask,
    )
    ts_single = calibrated["ts2vec_quantile_dynamic"].astype(
        np.float32
    )
    if_single = calibrated["iforest_quantile_dynamic"].astype(
        np.float32
    )

    base_fusions = {
        f"fuse_{operator}_then_dynamic": build_base_fusion(
            ts_base, if_base, operator
        )
        for operator in (
            "mean",
            "if70_ts30",
            "if30_ts70",
            "max",
            "min",
        )
    }
    signals = {
        "ts2vec_single": ts_single,
        "iforest_single": if_single,
        **{
            name: compute_risk(
                users, dates, fused_base, active_mask
            )
            for name, fused_base in base_fusions.items()
        },
    }

    validation_rows = []
    validation_hits: dict[str, set[str]] = {}
    for name, signal in signals.items():
        result, hit_sets = evaluate_signal(
            users,
            dates,
            active_mask,
            signal,
            labels,
            incidents,
            train_end,
            val_end,
            name,
            PRIMARY_K,
        )
        validation = result["splits"]["validation"]
        validation_hits[name] = hit_sets["validation"]
        validation_rows.append(
            {
                "method": name,
                "validation_recall_3d": validation["horizons"]["3"][
                    "event_recall"
                ],
                "validation_recall_1d": validation["horizons"]["1"][
                    "event_recall"
                ],
                "validation_precision": validation["precision_at_k"],
                "validation_false_per_day": validation[
                    "false_alerts_per_day"
                ],
                "validation_scenario_macro": validation[
                    "scenario_macro_recall_3d"
                ],
            }
        )

    selected_row = min(
        validation_rows,
        key=lambda row: (
            -row["validation_recall_3d"],
            -row["validation_recall_1d"],
            -row["validation_precision"],
            row["validation_false_per_day"],
            row["method"],
        ),
    )
    selected_name = selected_row["method"]
    pd.DataFrame(validation_rows).sort_values(
        [
            "validation_recall_3d",
            "validation_recall_1d",
            "validation_precision",
        ],
        ascending=False,
    ).to_csv(SELECTION_FILE, index=False, encoding="utf-8")

    report_methods = list(signals)
    results = []
    hit_rows = []
    scenario_rows = []
    for name in report_methods:
        for k in REPORT_K:
            result, hit_sets = evaluate_signal(
                users,
                dates,
                active_mask,
                signals[name],
                labels,
                incidents,
                train_end,
                val_end,
                name,
                k,
            )
            results.append(result)
            for split, ids in hit_sets.items():
                for incident_id in sorted(ids):
                    hit_rows.append(
                        {
                            "method": name,
                            "k": k,
                            "split": split,
                            "incident_id": incident_id,
                        }
                    )
                for scenario, values in result["splits"][split][
                    "scenario_recalls"
                ].items():
                    scenario_rows.append(
                        {
                            "method": name,
                            "k": k,
                            "split": split,
                            "scenario": scenario,
                            **values,
                        }
                    )

    pd.DataFrame(hit_rows).to_csv(
        HITS_FILE, index=False, encoding="utf-8"
    )
    pd.DataFrame(scenario_rows).to_csv(
        SCENARIO_FILE, index=False, encoding="utf-8"
    )
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    selected_validation_hits = validation_hits[selected_name]
    ts_validation_hits = validation_hits["ts2vec_single"]
    if_validation_hits = validation_hits["iforest_single"]
    summary = {
        "protocol": {
            "primary_k": PRIMARY_K,
            "report_k": REPORT_K,
            "cooldown_days": COOLDOWN_DAYS,
            "selection_split": "validation",
            "test_used_for_selection": False,
            "pipeline": (
                "Daily-percentile each base anomaly score, fuse the base "
                "scores, then apply one shared dynamic-risk aggregation."
            ),
            "operators": [
                "mean",
                "if70_ts30",
                "if30_ts70",
                "max",
                "min",
            ],
        },
        "selected_on_validation": selected_row,
        "validation_hit_sets": {
            "selected": sorted(selected_validation_hits),
            "added_over_ts2vec": sorted(
                selected_validation_hits - ts_validation_hits
            ),
            "lost_vs_ts2vec": sorted(
                ts_validation_hits - selected_validation_hits
            ),
            "added_over_iforest": sorted(
                selected_validation_hits - if_validation_hits
            ),
            "lost_vs_iforest": sorted(
                if_validation_hits - selected_validation_hits
            ),
        },
        "outputs": {
            "selection_file": str(SELECTION_FILE),
            "results_file": str(RESULTS_FILE),
            "hits_file": str(HITS_FILE),
            "scenario_file": str(SCENARIO_FILE),
        },
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(
        pd.DataFrame(validation_rows)
        .sort_values(
            [
                "validation_recall_3d",
                "validation_recall_1d",
                "validation_precision",
            ],
            ascending=False,
        )
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
