from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r42"
DATASET = PROCESSED / "r4.2_sequence_dataset.npz"
CALIBRATED = PROCESSED / "r4.2_calibrated_method_scores.npz"
INCIDENTS = PROCESSED / "r4.2_incidents.csv"

SELECTION_FILE = PROCESSED / "r4.2_conditional_combination_selection.csv"
SELECTED_ALERTS_FILE = PROCESSED / "r4.2_conditional_combination_alerts.csv"
HITS_FILE = PROCESSED / "r4.2_conditional_combination_hits.csv"
SUMMARY_FILE = PROCESSED / "r4.2_conditional_combination_summary.json"
RESULTS_FILE = PROCESSED / "r4.2_conditional_combination_results.json"
SCENARIO_FILE = PROCESSED / "r4.2_conditional_combination_scenarios.csv"

PRIMARY_K = 10
COOLDOWN_DAYS = 7
REPORT_K = [5, 10, 20]
HORIZON_DAYS = 3
HIGH_TAIL_QUANTILE = 0.95
HIGH_TAIL_SCORE = -math.log(1.0 - HIGH_TAIL_QUANTILE)
PERSISTENCE_FRACTION = 2.0 / 7.0
DISAGREEMENT_MARGIN = 0.25
PERSONAL_CONFIRM_PERCENTILE = 0.90

sys.path.insert(0, str(ROOT))
from production_protocol_r42 import (  # noqa: E402
    build_alerts,
    evaluate,
    label_matrix,
)


def daily_percentile(
    scores: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    percentiles = np.full(scores.shape, np.nan, dtype=np.float32)
    for time_index in range(scores.shape[1]):
        active = np.flatnonzero(active_mask[:, time_index])
        if len(active) <= 1:
            percentiles[active, time_index] = 1.0
            continue
        values = scores[active, time_index]
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite score at time index {time_index}")
        ranks = rankdata(values, method="average")
        percentiles[active, time_index] = (
            (ranks - 1.0) / (len(active) - 1.0)
        )
    return percentiles


def persistence_fraction(
    raw_quantile: np.ndarray,
    active_mask: np.ndarray,
    user_index: int,
    time_index: int,
) -> float:
    start = max(0, time_index - COOLDOWN_DAYS + 1)
    active = active_mask[user_index, start : time_index + 1]
    values = raw_quantile[user_index, start : time_index + 1][active]
    if len(values) == 0:
        return 0.0
    return float(np.mean(values >= HIGH_TAIL_SCORE))


def build_rescue_alerts(
    primary_score: np.ndarray,
    secondary_score: np.ndarray,
    primary_percentile: np.ndarray,
    secondary_percentile: np.ndarray,
    secondary_quantile: np.ndarray,
    personal_percentile: np.ndarray,
    active_mask: np.ndarray,
    k: int,
    q: int,
    gate_name: str,
    start_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    alerts = np.zeros(primary_score.shape, dtype=np.uint8)
    rescue_flags = np.zeros(primary_score.shape, dtype=np.uint8)
    last_alert = np.full(primary_score.shape[0], -10_000, dtype=np.int32)

    if q < 0 or q > k:
        raise ValueError("q must satisfy 0 <= q <= k")
    if gate_name not in {"none", "disagreement", "persistence", "personal"}:
        raise ValueError(f"Unknown gate: {gate_name}")

    primary_slots = k - q
    for time_index in range(start_index, primary_score.shape[1]):
        available = active_mask[:, time_index] & (
            time_index - last_alert >= COOLDOWN_DAYS
        )
        candidate_indices = np.flatnonzero(available)
        if len(candidate_indices) == 0:
            continue

        primary_order = candidate_indices[
            np.lexsort(
                (
                    candidate_indices,
                    -primary_score[candidate_indices, time_index],
                )
            )
        ]
        selected: list[int] = []
        selected.extend(primary_order[:primary_slots].tolist())

        if q > 0:
            primary_top_k = set(primary_order[:k].tolist())
            secondary_order = candidate_indices[
                np.lexsort(
                    (
                        candidate_indices,
                        -secondary_score[candidate_indices, time_index],
                    )
                )
            ]
            rescue_count = 0
            for user_index in secondary_order:
                if user_index in selected or user_index in primary_top_k:
                    continue
                passes = True
                if gate_name == "disagreement":
                    passes = (
                        secondary_percentile[user_index, time_index]
                        - primary_percentile[user_index, time_index]
                        >= DISAGREEMENT_MARGIN
                    )
                elif gate_name == "persistence":
                    passes = (
                        persistence_fraction(
                            secondary_quantile,
                            active_mask,
                            int(user_index),
                            time_index,
                        )
                        >= PERSISTENCE_FRACTION
                    )
                elif gate_name == "personal":
                    passes = (
                        personal_percentile[user_index, time_index]
                        >= PERSONAL_CONFIRM_PERCENTILE
                    )
                if not passes:
                    continue
                selected.append(int(user_index))
                rescue_flags[user_index, time_index] = 1
                rescue_count += 1
                if rescue_count >= q:
                    break

        if len(selected) < k:
            for user_index in primary_order:
                if user_index not in selected:
                    selected.append(int(user_index))
                if len(selected) >= k:
                    break

        selected_array = np.asarray(selected[:k], dtype=np.int64)
        alerts[selected_array, time_index] = 1
        last_alert[selected_array] = time_index

    return alerts, rescue_flags


def split_bounds(
    dates: pd.DatetimeIndex,
    train_end: int,
    val_end: int,
) -> dict[str, tuple[int, int]]:
    return {
        "validation": (train_end + 1, val_end),
        "test": (val_end + 1, len(dates) - 1),
    }


def incident_hit_ids(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    alerts: np.ndarray,
    incidents: pd.DataFrame,
    start_index: int,
    end_index: int,
) -> set[str]:
    user_index = {user: index for index, user in enumerate(users)}
    date_index = {
        date.strftime("%Y-%m-%d"): index for index, date in enumerate(dates)
    }
    start_date = dates[start_index].normalize()
    end_date = dates[end_index].normalize()
    split_incidents = incidents[
        (incidents["start_date"] >= start_date)
        & (incidents["start_date"] <= end_date)
    ]
    hit_ids: set[str] = set()
    for incident in split_incidents.itertuples(index=False):
        accounts = [
            user_index[account]
            for account in incident.observed_accounts.split("|")
            if account in user_index
        ]
        if not accounts:
            continue
        candidate_dates = pd.date_range(
            incident.start_date - pd.Timedelta(days=COOLDOWN_DAYS - 1),
            min(
                incident.end_date,
                incident.start_date + pd.Timedelta(days=HORIZON_DAYS - 1),
            ),
            freq="D",
        )
        for date in candidate_dates:
            date_text = date.strftime("%Y-%m-%d")
            if date_text not in date_index:
                continue
            time_index = date_index[date_text]
            if alerts[np.asarray(accounts), time_index].any():
                hit_ids.add(str(incident.incident_id))
                break
    return hit_ids


def scenario_macro(
    hit_ids: set[str],
    incidents: pd.DataFrame,
    dates: pd.DatetimeIndex,
    start_index: int,
    end_index: int,
) -> tuple[float, dict[str, dict[str, float | int]]]:
    start_date = dates[start_index].normalize()
    end_date = dates[end_index].normalize()
    split = incidents[
        (incidents["start_date"] >= start_date)
        & (incidents["start_date"] <= end_date)
    ]
    rows: dict[str, dict[str, float | int]] = {}
    recalls = []
    for scenario, group in split.groupby("scenario", sort=True):
        total = int(len(group))
        hits = int(
            group["incident_id"].astype(str).isin(hit_ids).sum()
        )
        recall = hits / total if total else 0.0
        rows[str(scenario)] = {
            "hits": hits,
            "total": total,
            "recall": recall,
        }
        recalls.append(recall)
    macro = float(np.mean(recalls)) if recalls else 0.0
    return macro, rows


def selection_key(row: dict, gate_count: int) -> tuple:
    return (
        -float(row["validation_recall_3d"]),
        -float(row["validation_recall_1d"]),
        -float(row["validation_precision"]),
        float(row["validation_false_per_day"]),
        gate_count,
        str(row["variant"]),
    )


def evaluate_at_k(
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    alerts: np.ndarray,
    labels: np.ndarray,
    incidents: pd.DataFrame,
    train_end: int,
    val_end: int,
    method: str,
    k: int,
) -> tuple[dict, dict[str, set[str]]]:
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
    hit_sets = {}
    for split, (start_index, end_index) in split_bounds(
        dates, train_end, val_end
    ).items():
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


def make_alerts(
    spec: dict,
    signals: dict[str, np.ndarray],
    raw_quantiles: dict[str, np.ndarray],
    personal_percentile: np.ndarray,
    active_mask: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    kind = spec["kind"]
    if kind == "single":
        signal = signals[spec["method"]]
        return build_alerts(signal, active_mask, k, COOLDOWN_DAYS), None
    if kind == "mean":
        signal = (signals["ts2vec"] + signals["iforest"]) / 2.0
        return build_alerts(signal, active_mask, k, COOLDOWN_DAYS), None
    if kind == "rescue":
        primary = spec["primary"]
        secondary = spec["secondary"]
        return build_rescue_alerts(
            primary_score=signals[primary],
            secondary_score=signals[secondary],
            primary_percentile=signals[primary],
            secondary_percentile=signals[secondary],
            secondary_quantile=raw_quantiles[secondary],
            personal_percentile=personal_percentile,
            active_mask=active_mask,
            k=k,
            q=int(spec["q"]),
            gate_name=spec["gate"],
        )
    raise ValueError(f"Unknown method kind: {kind}")


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

    raw_quantiles = {
        "ts2vec": calibrated["ts2vec_quantile"].astype(np.float32),
        "iforest": calibrated["iforest_quantile"].astype(np.float32),
        "zscore": calibrated["zscore_quantile"].astype(np.float32),
    }
    signals = {
        "ts2vec": daily_percentile(
            calibrated["ts2vec_quantile_dynamic"].astype(np.float32),
            active_mask,
        ),
        "iforest": daily_percentile(
            calibrated["iforest_quantile_dynamic"].astype(np.float32),
            active_mask,
        ),
        "zscore": daily_percentile(
            calibrated["zscore_quantile_dynamic"].astype(np.float32),
            active_mask,
        ),
    }
    personal_percentile = signals["zscore"]

    baseline_specs = [
        {"variant": "ts2vec_single", "kind": "single", "method": "ts2vec"},
        {"variant": "iforest_single", "kind": "single", "method": "iforest"},
        {"variant": "zscore_single", "kind": "single", "method": "zscore"},
        {"variant": "equal_percentile_mean", "kind": "mean"},
    ]
    unconditional_specs = []
    for primary, secondary, prefix in (
        ("iforest", "ts2vec", "iforest_primary_ts2vec_rescue"),
        ("ts2vec", "iforest", "ts2vec_primary_iforest_rescue"),
    ):
        for q in (1, 2, 3):
            unconditional_specs.append(
                {
                    "variant": f"{prefix}_q{q}_nogate",
                    "kind": "rescue",
                    "primary": primary,
                    "secondary": secondary,
                    "q": q,
                    "gate": "none",
                }
            )

    validation_rows = []
    all_specs = baseline_specs + unconditional_specs
    for spec in all_specs:
        alerts, _ = make_alerts(
            spec,
            signals,
            raw_quantiles,
            personal_percentile,
            active_mask,
            PRIMARY_K,
        )
        result, _ = evaluate_at_k(
            users,
            dates,
            active_mask,
            alerts,
            labels,
            incidents,
            train_end,
            val_end,
            spec["variant"],
            PRIMARY_K,
        )
        validation = result["splits"]["validation"]
        validation_rows.append(
            {
                "variant": spec["variant"],
                "kind": spec["kind"],
                "primary": spec.get("primary", ""),
                "secondary": spec.get("secondary", ""),
                "q": spec.get("q", 0),
                "gate": spec.get("gate", ""),
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

    unconditional_frame = pd.DataFrame(
        [row for row in validation_rows if row["kind"] == "rescue"]
    )
    best_unconditional = min(
        unconditional_frame.to_dict(orient="records"),
        key=lambda row: selection_key(row, 0),
    )
    best_unconditional_spec = next(
        spec
        for spec in unconditional_specs
        if spec["variant"] == best_unconditional["variant"]
    )

    gated_specs = []
    for gate in ("disagreement", "persistence", "personal"):
        gated_specs.append(
            {
                **best_unconditional_spec,
                "variant": (
                    f"{best_unconditional_spec['variant'].replace('_nogate', '')}"
                    f"_{gate}"
                ),
                "gate": gate,
            }
        )

    for spec in gated_specs:
        alerts, _ = make_alerts(
            spec,
            signals,
            raw_quantiles,
            personal_percentile,
            active_mask,
            PRIMARY_K,
        )
        result, _ = evaluate_at_k(
            users,
            dates,
            active_mask,
            alerts,
            labels,
            incidents,
            train_end,
            val_end,
            spec["variant"],
            PRIMARY_K,
        )
        validation = result["splits"]["validation"]
        validation_rows.append(
            {
                "variant": spec["variant"],
                "kind": spec["kind"],
                "primary": spec.get("primary", ""),
                "secondary": spec.get("secondary", ""),
                "q": spec.get("q", 0),
                "gate": spec.get("gate", ""),
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

    all_specs = baseline_specs + unconditional_specs + gated_specs
    selected_row = min(
        validation_rows,
        key=lambda row: selection_key(
            row, 0 if row["kind"] != "rescue" or row["gate"] == "none" else 1
        ),
    )
    selected_spec = next(
        spec for spec in all_specs if spec["variant"] == selected_row["variant"]
    )
    best_gated_row = min(
        [
            row
            for row in validation_rows
            if row["kind"] == "rescue" and row["gate"] != "none"
        ],
        key=lambda row: selection_key(row, 1),
    )
    best_gated_spec = next(
        spec for spec in gated_specs
        if spec["variant"] == best_gated_row["variant"]
    )
    opposite_direction_spec = next(
        spec
        for spec in unconditional_specs
        if spec["variant"]
        == "iforest_primary_ts2vec_rescue_q1_nogate"
    )
    pd.DataFrame(validation_rows).sort_values(
        ["validation_recall_3d", "validation_recall_1d", "validation_precision"],
        ascending=False,
    ).to_csv(SELECTION_FILE, index=False, encoding="utf-8")

    comparison_specs = list(baseline_specs)
    for finalist in (
        selected_spec,
        best_unconditional_spec,
        best_gated_spec,
        opposite_direction_spec,
    ):
        if all(
            spec["variant"] != finalist["variant"]
            for spec in comparison_specs
        ):
            comparison_specs.append(finalist)
    results = []
    hit_rows = []
    alert_rows = []
    scenario_rows = []
    hit_sets_by_method: dict[tuple[str, int], dict[str, set[str]]] = {}
    for spec in comparison_specs:
        for k in REPORT_K:
            alerts, rescue_flags = make_alerts(
                spec,
                signals,
                raw_quantiles,
                personal_percentile,
                active_mask,
                k,
            )
            result, hit_sets = evaluate_at_k(
                users,
                dates,
                active_mask,
                alerts,
                labels,
                incidents,
                train_end,
                val_end,
                spec["variant"],
                k,
            )
            result["spec"] = spec
            result["rescue_slots"] = (
                int(rescue_flags.sum()) if rescue_flags is not None else 0
            )
            results.append(result)
            hit_sets_by_method[(spec["variant"], k)] = hit_sets
            for split, ids in hit_sets.items():
                for incident_id in sorted(ids):
                    hit_rows.append(
                        {
                            "variant": spec["variant"],
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
                            "variant": spec["variant"],
                            "k": k,
                            "split": split,
                            "scenario": scenario,
                            **values,
                        }
                    )
            user_indices, time_indices = np.nonzero(alerts)
            for user_index, time_index in zip(user_indices, time_indices):
                alert_rows.append(
                    {
                        "variant": spec["variant"],
                        "k": k,
                        "date": dates[time_index].strftime("%Y-%m-%d"),
                        "user_id": users[user_index],
                        "rescued": (
                            int(rescue_flags[user_index, time_index])
                            if rescue_flags is not None
                            else 0
                        ),
                    }
                )

    pd.DataFrame(hit_rows).to_csv(HITS_FILE, index=False, encoding="utf-8")
    pd.DataFrame(alert_rows).to_csv(
        SELECTED_ALERTS_FILE, index=False, encoding="utf-8"
    )
    pd.DataFrame(scenario_rows).to_csv(
        SCENARIO_FILE, index=False, encoding="utf-8"
    )
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ts_hits_val = hit_sets_by_method[("ts2vec_single", PRIMARY_K)]["validation"]
    if_hits_val = hit_sets_by_method[("iforest_single", PRIMARY_K)]["validation"]
    selected_hits_val = hit_sets_by_method[
        (selected_spec["variant"], PRIMARY_K)
    ]["validation"]
    ts_hits_test = hit_sets_by_method[("ts2vec_single", PRIMARY_K)]["test"]
    if_hits_test = hit_sets_by_method[("iforest_single", PRIMARY_K)]["test"]
    selected_hits_test = hit_sets_by_method[
        (selected_spec["variant"], PRIMARY_K)
    ]["test"]

    summary = {
        "protocol": {
            "primary_k": PRIMARY_K,
            "cooldown_days": COOLDOWN_DAYS,
            "report_k": REPORT_K,
            "horizon_days": HORIZON_DAYS,
            "selection_split": "validation",
            "test_used_for_selection": False,
            "score_inputs": {
                "ts2vec": "ts2vec_quantile_dynamic -> daily percentile",
                "iforest": "iforest_quantile_dynamic -> daily percentile",
                "personal_confirm": "zscore_quantile_dynamic -> daily percentile",
            },
            "gate_rules": {
                "disagreement": (
                    "secondary daily percentile - primary daily percentile "
                    f">= {DISAGREEMENT_MARGIN}"
                ),
                "persistence": (
                    "at least 2 of last 7 active days have secondary calibrated "
                    f"quantile score >= {HIGH_TAIL_SCORE:.6f}"
                ),
                "personal": (
                    "personal daily percentile "
                    f">= {PERSONAL_CONFIRM_PERCENTILE}"
                ),
            },
        },
        "best_unconditional": best_unconditional,
        "selected_on_validation": selected_row,
        "selected_spec": selected_spec,
        "best_gated_finalist": best_gated_row,
        "best_gated_finalist_spec": best_gated_spec,
        "opposite_direction_finalist_spec": opposite_direction_spec,
        "test_reporting_note": (
            "The validation-selected method and the best unconditional/gated "
            "finalists are reported on the development test for transparency. "
            "The test is not used to change selection."
        ),
        "validation_hit_sets": {
            "ts2vec": sorted(ts_hits_val),
            "iforest": sorted(if_hits_val),
            "selected": sorted(selected_hits_val),
            "selected_added_over_ts2vec": sorted(selected_hits_val - ts_hits_val),
            "selected_lost_vs_ts2vec": sorted(ts_hits_val - selected_hits_val),
            "selected_added_over_iforest": sorted(selected_hits_val - if_hits_val),
            "selected_lost_vs_iforest": sorted(if_hits_val - selected_hits_val),
        },
        "test_hit_sets": {
            "ts2vec": sorted(ts_hits_test),
            "iforest": sorted(if_hits_test),
            "selected": sorted(selected_hits_test),
            "selected_added_over_ts2vec": sorted(selected_hits_test - ts_hits_test),
            "selected_lost_vs_ts2vec": sorted(ts_hits_test - selected_hits_test),
            "selected_added_over_iforest": sorted(selected_hits_test - if_hits_test),
            "selected_lost_vs_iforest": sorted(if_hits_test - selected_hits_test),
        },
        "finalist_hit_sets": {
            spec["variant"]: {
                split: sorted(
                    hit_sets_by_method[(spec["variant"], PRIMARY_K)][split]
                )
                for split in ("validation", "test")
            }
            for spec in (
                selected_spec,
                best_unconditional_spec,
                best_gated_spec,
                opposite_direction_spec,
            )
        },
        "outputs": {
            "selection_file": str(SELECTION_FILE),
            "alerts_file": str(SELECTED_ALERTS_FILE),
            "hits_file": str(HITS_FILE),
            "scenario_file": str(SCENARIO_FILE),
            "results_file": str(RESULTS_FILE),
        },
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
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
