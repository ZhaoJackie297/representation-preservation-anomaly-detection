from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed_r52"
INCIDENTS = PROCESSED / "r5.2_incidents.csv"
HITS = PROCESSED / "r5.2_external_hits.csv"
RESULTS = PROCESSED / "r5.2_external_results.json"
STATISTICS = PROCESSED / "r5.2_external_statistics.json"

PRIMARY_K = 10
PRIMARY_HORIZON = 3
Z_95 = 1.959963984540054
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20261110


def wilson_interval(successes: int, total: int) -> dict:
    if total == 0:
        return {"lower": 0.0, "upper": 0.0}
    proportion = successes / total
    denominator = 1.0 + Z_95**2 / total
    center = (
        proportion + Z_95**2 / (2 * total)
    ) / denominator
    margin = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + Z_95**2 / (4 * total**2)
        )
        / denominator
    )
    return {
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def mcnemar_exact(ts_only: int, other_only: int) -> dict:
    discordant = ts_only + other_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = binom.cdf(
            min(ts_only, other_only),
            discordant,
            0.5,
        )
        p_value = min(1.0, 2.0 * float(tail))
    return {
        "ts2vec_only_hits": ts_only,
        "other_method_only_hits": other_only,
        "discordant_events": discordant,
        "exact_two_sided_p": p_value,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted = {}
    running_max = 0.0
    for rank, (method, value) in enumerate(ordered):
        candidate = min(1.0, value * (count - rank))
        running_max = max(running_max, candidate)
        adjusted[method] = running_max
    return adjusted


def main() -> None:
    incidents = pd.read_csv(
        INCIDENTS,
        parse_dates=[
            "first_observed_date",
            "last_observed_date",
        ],
    )
    all_incident_ids = set(incidents["incident_id"])
    hits = pd.read_csv(HITS)
    primary = hits[
        (hits["k"] == PRIMARY_K)
        & (hits["horizon_days"] == PRIMARY_HORIZON)
    ]
    methods = sorted(primary["method"].unique())
    if "ts2vec_personal_quantile_dynamic" not in methods:
        raise RuntimeError("Missing TS2Vec primary result.")

    hit_sets = {
        method: set(
            primary.loc[primary["method"] == method, "incident_id"]
        )
        for method in methods
    }
    ts_set = hit_sets["ts2vec_personal_quantile_dynamic"]
    method_rows = []
    for method in methods:
        selected = hit_sets[method]
        ci = wilson_interval(len(selected), len(all_incident_ids))
        method_rows.append(
            {
                "method": method,
                "hits": len(selected),
                "events": len(all_incident_ids),
                "event_recall": len(selected)
                / len(all_incident_ids),
                "recall_ci95_lower": ci["lower"],
                "recall_ci95_upper": ci["upper"],
                "unique_vs_ts2vec": len(selected - ts_set),
                "ts2vec_only_vs_method": len(ts_set - selected),
                "shared_with_ts2vec": len(selected & ts_set),
            }
        )

    p_values = {}
    pairwise = {}
    for method in methods:
        if method == "ts2vec_personal_quantile_dynamic":
            continue
        test = mcnemar_exact(
            len(ts_set - hit_sets[method]),
            len(hit_sets[method] - ts_set),
        )
        pairwise[method] = test
        p_values[method] = test["exact_two_sided_p"]
    adjusted = holm_adjust(p_values)
    for method in pairwise:
        pairwise[method]["holm_adjusted_p"] = adjusted[method]

    scenario_metrics = {}
    for method in methods:
        by_scenario = {}
        for scenario, frame in incidents.groupby("scenario"):
            ids = set(frame["incident_id"])
            by_scenario[str(int(scenario))] = {
                "hits": len(ids & hit_sets[method]),
                "events": len(ids),
                "event_recall": (
                    len(ids & hit_sets[method]) / len(ids)
                ),
            }
        scenario_metrics[method] = {
            "by_scenario": by_scenario,
            "macro_event_recall": float(
                np.mean(
                    [
                        value["event_recall"]
                        for value in by_scenario.values()
                    ]
                )
            ),
        }

    leave_one_scenario_out = {}
    for scenario in sorted(incidents["scenario"].unique()):
        retained = incidents[incidents["scenario"] != scenario]
        retained_ids = set(retained["incident_id"])
        leave_one_scenario_out[str(int(scenario))] = {
            method: {
                "hits": len(hit_sets[method] & retained_ids),
                "events": len(retained_ids),
                "event_recall": (
                    len(hit_sets[method] & retained_ids)
                    / len(retained_ids)
                ),
            }
            for method in methods
        }

    incident_accounts = {
        row.incident_id: set(row.observed_accounts.split("|"))
        for row in incidents.itertuples(index=False)
    }
    account_incidents = {}
    for incident_id, accounts in incident_accounts.items():
        for account in accounts:
            account_incidents.setdefault(account, set()).add(incident_id)
    overlapping_pairs = 0
    incident_rows = list(incidents.itertuples(index=False))
    for index, left in enumerate(incident_rows):
        for right in incident_rows[index + 1 :]:
            if (
                left.first_observed_date <= right.last_observed_date
                and right.first_observed_date <= left.last_observed_date
            ):
                overlapping_pairs += 1

    month_clusters = {}
    for row in incidents.itertuples(index=False):
        month = str(row.first_observed_date.to_period("M"))
        month_clusters.setdefault(month, []).append(
            row.incident_id
        )
    cluster_names = sorted(month_clusters)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_recalls = {
        method: [] for method in methods
    }
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_months = rng.choice(
            cluster_names,
            size=len(cluster_names),
            replace=True,
        )
        sampled_ids = [
            incident_id
            for month in sampled_months
            for incident_id in month_clusters[str(month)]
        ]
        for method in methods:
            bootstrap_recalls[method].append(
                np.mean(
                    [
                        incident_id in hit_sets[method]
                        for incident_id in sampled_ids
                    ]
                )
            )

    ts_method = "ts2vec_personal_quantile_dynamic"
    cluster_bootstrap = {}
    for method in methods:
        values = np.asarray(bootstrap_recalls[method], dtype=float)
        entry = {
            "event_recall_mean": float(values.mean()),
            "event_recall_ci95_lower": float(
                np.quantile(values, 0.025)
            ),
            "event_recall_ci95_upper": float(
                np.quantile(values, 0.975)
            ),
        }
        if method != ts_method:
            differences = values - np.asarray(
                bootstrap_recalls[ts_method], dtype=float
            )
            entry["difference_vs_ts2vec_mean"] = float(
                differences.mean()
            )
            entry[
                "difference_vs_ts2vec_ci95_lower"
            ] = float(np.quantile(differences, 0.025))
            entry[
                "difference_vs_ts2vec_ci95_upper"
            ] = float(np.quantile(differences, 0.975))
        cluster_bootstrap[method] = entry

    per_k = []
    pairwise_by_k = {}
    for k in sorted(hits["k"].unique()):
        for horizon in sorted(hits["horizon_days"].unique()):
            subset = hits[
                (hits["k"] == k)
                & (hits["horizon_days"] == horizon)
            ]
            for method in methods:
                count = len(
                    set(
                        subset.loc[
                            subset["method"] == method, "incident_id"
                        ]
                    )
                )
                ci = wilson_interval(
                    count, len(all_incident_ids)
                )
                per_k.append(
                    {
                        "k": int(k),
                        "horizon_days": int(horizon),
                        "method": method,
                        "hits": count,
                        "events": len(all_incident_ids),
                        "event_recall": count
                        / len(all_incident_ids),
                        "recall_ci95_lower": ci["lower"],
                        "recall_ci95_upper": ci["upper"],
                    }
                )
        horizon_subset = hits[
            (hits["k"] == k)
            & (hits["horizon_days"] == PRIMARY_HORIZON)
        ]
        k_sets = {
            method: set(
                horizon_subset.loc[
                    horizon_subset["method"] == method,
                    "incident_id",
                ]
            )
            for method in methods
        }
        k_p_values = {}
        k_pairwise = {}
        for method in methods:
            if method == "ts2vec_personal_quantile_dynamic":
                continue
            test = mcnemar_exact(
                len(k_sets["ts2vec_personal_quantile_dynamic"] - k_sets[method]),
                len(k_sets[method] - k_sets["ts2vec_personal_quantile_dynamic"]),
            )
            k_pairwise[method] = test
            k_p_values[method] = test["exact_two_sided_p"]
        k_adjusted = holm_adjust(k_p_values)
        for method in k_pairwise:
            k_pairwise[method]["holm_adjusted_p"] = k_adjusted[method]
        pairwise_by_k[str(k)] = k_pairwise

    result_rows = json.loads(
        RESULTS.read_text(encoding="utf-8")
    )
    primary_results = {
        row["method"]: row["external_test"]
        for row in result_rows
        if row["k"] == PRIMARY_K
    }
    output = {
        "primary_k": PRIMARY_K,
        "primary_horizon_days": PRIMARY_HORIZON,
        "event_count": len(all_incident_ids),
        "methods": method_rows,
        "pairwise_against_ts2vec": pairwise,
        "scenario_metrics": scenario_metrics,
        "leave_one_scenario_out": leave_one_scenario_out,
        "event_dependence_diagnostics": {
            "overlapping_incident_interval_pairs": overlapping_pairs,
            "accounts_in_multiple_incidents": sum(
                len(ids) > 1
                for ids in account_incidents.values()
            ),
            "month_clusters": len(cluster_names),
            "cluster_sizes": {
                month: len(ids)
                for month, ids in month_clusters.items()
            },
        },
        "month_cluster_bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "methods": cluster_bootstrap,
            "note": (
                "Resamples calendar months with replacement to reflect "
                "within-month dependence. This is an exploratory "
                "cluster-robust sensitivity analysis."
            ),
        },
        "primary_operational_metrics": primary_results,
        "recall_by_k_and_horizon": per_k,
        "pairwise_against_ts2vec_by_k": pairwise_by_k,
        "statistical_note": (
            "Wilson intervals are marginal event-level intervals. McNemar "
            "tests are event-paired and exploratory because incidents "
            "overlap in time. Holm adjustment is across the three "
            "non-TS2Vec methods. Prefer scenario macro recall and the "
            "month-cluster bootstrap for robustness interpretation."
        ),
    }
    STATISTICS.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        pd.DataFrame(method_rows).to_string(index=False),
        flush=True,
    )
    print(
        json.dumps(pairwise, ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
