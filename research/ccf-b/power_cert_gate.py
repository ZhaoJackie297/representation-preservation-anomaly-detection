from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


Z_ONE_SIDED_ALPHA_025 = 1.959963984540054
Z_POWER_080 = 0.8416212335729143
PROJECTED_SEEDS = 10


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_at_k(
    rows: list[dict[str, str]],
    *,
    treatment: str,
    control: str,
    k: int,
) -> list[dict[str, float]]:
    by_seed: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if int(row["k"]) != k or row.get("split") not in (None, "", "test"):
            continue
        by_seed.setdefault(row["seed"], {})[row["method"]] = row
    pairs = []
    for seed, methods in sorted(by_seed.items()):
        if treatment not in methods or control not in methods:
            continue
        left = methods[treatment]
        right = methods[control]
        pairs.append(
            {
                "seed": float(seed),
                "delta_pr_auc": float(left["pr_auc"]) - float(right["pr_auc"]),
                "delta_event_recall": float(left["event_recall_3d"])
                - float(right["event_recall_3d"]),
                "baseline_event_recall": float(right["event_recall_3d"]),
                "baseline_event_hits": float(right["incident_hits"]),
                "events": float(right["incident_total"]),
            }
        )
    return pairs


def projected_mde(
    values: list[float],
    *,
    projected_seeds: int = PROJECTED_SEEDS,
) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("at least two paired seeds are required")
    observed_mean = mean(values)
    observed_sd = stdev(values)
    standard_error = observed_sd / math.sqrt(projected_seeds)
    mde = (Z_ONE_SIDED_ALPHA_025 + Z_POWER_080) * standard_error
    return {
        "observed_mean_delta": observed_mean,
        "observed_sd": observed_sd,
        "projected_se_mde": mde,
        "projected_seeds": projected_seeds,
    }


def binomial_event_mde(
    baseline_recall: float,
    events: int,
    *,
    correlation: float = 0.0,
) -> float:
    if events <= 0 or not 0 < baseline_recall < 1:
        raise ValueError("baseline recall must be in (0,1) and events positive")
    # Independent arm variance; the correlation adjustment is conservative
    # when methods share the same event population.
    independent_variance = (
        2.0 * baseline_recall * (1.0 - baseline_recall) / events
    )
    paired_variance = max(
        0.0,
        independent_variance * (1.0 - correlation),
    )
    return (
        Z_ONE_SIDED_ALPHA_025 + Z_POWER_080
    ) * math.sqrt(paired_variance)


def binomial_mde_grid(
    baseline_recall: float,
    events: int,
) -> dict[str, float]:
    return {
        f"rho_{rho:.1f}": binomial_event_mde(
            baseline_recall,
            events,
            correlation=rho,
        )
        for rho in (0.0, 0.5, 0.8, 0.9)
    }


def analyze(
    *,
    r42_path: Path,
    r52_path: Path,
) -> dict[str, object]:
    r42 = read_rows(r42_path)
    r52 = read_rows(r52_path)
    r42_pairs = paired_at_k(
        r42,
        treatment="T4_full_soft_euclidean",
        control="T3_instance_soft_euclidean",
        k=10,
    )
    # The r5.2 cache contains T4 endpoints but not a T3 row. Use the
    # Euclidean-vs-original contrast only as a variance proxy, never as the
    # preregistered A4-vs-A1+ test.
    r52_pairs = paired_at_k(
        r52,
        treatment="T4_bank_euclidean",
        control="T4_original",
        k=10,
    )
    for pairs in (r42_pairs, r52_pairs):
        if not pairs:
            raise ValueError("no paired seeds found")

    r42_pr = projected_mde([row["delta_pr_auc"] for row in r42_pairs])
    r42_recall = projected_mde(
        [row["delta_event_recall"] for row in r42_pairs]
    )
    r52_pr = projected_mde([row["delta_pr_auc"] for row in r52_pairs])
    r52_recall = projected_mde(
        [row["delta_event_recall"] for row in r52_pairs]
    )
    r42_baseline_recall = mean(
        row["baseline_event_recall"] for row in r42_pairs
    )
    r52_baseline_recall = mean(
        row["baseline_event_recall"] for row in r52_pairs
    )
    return {
        "status": "PROXY_ONLY_NOT_A_GATE_REPLACEMENT",
        "r4_2": {
            "comparison": "T4_full_soft_euclidean - T3_instance_soft_euclidean",
            "pairs": r42_pairs,
            "pr_auc_seed_proxy": r42_pr,
            "k10_seed_proxy": r42_recall,
            "baseline_event_recall": r42_baseline_recall,
            "events": int(r42_pairs[0]["events"]),
            "binomial_k10_mde_independent": binomial_event_mde(
                r42_baseline_recall,
                int(r42_pairs[0]["events"]),
            ),
            "binomial_k10_mde_grid": binomial_mde_grid(
                r42_baseline_recall,
                int(r42_pairs[0]["events"]),
            ),
        },
        "r5_2": {
            "comparison": "T4_bank_euclidean - T4_original (variance proxy)",
            "pairs": r52_pairs,
            "pr_auc_seed_proxy": r52_pr,
            "k10_seed_proxy": r52_recall,
            "baseline_event_recall": r52_baseline_recall,
            "events": int(r52_pairs[0]["events"]),
            "binomial_k10_mde_independent": binomial_event_mde(
                r52_baseline_recall,
                int(r52_pairs[0]["events"]),
            ),
            "binomial_k10_mde_grid": binomial_mde_grid(
                r52_baseline_recall,
                int(r52_pairs[0]["events"]),
            ),
        },
        "interpretation": (
            "These are projections from three historical seeds and a 99-event "
            "binomial approximation. They do not estimate A4-vs-A1+ and cannot "
            "replace the preregistered event/user-cluster power simulation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r42-metrics", type=Path, required=True)
    parser.add_argument("--r52-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(r42_path=args.r42_metrics, r52_path=args.r52_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
