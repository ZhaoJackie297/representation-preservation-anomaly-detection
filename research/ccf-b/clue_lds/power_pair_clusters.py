from __future__ import annotations

import argparse
import json
import math

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def power_for_shift(
    *,
    baseline_recall: float,
    log_odds_shift: float,
    cluster_sd: float,
    pairs: int,
    events_per_pair: int,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    cluster_effect = rng.normal(
        0.0, cluster_sd, size=(repetitions, pairs)
    )
    baseline_probability = sigmoid(
        logit(baseline_recall) + cluster_effect
    )
    treatment_probability = sigmoid(
        logit(baseline_recall) + cluster_effect + log_odds_shift
    )
    baseline_hits = rng.binomial(
        n=1,
        p=baseline_probability[..., None],
        size=(repetitions, pairs, events_per_pair),
    )
    treatment_hits = rng.binomial(
        n=1,
        p=treatment_probability[..., None],
        size=(repetitions, pairs, events_per_pair),
    )
    cluster_differences = (
        treatment_hits.mean(axis=2) - baseline_hits.mean(axis=2)
    )
    mean_difference = cluster_differences.mean(axis=1)
    standard_error = (
        cluster_differences.std(axis=1, ddof=1) / math.sqrt(pairs)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = mean_difference / standard_error
    zero_se = standard_error <= 0
    statistic = np.where(
        zero_se & (mean_difference > 0),
        np.inf,
        np.where(
            zero_se & (mean_difference < 0),
            -np.inf,
            statistic,
        ),
    )
    # Conservative fixed critical value for one-sided alpha=0.025.
    critical = 1.959963984540054
    power = float(np.mean(statistic >= critical))
    treatment_recall = float(treatment_probability.mean())
    return power, treatment_recall - baseline_recall


def find_mde(
    *,
    baseline_recall: float,
    cluster_sd: float,
    pairs: int,
    events_per_pair: int,
    repetitions: int,
    target_power: float,
    seed: int,
) -> dict[str, float]:
    low, high = 0.0, 5.0
    for _ in range(16):
        mid = (low + high) / 2
        power, _ = power_for_shift(
            baseline_recall=baseline_recall,
            log_odds_shift=mid,
            cluster_sd=cluster_sd,
            pairs=pairs,
            events_per_pair=events_per_pair,
            repetitions=repetitions,
            seed=seed,
        )
        if power >= target_power:
            high = mid
        else:
            low = mid
    power, delta = power_for_shift(
        baseline_recall=baseline_recall,
        log_odds_shift=high,
        cluster_sd=cluster_sd,
        pairs=pairs,
        events_per_pair=events_per_pair,
        repetitions=repetitions,
        seed=seed,
    )
    treatment_recall = baseline_recall + delta
    return {
        "baseline_recall": baseline_recall,
        "cluster_sd": cluster_sd,
        "log_odds_shift": high,
        "absolute_recall_delta": delta,
        "treatment_recall": treatment_recall,
        "power": power,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--events-per-pair", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=2_000)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    baselines = (0.10, 0.25, 0.50)
    cluster_sds = (0.5, 1.0, 1.5)
    results = []
    for baseline in baselines:
        for cluster_sd in cluster_sds:
            results.append(
                find_mde(
                    baseline_recall=baseline,
                    cluster_sd=cluster_sd,
                    pairs=args.pairs,
                    events_per_pair=args.events_per_pair,
                    repetitions=args.repetitions,
                    target_power=args.target_power,
                    seed=args.seed,
                )
            )
    print(
        json.dumps(
            {
                "pairs": args.pairs,
                "events": args.pairs * args.events_per_pair,
                "repetitions": args.repetitions,
                "target_power": args.target_power,
                "one_sided_alpha": 0.025,
                "cluster_unit": "injected_user_pair",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
