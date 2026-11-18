from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
PRIMARY_K = 10

sys.path.insert(0, str(ROOT))
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from production_protocol_r42 import build_alerts  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    evaluate_alert_system,
    load_r42,
    load_r52,
    q_from_alpha,
)


METHODS = [
    "iforest_complete",
    "ts_mask_dual_iforest",
    "ts_dual_mean_complete",
    "ts_dual_mean_dual_iforest",
    "ts_dual_max_dual_iforest",
]
CANDIDATE = "ts_dual_mean_dual_iforest"
COMPARISONS = [
    ("iforest_complete", CANDIDATE),
    ("ts_mask_dual_iforest", CANDIDATE),
    ("ts_dual_mean_complete", CANDIDATE),
    ("ts_dual_max_dual_iforest", CANDIDATE),
]


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def extract_hit_ids(
    result: dict,
    split: str,
    external: bool,
) -> set[str]:
    section = (
        result["external_test"]
        if external
        else result["splits"][split]
    )
    if external:
        return set(section["horizons"]["3"]["incident_ids"])
    return set(section["incident_ids"])


def holm_adjust(values: list[float]) -> list[float]:
    count = len(values)
    order = sorted(range(count), key=lambda index: values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def main() -> None:
    arrays = np.load(
        OUT / "ts2vec_dual_score_arrays.npz",
        allow_pickle=False,
    )
    view_scores = np.load(
        OUT / "iforest_view_ensemble_scores.npz",
        allow_pickle=False,
    )
    r42 = load_r42()
    r52 = load_r52()
    q = q_from_alpha(PRIMARY_K, 0.4)

    blocks = [
        {
            "name": "r42_original",
            "data": r42,
            "external": False,
            "splits": ["validation", "test"],
            "original_iforest": view_scores["r42_original"],
        },
        {
            "name": "r42_retrained",
            "data": r42,
            "external": False,
            "splits": ["validation", "test"],
            "original_iforest": view_scores["retrained_original"],
        },
        {
            "name": "r52_external",
            "data": r52,
            "external": True,
            "splits": ["external_test"],
            "original_iforest": view_scores["r52_original"],
        },
    ]

    event_rows = []
    recall_rows = []
    for block in blocks:
        block_name = str(block["name"])
        data = block["data"]
        external = bool(block["external"])
        auxiliary = arrays[f"{block_name}_dual_view_iforest"]
        signals = {
            "ts_mask_dual_iforest": arrays[
                f"{block_name}_ts_mask_signal"
            ],
            "ts_dual_mean_complete": arrays[
                f"{block_name}_ts_dual_mean_signal"
            ],
            "ts_dual_mean_dual_iforest": arrays[
                f"{block_name}_ts_dual_mean_signal"
            ],
            "ts_dual_max_dual_iforest": arrays[
                f"{block_name}_ts_dual_max_signal"
            ],
        }
        hit_sets_by_split: dict[str, dict[str, set[str]]] = {}
        for split in block["splits"]:
            hit_sets = {}
            for method in METHODS:
                if method == "iforest_complete":
                    alerts = build_alerts(
                        block["original_iforest"],
                        data["active_mask"],
                        PRIMARY_K,
                        7,
                    )
                elif method == "ts_dual_mean_complete":
                    alerts = build_alerts(
                        signals[method],
                        data["active_mask"],
                        PRIMARY_K,
                        7,
                    )
                else:
                    alerts = build_unique_quota_alerts(
                        signals[method],
                        auxiliary,
                        data["active_mask"],
                        PRIMARY_K,
                        q,
                    )
                result = evaluate_alert_system(
                    data,
                    alerts,
                    method,
                    PRIMARY_K,
                    external=external,
                )
                hit_sets[method] = extract_hit_ids(
                    result, split, external
                )

                section = (
                    result["external_test"]
                    if external
                    else result["splits"][split]
                )
                horizon = section["horizons"]["3"]
                total = int(horizon["incidents_total"])
                hits = len(hit_sets[method])
                low, high = wilson_interval(hits, total)
                recall_rows.append(
                    {
                        "data_block": block_name,
                        "split": split,
                        "method": method,
                        "incident_hits": hits,
                        "incident_total": total,
                        "event_recall_3d": hits / total if total else 0.0,
                        "wilson_95_low": low,
                        "wilson_95_high": high,
                    }
                )
            hit_sets_by_split[split] = hit_sets

        for split, hit_sets in hit_sets_by_split.items():
            for baseline, candidate in COMPARISONS:
                baseline_hits = hit_sets[baseline]
                candidate_hits = hit_sets[candidate]
                both = len(baseline_hits & candidate_hits)
                baseline_only = len(baseline_hits - candidate_hits)
                candidate_only = len(candidate_hits - baseline_hits)
                # Incident universe is identical within a split; derive
                # the neither count from the evaluated incident total.
                total = next(
                    row["incident_total"]
                    for row in recall_rows
                    if row["data_block"] == block_name
                    and row["split"] == split
                )
                neither = total - both - baseline_only - candidate_only
                discordant = baseline_only + candidate_only
                p_value = (
                    binomtest(
                        baseline_only,
                        discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                    if discordant
                    else 1.0
                )
                event_rows.append(
                    {
                        "data_block": block_name,
                        "split": split,
                        "baseline": baseline,
                        "candidate": candidate,
                        "both_hit": both,
                        "baseline_only": baseline_only,
                        "candidate_only": candidate_only,
                        "neither_hit": neither,
                        "discordant": discordant,
                        "mcnemar_exact_p": p_value,
                    }
                )

    event_tests = pd.DataFrame(event_rows)
    event_tests["holm_p_within_comparison"] = np.nan
    for baseline, candidate in COMPARISONS:
        mask = (
            (event_tests["baseline"] == baseline)
            & (event_tests["candidate"] == candidate)
        )
        event_tests.loc[mask, "holm_p_within_comparison"] = holm_adjust(
            event_tests.loc[mask, "mcnemar_exact_p"].tolist()
        )
    event_tests.to_csv(
        OUT / "ts2vec_dual_score_paired_event_tests.csv",
        index=False,
        encoding="utf-8",
    )

    recall_intervals = pd.DataFrame(recall_rows)
    recall_intervals.to_csv(
        OUT / "ts2vec_dual_score_wilson_intervals.csv",
        index=False,
        encoding="utf-8",
    )

    rankings = pd.read_csv(
        OUT / "ts2vec_dual_score_k10_rankings.csv"
    )
    utility_rows = []
    for baseline, candidate in COMPARISONS:
        pivot = rankings.pivot_table(
            index=["data_block", "split", "profile"],
            columns="method",
            values="utility",
        )
        difference = pivot[candidate] - pivot[baseline]
        wins = int((difference > 1e-12).sum())
        losses = int((difference < -1e-12).sum())
        ties = int(len(difference) - wins - losses)
        p_value = (
            binomtest(
                wins,
                wins + losses,
                p=0.5,
                alternative="two-sided",
            ).pvalue
            if wins + losses
            else 1.0
        )
        utility_rows.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "units": len(difference),
                "candidate_wins": wins,
                "baseline_wins": losses,
                "ties": ties,
                "mean_utility_difference": float(difference.mean()),
                "median_utility_difference": float(difference.median()),
                "min_utility_difference": float(difference.min()),
                "max_utility_difference": float(difference.max()),
                "sign_test_exact_p": p_value,
            }
        )
    utility_tests = pd.DataFrame(utility_rows)
    utility_tests["holm_p"] = holm_adjust(
        utility_tests["sign_test_exact_p"].tolist()
    )
    utility_tests.to_csv(
        OUT / "ts2vec_dual_score_paired_utility_tests.csv",
        index=False,
        encoding="utf-8",
    )

    summary = {
        "primary_k": PRIMARY_K,
        "q": q,
        "candidate": CANDIDATE,
        "note": (
            "Paired tests are exploratory because event counts are small "
            "and r4.2 original/retrained share the same event definitions."
        ),
        "event_tests": event_tests.to_dict(orient="records"),
        "utility_tests": utility_tests.to_dict(orient="records"),
        "outputs": {
            "event_tests": str(
                OUT / "ts2vec_dual_score_paired_event_tests.csv"
            ),
            "recall_intervals": str(
                OUT / "ts2vec_dual_score_wilson_intervals.csv"
            ),
            "utility_tests": str(
                OUT / "ts2vec_dual_score_paired_utility_tests.csv"
            ),
        },
    }
    (OUT / "ts2vec_dual_score_paired_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nPaired event tests", flush=True)
    print(event_tests.to_string(index=False), flush=True)
    print("\nPaired utility tests", flush=True)
    print(utility_tests.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
