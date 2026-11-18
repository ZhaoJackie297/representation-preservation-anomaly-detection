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
    "fixed_dual_iforest",
    "channel_top2_iforest",
    "triple_mean_complete",
    "triple_mean_iforest",
]
CANDIDATE = "triple_mean_iforest"
COMPARISONS = [
    ("iforest_complete", CANDIDATE),
    ("fixed_dual_iforest", CANDIDATE),
    ("channel_top2_iforest", CANDIDATE),
    ("triple_mean_complete", CANDIDATE),
]


def wilson(successes: int, total: int) -> tuple[float, float]:
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


def extract_hit_ids(result: dict, split: str, external: bool) -> set[str]:
    if external:
        return set(
            result["external_test"]["horizons"]["3"]["incident_ids"]
        )
    return set(result["splits"][split]["incident_ids"])


def holm(values: list[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda index: values[index])
    adjusted = [1.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (n - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def main() -> None:
    arrays = np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    )
    dual_arrays = np.load(
        OUT / "ts2vec_dual_score_arrays.npz",
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
        },
        {
            "name": "r42_retrained",
            "data": r42,
            "external": False,
            "splits": ["validation", "test"],
        },
        {
            "name": "r52_external",
            "data": r52,
            "external": True,
            "splits": ["external_test"],
        },
    ]

    event_rows = []
    interval_rows = []
    for block in blocks:
        name = str(block["name"])
        data = block["data"]
        external = bool(block["external"])
        auxiliary = dual_arrays[f"{name}_dual_view_iforest"]
        signals = {
            "fixed_dual_iforest": arrays[
                f"{name}_fixed_dual_iforest_signal"
            ],
            "channel_top2_iforest": arrays[
                f"{name}_channel_top2_iforest_signal"
            ],
            "triple_mean_iforest": arrays[
                f"{name}_triple_mean_iforest_signal"
            ],
            "triple_mean_complete": arrays[
                f"{name}_triple_mean_complete_signal"
            ],
        }
        for split in block["splits"]:
            hits_by_method = {}
            totals = {}
            for method in METHODS:
                if method == "iforest_complete":
                    alerts = build_alerts(
                        auxiliary,
                        data["active_mask"],
                        PRIMARY_K,
                        7,
                    )
                elif method == "triple_mean_complete":
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
                hits_by_method[method] = extract_hit_ids(
                    result, split, external
                )
                section = (
                    result["external_test"]
                    if external
                    else result["splits"][split]
                )
                total = int(section["horizons"]["3"]["incidents_total"])
                hits = len(hits_by_method[method])
                low, high = wilson(hits, total)
                interval_rows.append(
                    {
                        "data_block": name,
                        "split": split,
                        "method": method,
                        "incident_hits": hits,
                        "incident_total": total,
                        "event_recall_3d": hits / total if total else 0.0,
                        "wilson_95_low": low,
                        "wilson_95_high": high,
                    }
                )
                totals[method] = total

            for baseline, candidate in COMPARISONS:
                baseline_hits = hits_by_method[baseline]
                candidate_hits = hits_by_method[candidate]
                both = len(baseline_hits & candidate_hits)
                baseline_only = len(baseline_hits - candidate_hits)
                candidate_only = len(candidate_hits - baseline_hits)
                total = totals[candidate]
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
                        "data_block": name,
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
        event_tests.loc[mask, "holm_p_within_comparison"] = holm(
            event_tests.loc[mask, "mcnemar_exact_p"].tolist()
        )
    event_tests.to_csv(
        OUT / "channel_evidence_paired_event_tests.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(interval_rows).to_csv(
        OUT / "channel_evidence_wilson_intervals.csv",
        index=False,
        encoding="utf-8",
    )

    rankings = pd.read_csv(
        OUT / "channel_evidence_k10_rankings.csv"
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
    utility_tests["holm_p"] = holm(
        utility_tests["sign_test_exact_p"].tolist()
    )
    utility_tests.to_csv(
        OUT / "channel_evidence_paired_utility_tests.csv",
        index=False,
        encoding="utf-8",
    )

    summary = {
        "primary_k": PRIMARY_K,
        "q": q,
        "candidate": CANDIDATE,
        "event_tests": event_tests.to_dict(orient="records"),
        "utility_tests": utility_tests.to_dict(orient="records"),
        "outputs": {
            "event_tests": str(
                OUT / "channel_evidence_paired_event_tests.csv"
            ),
            "wilson_intervals": str(
                OUT / "channel_evidence_wilson_intervals.csv"
            ),
            "utility_tests": str(
                OUT / "channel_evidence_paired_utility_tests.csv"
            ),
        },
    }
    (OUT / "channel_evidence_paired_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nPaired event tests", flush=True)
    print(event_tests.to_string(index=False), flush=True)
    print("\nPaired utility tests", flush=True)
    print(utility_tests.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
