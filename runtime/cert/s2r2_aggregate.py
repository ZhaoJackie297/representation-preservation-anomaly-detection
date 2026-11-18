from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"

sys.path.insert(0, str(ROOT))
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from quota_relationship_sim import (  # noqa: E402
    evaluate_alert_system,
    load_r42,
    load_r52,
)


BLOCKS = {
    "r42_original": ("r42", False),
    "r42_retrained": ("r42", False),
    "r52_external": ("r52", True),
}


def event_ids(
    data: dict,
    alerts: np.ndarray,
    *,
    external: bool,
    split: str | None,
) -> set[int]:
    result = evaluate_alert_system(
        data, alerts, "S2R2", 10, external=external
    )
    if external:
        return set(
            result["external_test"]["horizons"]["3"][
                "incident_ids"
            ]
        )
    return set(result["splits"][split]["incident_ids"])


def main() -> None:
    r42 = load_r42()
    r52 = load_r52()
    families = {"r42": r42, "r52": r52}
    frames = []
    event_rows = []
    for block_name, (family, external) in BLOCKS.items():
        data = families[family]
        frame = pd.read_csv(
            OUT / f"s2r2_{block_name}_metrics.csv"
        )
        frame.insert(0, "research_block", block_name)
        frames.append(frame)
        arrays = np.load(
            OUT / f"s2r2_{block_name}_components.npz",
            allow_pickle=False,
        )
        methods = {
            "single": (
                arrays["single_signal"],
                arrays["single_aligned_iforest"],
            ),
            "multiprototype": (
                arrays["prototype_signal"],
                arrays["prototype_aligned_iforest"],
            ),
        }
        hit_sets = {}
        for method, (signal, auxiliary) in methods.items():
            alerts = build_unique_quota_alerts(
                signal,
                auxiliary,
                data["active_mask"],
                10,
                4,
            )
            hit_sets[method] = {}
            splits = (
                ["external_test"] if external else ["validation", "test"]
            )
            for split in splits:
                hit_sets[method][split] = event_ids(
                    data,
                    alerts,
                    external=external,
                    split=split,
                )
        for split in hit_sets["single"]:
            single = hit_sets["single"][split]
            multi = hit_sets["multiprototype"][split]
            single_only = len(single - multi)
            multi_only = len(multi - single)
            discordant = single_only + multi_only
            p_value = (
                float(
                    binomtest(
                        multi_only,
                        discordant,
                        0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if discordant
                else 1.0
            )
            event_rows.append(
                {
                    "data_block": block_name,
                    "split": split,
                    "single_hits": len(single),
                    "multiprototype_hits": len(multi),
                    "single_only": single_only,
                    "multiprototype_only": multi_only,
                    "mcnemar_exact_p": p_value,
                }
            )
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(
        OUT / "s2r2_combined_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    k10 = metrics[
        (metrics["k"] == 10)
        & (metrics["variant"] == "aligned_iforest")
    ]
    means = k10.groupby("method", as_index=False)[
        "balanced"
    ].mean()
    means["balanced"] *= 100.0
    by_k = metrics[
        metrics["variant"] == "aligned_iforest"
    ].groupby(["k", "method"], as_index=False)[
        "balanced"
    ].mean()
    by_k["balanced"] *= 100.0
    by_block = (
        100
        * k10.pivot_table(
            index=["data_block", "split"],
            columns="method",
            values="balanced",
        )
    )
    paired = by_block.reset_index()
    paired["difference"] = (
        paired["S2R2_multiprototype"] - paired["S2R2_single"]
    )
    t_stat = ttest_rel(
        paired["S2R2_multiprototype"], paired["S2R2_single"]
    )
    wil = wilcoxon(
        paired["S2R2_multiprototype"], paired["S2R2_single"]
    )
    event_frame = pd.DataFrame(event_rows)
    event_frame.to_csv(
        OUT / "s2r2_event_mcnemar.csv",
        index=False,
        encoding="utf-8",
    )
    event_single_only = int(event_frame["single_only"].sum())
    event_multi_only = int(event_frame["multiprototype_only"].sum())
    event_discordant = event_single_only + event_multi_only
    event_p = (
        float(
            binomtest(
                event_multi_only,
                event_discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
        if event_discordant
        else 1.0
    )
    summary = {
        "k10_mean": means.to_dict(orient="records"),
        "k10_paired": {
            "mean_difference": float(paired["difference"].mean()),
            "wins": int((paired["difference"] > 0).sum()),
            "ties": int((paired["difference"] == 0).sum()),
            "losses": int((paired["difference"] < 0).sum()),
            "paired_t_p": float(t_stat.pvalue),
            "wilcoxon_p": float(wil.pvalue),
        },
        "events": {
            "single_total": int(event_frame["single_hits"].sum()),
            "multiprototype_total": int(
                event_frame["multiprototype_hits"].sum()
            ),
            "single_only": event_single_only,
            "multiprototype_only": event_multi_only,
            "mcnemar_exact_p": event_p,
        },
        "by_k": by_k.to_dict(orient="records"),
        "by_block": paired.to_dict(orient="records"),
    }
    (OUT / "s2r2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    by_k.to_csv(
        OUT / "s2r2_by_k.csv",
        index=False,
        encoding="utf-8",
    )
    by_block.to_csv(
        OUT / "s2r2_k10_blocks.csv",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
