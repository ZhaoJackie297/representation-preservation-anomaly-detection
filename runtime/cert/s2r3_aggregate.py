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
    external: bool,
    split: str | None,
) -> set[int]:
    result = evaluate_alert_system(
        data, alerts, "S2R3", 10, external=external
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
            OUT / f"s2r3_{block_name}_metrics.csv"
        )
        frames.append(frame)
        components = np.load(
            OUT / f"s2r3_{block_name}_components.npz",
            allow_pickle=False
        )
        synthetic_alerts = build_unique_quota_alerts(
            components["synthetic_signal"],
            components["aligned_iforest"],
            data["active_mask"],
            10,
            4,
        )
        s2r2_components = np.load(
            OUT / f"s2r2_{block_name}_components.npz",
            allow_pickle=False,
        )
        baseline_alerts = build_unique_quota_alerts(
            s2r2_components["single_signal"],
            s2r2_components["single_aligned_iforest"],
            data["active_mask"],
            10,
            4,
        )
        splits = (
            ["external_test"] if external else ["validation", "test"]
        )
        for split in splits:
            baseline = event_ids(
                data, baseline_alerts, external, split
            )
            synthetic = event_ids(
                data, synthetic_alerts, external, split
            )
            base_only = len(baseline - synthetic)
            synth_only = len(synthetic - baseline)
            discordant = base_only + synth_only
            p_value = (
                float(
                    binomtest(
                        synth_only,
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
                    "baseline_hits": len(baseline),
                    "synthetic_hits": len(synthetic),
                    "baseline_only": base_only,
                    "synthetic_only": synth_only,
                    "mcnemar_exact_p": p_value,
                }
            )
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(
        OUT / "s2r3_combined_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    k10 = metrics[
        (metrics["k"] == 10)
        & (metrics["variant"] == "aligned_iforest")
    ].copy()
    baseline = pd.read_csv(
        OUT / "s2r2_combined_metrics.csv"
    )
    baseline = baseline[
        (baseline["k"] == 10)
        & (baseline["method"] == "S2R2_single")
        & (baseline["variant"] == "aligned_iforest")
    ][["data_block", "split", "balanced"]].rename(
        columns={"balanced": "baseline"}
    )
    compare = k10[
        ["data_block", "split", "balanced"]
    ].rename(columns={"balanced": "synthetic"})
    compare = compare.merge(
        baseline, on=["data_block", "split"]
    )
    compare["difference"] = (
        compare["synthetic"] - compare["baseline"]
    )
    compare.to_csv(
        OUT / "s2r3_vs_baseline_k10.csv",
        index=False,
        encoding="utf-8",
    )
    by_k = metrics[
        metrics["variant"] == "aligned_iforest"
    ].groupby("k", as_index=False)["balanced"].mean()
    by_k["balanced"] *= 100.0
    event_frame = pd.DataFrame(event_rows)
    event_frame.to_csv(
        OUT / "s2r3_event_mcnemar.csv",
        index=False,
        encoding="utf-8",
    )
    base_only = int(event_frame["baseline_only"].sum())
    synth_only = int(event_frame["synthetic_only"].sum())
    discordant = base_only + synth_only
    event_p = (
        float(
            binomtest(
                synth_only,
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    summary = {
        "baseline_k10_mean": float(compare["baseline"].mean()),
        "synthetic_k10_mean": float(compare["synthetic"].mean()),
        "mean_difference": float(compare["difference"].mean()),
        "wins": int((compare["difference"] > 0).sum()),
        "ties": int((compare["difference"] == 0).sum()),
        "losses": int((compare["difference"] < 0).sum()),
        "paired_t_p": float(
            ttest_rel(
                compare["synthetic"], compare["baseline"]
            ).pvalue
        ),
        "wilcoxon_p": float(
            wilcoxon(
                compare["synthetic"], compare["baseline"]
            ).pvalue
        ),
        "events": {
            "baseline_total": int(
                event_frame["baseline_hits"].sum()
            ),
            "synthetic_total": int(
                event_frame["synthetic_hits"].sum()
            ),
            "baseline_only": base_only,
            "synthetic_only": synth_only,
            "mcnemar_exact_p": event_p,
        },
        "by_k": by_k.to_dict(orient="records"),
        "by_block": compare.to_dict(orient="records"),
    }
    (OUT / "s2r3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
