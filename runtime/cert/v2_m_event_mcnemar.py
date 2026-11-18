from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"

sys.path.insert(0, str(ROOT))
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


def hit_ids(
    data: dict,
    alerts: np.ndarray,
    *,
    external: bool,
    split: str | None,
) -> set[int]:
    result = evaluate_alert_system(
        data, alerts, "comparison", 10, external=external
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
    v2_components = np.load(
        OUT / "v2_pipeline_components.npz",
        allow_pickle=False,
    )
    v2m_components = np.load(
        OUT / "v2_m_ensemble_components.npz",
        allow_pickle=False,
    )
    rows = []
    for block_name, (family, external) in BLOCKS.items():
        data = families[family]
        v2_alerts = v2_components[f"{block_name}_alerts_k10"]
        v2m_alerts = v2m_components[f"{block_name}_alerts_k10"]
        splits = (
            ["external_test"] if external else ["validation", "test"]
        )
        for split in splits:
            v2_hits = hit_ids(
                data,
                v2_alerts,
                external=external,
                split=split,
            )
            v2m_hits = hit_ids(
                data,
                v2m_alerts,
                external=external,
                split=split,
            )
            universe = v2_hits | v2m_hits
            both = len(v2_hits & v2m_hits)
            v2_only = len(v2_hits - v2m_hits)
            v2m_only = len(v2m_hits - v2_hits)
            discordant = v2_only + v2m_only
            p_value = (
                float(
                    binomtest(
                        v2m_only,
                        discordant,
                        0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if discordant
                else 1.0
            )
            rows.append(
                {
                    "data_block": block_name,
                    "split": split,
                    "v2_hits": len(v2_hits),
                    "v2m_hits": len(v2m_hits),
                    "both": both,
                    "v2_only": v2_only,
                    "v2m_only": v2m_only,
                    "union": len(universe),
                    "mcnemar_exact_p": p_value,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        OUT / "v2_m_event_mcnemar.csv",
        index=False,
        encoding="utf-8",
    )
    total_v2_only = int(frame["v2_only"].sum())
    total_v2m_only = int(frame["v2m_only"].sum())
    discordant = total_v2_only + total_v2m_only
    aggregate_p = (
        float(
            binomtest(
                total_v2m_only,
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    summary = {
        "rows": rows,
        "aggregate": {
            "v2_hits": int(frame["v2_hits"].sum()),
            "v2m_hits": int(frame["v2m_hits"].sum()),
            "v2_only": total_v2_only,
            "v2m_only": total_v2m_only,
            "mcnemar_exact_p": aggregate_p,
        },
    }
    (OUT / "v2_m_event_mcnemar_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
