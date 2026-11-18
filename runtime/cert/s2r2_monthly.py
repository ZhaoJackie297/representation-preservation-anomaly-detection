from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"

sys.path.insert(0, str(ROOT))
from feature_augmented_iforest_sim import (  # noqa: E402
    build_unique_quota_alerts,
)
from frozen_model_evidence_cards import monthly_stability  # noqa: E402
from quota_relationship_sim import load_r42, load_r52  # noqa: E402


BLOCKS = {
    "r42_original": ("r42", False),
    "r42_retrained": ("r42", False),
    "r52_external": ("r52", True),
}


def main() -> None:
    r42 = load_r42()
    r52 = load_r52()
    families = {"r42": r42, "r52": r52}
    frames = []
    for block_name, (family, external) in BLOCKS.items():
        data = families[family]
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
        for method, (signal, auxiliary) in methods.items():
            alerts = build_unique_quota_alerts(
                signal,
                auxiliary,
                data["active_mask"],
                10,
                4,
            )
            frame = monthly_stability(
                block_name,
                data,
                alerts,
                data["labels"],
                external=external,
                splits=(
                    ["external_test"]
                    if external
                    else ["validation", "test"]
                ),
            )
            frame.insert(3, "method", method)
            frames.append(frame)
    monthly = pd.concat(frames, ignore_index=True)
    monthly.to_csv(
        OUT / "s2r2_monthly_stability.csv",
        index=False,
        encoding="utf-8",
    )
    pivot = monthly.pivot_table(
        index=["data_block", "split", "month"],
        columns="method",
        values="balanced_utility",
    ).reset_index()
    pivot["difference"] = (
        pivot["multiprototype"] - pivot["single"]
    )
    pivot.to_csv(
        OUT / "s2r2_monthly_comparison.csv",
        index=False,
        encoding="utf-8",
    )
    summary = {
        "months_with_events": int(len(pivot)),
        "wins": int((pivot["difference"] > 1e-12).sum()),
        "ties": int((pivot["difference"].abs() <= 1e-12).sum()),
        "losses": int((pivot["difference"] < -1e-12).sum()),
        "mean_difference": float(pivot["difference"].mean()),
        "min_difference": float(pivot["difference"].min()),
        "max_difference": float(pivot["difference"].max()),
        "single_mean": float(pivot["single"].mean()),
        "multiprototype_mean": float(
            pivot["multiprototype"].mean()
        ),
        "single_min": float(pivot["single"].min()),
        "multiprototype_min": float(
            pivot["multiprototype"].min()
        ),
    }
    (OUT / "s2r2_monthly_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
