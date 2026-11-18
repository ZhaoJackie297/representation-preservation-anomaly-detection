from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "experiments" / "representation_preservation"


def extract_test(
    frame: pd.DataFrame,
    method_id: str,
) -> dict:
    selected = frame[
        (frame["method_id"] == method_id)
        & (frame["split"] == "test")
    ].sort_values("k")
    if selected.empty:
        raise RuntimeError(f"Missing test metrics for {method_id}")
    first = selected.iloc[0]
    return {
        "test_pr_auc": float(first["pr_auc_mean"]),
        "test_roc_auc": float(first["roc_auc_mean"]),
        "test_k5": float(
            selected.loc[selected["k"] == 5, "incident_hits_mean"].iloc[0]
        ),
        "test_k10": float(
            selected.loc[selected["k"] == 10, "incident_hits_mean"].iloc[0]
        ),
        "test_k20": float(
            selected.loc[selected["k"] == 20, "incident_hits_mean"].iloc[0]
        ),
    }


def main() -> None:
    canonical_summary = pd.read_csv(
        BASE / "canonical" / "canonical_metric_summary.csv"
    )
    rows = {}
    for label, method_id in (
        ("iForest-27D", "E11_iforest_27d"),
        ("T0 mask", "T0_mask_original"),
        ("T4+PAI-A1", "T4_full_soft_pai_a1"),
    ):
        rows[label] = extract_test(canonical_summary, method_id)

    table = pd.DataFrame.from_dict(rows, orient="index").reset_index()
    table = table.rename(columns={"index": "method"})
    table.to_csv(
        BASE / "e11_iforest" / "iforest_vs_reference.csv",
        index=False,
    )
    table.to_csv(
        BASE / "canonical" / "e11_iforest_vs_reference.csv",
        index=False,
    )

    deltas = pd.DataFrame(
        [
            {
                "comparison": "iForest - T0 mask",
                **{
                    column: rows["iForest-27D"][column]
                    - rows["T0 mask"][column]
                    for column in rows["T0 mask"]
                },
            },
            {
                "comparison": "T4+PAI-A1 - iForest",
                **{
                    column: rows["T4+PAI-A1"][column]
                    - rows["iForest-27D"][column]
                    for column in rows["T0 mask"]
                },
            },
        ]
    )
    deltas.to_csv(
        BASE / "e11_iforest" / "iforest_deltas.csv",
        index=False,
    )
    print(table.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
