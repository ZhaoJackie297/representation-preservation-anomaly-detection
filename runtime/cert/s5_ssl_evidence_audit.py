from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
CARDS = OUT / "s4_ssl_evidence_cards_v2ms_k10.csv"


def main() -> None:
    cards = pd.read_csv(CARDS)
    required = [
        "ssl_nearest_normal_distance",
        "ssl_velocity_l2",
        "ssl_curvature",
        "ssl_masked_strength",
        "ssl_manifold_strength",
        "ssl_counterfactual_strength",
        "ssl_high_view_count",
        "ssl_dominant_view",
        "ssl_view_agreement",
        "ssl_evidence_explanation",
    ]
    missing_columns = [
        column for column in required if column not in cards
    ]
    missing_cells = {
        column: int(cards[column].isna().sum())
        for column in required
        if column in cards
    }
    dominant_expected = np.argmax(
        cards[
            [
                "ssl_masked_strength",
                "ssl_manifold_strength",
                "ssl_counterfactual_strength",
            ]
        ].to_numpy(),
        axis=1,
    )
    expected_names = np.asarray(
        [
            "masked_reconstruction",
            "personal_manifold",
            "channel_counterfactual",
        ]
    )[dominant_expected]
    dominant_errors = int(
        (
            expected_names
            != cards["ssl_dominant_view"].astype(str).to_numpy()
        ).sum()
    )
    agreement_errors = int(
        (
            cards["ssl_view_agreement"].astype(bool)
            != (cards["ssl_high_view_count"] >= 2)
        ).sum()
    )
    label_columns = [
        column
        for column in cards.columns
        if "malicious" in column.lower()
        or "label" in column.lower()
    ]
    summary = {
        "cards": int(len(cards)),
        "blocks": cards["data_block"].value_counts().to_dict(),
        "required_columns": required,
        "missing_columns": missing_columns,
        "missing_cells": missing_cells,
        "dominant_view_errors": dominant_errors,
        "agreement_errors": agreement_errors,
        "label_columns_in_cards": label_columns,
        "labels_in_cards": bool(label_columns),
        "high_view_count_distribution": {
            str(int(key)): int(value)
            for key, value in cards[
                "ssl_high_view_count"
            ].value_counts().sort_index().items()
        },
        "agreement_rate": float(
            cards["ssl_view_agreement"].astype(bool).mean()
        ),
        "dominant_view_counts": cards[
            "ssl_dominant_view"
        ].value_counts().to_dict(),
        "integrity_pass": bool(
            not missing_columns
            and not label_columns
            and dominant_errors == 0
            and agreement_errors == 0
            and cards[
                "ssl_evidence_explanation"
            ].fillna("").str.len().gt(0).all()
        ),
        "limitations": [
            "自监督视图解释不是因果结论",
            "最近正常距离使用训练期参考集",
            "轨迹证据不改变S3排序",
            "缺失值保留为不可计算，不用0伪造",
        ],
        "output": str(
            OUT / "s5_ssl_evidence_audit.json"
        ),
    }
    (OUT / "s5_ssl_evidence_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
