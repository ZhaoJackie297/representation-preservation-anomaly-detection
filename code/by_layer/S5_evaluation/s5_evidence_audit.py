from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
CARDS = OUT / "s4_evidence_cards_v2ms_k10.csv"
TIMELINES = OUT / "s4_user_timeline_summary_v2ms.csv"
V2M_S_METRICS = OUT / "v2m_s_metrics.csv"

sys.path.insert(0, str(ROOT))
from quota_relationship_sim import (  # noqa: E402
    load_r42_cached,
    load_r52_cached,
)


EXPECTED_SOURCE_REASON = {
    "ts_main": "TS_MAIN",
    "iforest_rescue": "IFOREST_RESCUE",
    "ts_fallback": "TS_FALLBACK",
}


def label_for_cards(
    cards: pd.DataFrame,
) -> tuple[pd.Series, dict]:
    r42 = load_r42_cached()
    r52 = load_r52_cached()
    families = {
        "r42_original": r42,
        "r42_retrained": r42,
        "r52_external": r52,
    }
    labels = []
    counts = {}
    for row in cards.itertuples(index=False):
        data = families[str(row.data_block)]
        users = data["users"].astype(str)
        dates = pd.to_datetime(data["dates"])
        user_index = int(
            np.flatnonzero(users == str(row.user_id))[0]
        )
        date_text = pd.Timestamp(row.date).strftime(
            "%Y-%m-%d"
        )
        date_index = int(
            np.flatnonzero(
                dates.strftime("%Y-%m-%d") == date_text
            )[0]
        )
        value = int(data["labels"][user_index, date_index])
        labels.append(value)
        counts[str(row.data_block)] = (
            counts.get(str(row.data_block), 0) + value
        )
    return pd.Series(labels, index=cards.index), counts


def main() -> None:
    cards = pd.read_csv(
        CARDS,
        parse_dates=["date"],
        dtype={"user_id": str},
    )
    timelines = pd.read_csv(
        TIMELINES,
        dtype={"user_id": str},
    )
    metrics = pd.read_csv(V2M_S_METRICS)
    labels, label_hits = label_for_cards(cards)
    cards["_audit_label"] = labels.to_numpy()

    integrity_rows = []
    channel_rows = []
    source_band_rows = []
    card_rows_by_block = {}
    for block_name in (
        "r42_original",
        "r42_retrained",
        "r52_external",
    ):
        block = cards[
            cards["data_block"] == block_name
        ].copy()
        card_rows_by_block[block_name] = int(len(block))
        duplicate_keys = int(
            block.duplicated(["date", "user_id"]).sum()
        )
        rank_problems = 0
        for _, date_group in block.groupby("date"):
            expected = min(10, len(date_group))
            if (
                date_group["rank"].min() != 1
                or date_group["rank"].max() != expected
                or date_group["rank"].nunique() != expected
            ):
                rank_problems += 1
        source_reason_errors = int(
            sum(
                EXPECTED_SOURCE_REASON.get(
                    str(row.source)
                )
                != str(row.reason_code)
                for row in block.itertuples()
            )
        )
        channel_errors = int(
            (
                ~block["top_channel_1"].isin(
                    ["logon", "device", "file", "email", "http"]
                )
            ).sum()
        )
        empty_explanations = int(
            block["explanation"].isna().sum()
            + block["explanation"].eq("").sum()
        )
        hazard_rows = int(block["hazard_3d"].notna().sum())
        integrity_rows.append(
            {
                "data_block": block_name,
                "cards": len(block),
                "duplicate_keys": duplicate_keys,
                "rank_problems": rank_problems,
                "source_reason_errors": source_reason_errors,
                "channel_errors": channel_errors,
                "empty_explanations": empty_explanations,
                "hazard_rows": hazard_rows,
                "label_column_in_cards": int(
                    "audit_malicious_account_day" in block.columns
                ),
                "audit_label_density": float(
                    block["_audit_label"].mean()
                ),
                "integrity_pass": bool(
                    duplicate_keys == 0
                    and rank_problems == 0
                    and source_reason_errors == 0
                    and channel_errors == 0
                    and empty_explanations == 0
                ),
            }
        )

        source_band = (
            block.groupby(["source", "queue_band"], dropna=False)
            .agg(
                cards=("user_id", "size"),
                audit_hits=("_audit_label", "sum"),
            )
            .reset_index()
        )
        source_band["audit_hit_density"] = (
            source_band["audit_hits"]
            / source_band["cards"].replace(0, np.nan)
        )
        source_band["data_block"] = block_name
        source_band_rows.append(source_band)

        channel = (
            block.groupby("top_channel_1")
            .agg(
                cards=("user_id", "size"),
                audit_hits=("_audit_label", "sum"),
            )
            .reset_index()
        )
        channel["audit_hit_density"] = (
            channel["audit_hits"]
            / channel["cards"].replace(0, np.nan)
        )
        channel["data_block"] = block_name
        channel_rows.append(channel)

    integrity = pd.DataFrame(integrity_rows)
    source_band = pd.concat(
        source_band_rows, ignore_index=True
    )
    channels = pd.concat(channel_rows, ignore_index=True)
    integrity.to_csv(
        OUT / "s5_card_integrity.csv",
        index=False,
        encoding="utf-8",
    )
    source_band.to_csv(
        OUT / "s5_source_band_audit.csv",
        index=False,
        encoding="utf-8",
    )
    channels.to_csv(
        OUT / "s5_channel_audit.csv",
        index=False,
        encoding="utf-8",
    )

    timeline_alert_sum = int(
        timelines["alert_count"].sum()
    )
    k10 = metrics[
        (metrics["k"] == 10)
        & (metrics["variant"] == "aligned_iforest")
    ].copy()
    summary = {
        "model": "V2-M-S + S4 evidence cards",
        "cards": int(len(cards)),
        "timelines": int(len(timelines)),
        "timeline_alert_sum": timeline_alert_sum,
        "card_timeline_consistency": bool(
            timeline_alert_sum == len(cards)
        ),
        "offline_labels_used_for_audit_only": True,
        "labels_written_to_cards": False,
        "blocks": card_rows_by_block,
        "audit_label_hits": {
            key: int(value)
            for key, value in label_hits.items()
        },
        "k10_metrics": k10[
            [
                "data_block",
                "split",
                "balanced",
                "incident_hits",
                "incident_total",
            ]
        ].to_dict(orient="records"),
        "integrity": integrity.to_dict(orient="records"),
        "source_band_audit": source_band.to_dict(
            orient="records"
        ),
        "channel_audit": channels.to_dict(
            orient="records"
        ),
        "limitations": [
            "离线恶意账户日只用于S5审计，未写入S4卡片",
            "r5.2是外部敏感性，不是独立确认",
            "通道命中密度不是因果归因",
            "hazard仅r4.2原模型覆盖",
        ],
        "outputs": {
            "integrity": str(OUT / "s5_card_integrity.csv"),
            "source_band": str(OUT / "s5_source_band_audit.csv"),
            "channel": str(OUT / "s5_channel_audit.csv"),
        },
    }
    (OUT / "s5_evidence_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
