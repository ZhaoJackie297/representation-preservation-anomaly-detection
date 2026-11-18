from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
INPUT = OUT / "s4_evidence_cards_v2ms_k10.csv"
BAND_ORDER = {"watch": 0, "elevated": 1, "critical": 2}


def trend_label(slope: float) -> str:
    if slope > 0.05:
        return "rising"
    if slope < -0.05:
        return "falling"
    return "stable"


def build_timeline(group: pd.DataFrame) -> dict:
    group = group.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(group["date"])
    days = (dates - dates.min()).dt.days.to_numpy(
        dtype=float
    )
    severity = group["severity"].to_numpy(dtype=float)
    if len(group) >= 2 and np.ptp(days) > 0:
        slope = float(np.polyfit(days, severity, 1)[0])
    else:
        slope = 0.0
    gaps = np.diff(np.unique(days))
    first_band = str(group.iloc[0]["queue_band"])
    last_band = str(group.iloc[-1]["queue_band"])
    band_escalated = (
        BAND_ORDER[last_band] > BAND_ORDER[first_band]
    )
    band_deescalated = (
        BAND_ORDER[last_band] < BAND_ORDER[first_band]
    )
    source_counts = Counter(group["source"].astype(str))
    channel_counts = Counter(group["top_channel_1"].astype(str))
    dominant_channel = (
        channel_counts.most_common(1)[0][0]
        if channel_counts
        else None
    )
    hazard_values = group["hazard_3d"].dropna()
    events = []
    for row in group.itertuples(index=False):
        events.append(
            {
                "date": pd.Timestamp(row.date).strftime(
                    "%Y-%m-%d"
                ),
                "rank": int(row.rank),
                "queue_band": str(row.queue_band),
                "source": str(row.source),
                "reason_code": str(row.reason_code),
                "severity": float(row.severity),
                "top_channels": [
                    str(row.top_channel_1),
                    str(row.top_channel_2),
                ],
                "representation_note": str(
                    row.representation_note
                ),
                "hazard_3d": (
                    None
                    if pd.isna(row.hazard_3d)
                    else float(row.hazard_3d)
                ),
                "explanation": str(row.explanation),
            }
        )
    return {
        "data_block": str(group.iloc[0]["data_block"]),
        "user_id": str(group.iloc[0]["user_id"]),
        "alert_count": int(len(group)),
        "first_alert_date": dates.min().strftime("%Y-%m-%d"),
        "last_alert_date": dates.max().strftime("%Y-%m-%d"),
        "span_days": int(days.max()),
        "mean_gap_days": (
            float(np.mean(gaps)) if len(gaps) else 0.0
        ),
        "max_severity": float(severity.max()),
        "mean_severity": float(severity.mean()),
        "last_severity": float(severity[-1]),
        "severity_slope_per_day": slope,
        "trend": trend_label(slope),
        "first_band": first_band,
        "last_band": last_band,
        "max_band": max(
            group["queue_band"].astype(str),
            key=lambda band: BAND_ORDER[band],
        ),
        "band_escalated": bool(band_escalated),
        "band_deescalated": bool(band_deescalated),
        "source_counts": dict(source_counts),
        "dominant_top_channel": dominant_channel,
        "max_hazard_3d": (
            float(hazard_values.max())
            if len(hazard_values)
            else None
        ),
        "timeline": events,
    }


def main() -> None:
    cards = pd.read_csv(
        INPUT,
        parse_dates=["date"],
        dtype={"user_id": str},
    )
    timelines = []
    for _, group in cards.groupby(
        ["data_block", "user_id"],
        sort=False,
    ):
        timelines.append(build_timeline(group))
    summary_rows = []
    for timeline in timelines:
        summary_rows.append(
            {
                key: value
                for key, value in timeline.items()
                if key != "timeline"
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUT / "s4_user_timeline_summary_v2ms.csv",
        index=False,
        encoding="utf-8",
    )
    with (
        OUT / "s4_user_timelines_v2ms.jsonl"
    ).open("w", encoding="utf-8") as stream:
        for timeline in timelines:
            stream.write(
                json.dumps(
                    timeline, ensure_ascii=False
                )
                + "\n"
            )

    examples = []
    for block_name in (
        "r42_original",
        "r42_retrained",
        "r52_external",
    ):
        block_summary = summary[
            summary["data_block"] == block_name
        ].sort_values(
            ["alert_count", "max_severity"],
            ascending=False,
        )
        selected_users = block_summary.head(2)["user_id"].tolist()
        for user in selected_users:
            examples.append(
                next(
                    timeline
                    for timeline in timelines
                    if timeline["data_block"] == block_name
                    and timeline["user_id"] == user
                )
            )

    lines = [
        "# S4 用户级风险时间线示例",
        "",
        "日期：2026-11-11",
        "",
        "说明：时间线描述模型进入队列的历史，不代表恶意事实。",
        "",
    ]
    for timeline in examples:
        lines.extend(
            [
                f"## {timeline['user_id']} / "
                f"{timeline['data_block']}",
                "",
                f"- 告警次数：{timeline['alert_count']}",
                f"- 首次/最近：{timeline['first_alert_date']} "
                f"→ {timeline['last_alert_date']}",
                f"- 趋势：{timeline['trend']}",
                f"- 最大severity：{timeline['max_severity']:.3f}",
                f"- 主要来源：{timeline['source_counts']}",
                f"- 最强通道：{timeline['dominant_top_channel']}",
                f"- 最大未来3天hazard："
                f"{timeline['max_hazard_3d'] if timeline['max_hazard_3d'] is not None else '不适用'}",
                "",
            ]
        )
        for event in timeline["timeline"]:
            hazard = (
                "NA"
                if event["hazard_3d"] is None
                else f"{event['hazard_3d']:.3%}"
            )
            lines.append(
                f"- {event['date']}：第{event['rank']}名，"
                f"{event['queue_band']}，{event['source']}，"
                f"severity={event['severity']:.3f}，"
                f"通道={'/'.join(event['top_channels'])}，"
                f"hazard={hazard}。"
            )
        lines.append("")
    (OUT / "s4_user_timeline_examples.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    summary_json = {
        "input": str(INPUT),
        "user_block_timelines": int(len(timelines)),
        "cards": int(len(cards)),
        "recurrent_users_ge_3_alerts": int(
            (summary["alert_count"] >= 3).sum()
        ),
        "escalating_users": int(
            (summary["trend"] == "rising").sum()
        ),
        "blocks": {
            block: {
                "users": int(
                    (
                        summary["data_block"] == block
                    ).sum()
                ),
                "alerts": int(
                    cards["data_block"].eq(block).sum()
                ),
            }
            for block in (
                "r42_original",
                "r42_retrained",
                "r52_external",
            )
        },
        "labels_in_timeline": False,
        "outputs": {
            "summary": str(
                OUT / "s4_user_timeline_summary_v2ms.csv"
            ),
            "timelines": str(
                OUT / "s4_user_timelines_v2ms.jsonl"
            ),
            "examples": str(
                OUT / "s4_user_timeline_examples.md"
            ),
        },
    }
    (OUT / "s4_user_timeline_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
