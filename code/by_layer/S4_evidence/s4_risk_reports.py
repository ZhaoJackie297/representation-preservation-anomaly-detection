from __future__ import annotations

import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
QUEUE_FILE = OUT / "s3_v2m_s_risk_queue_k10.csv"
CHANNELS = {
    "logon": "channel_logon_quantile",
    "device": "channel_device_quantile",
    "file": "channel_file_quantile",
    "email": "channel_email_quantile",
    "http": "channel_http_quantile",
}

sys.path.insert(0, str(ROOT))
from feature_augmented_iforest_sim import load_user_day_frame  # noqa: E402
from quota_relationship_sim import (  # noqa: E402
    load_r42_cached,
    load_r52_cached,
)


def load_block_evidence(
    block_name: str,
    data: dict,
) -> dict:
    if block_name == "r42_original":
        masked_path = OUT / "e17_r42_M0_masked_quantile.npy"
        manifold_paths = [
            OUT
            / f"e19_r42_learned_manifold_quantile_{seed}.npy"
            for seed in (20261111, 20261112, 20261113)
        ]
        channel_key = "r42_original_channel_"
    elif block_name == "r42_retrained":
        masked_path = (
            OUT / "e19_r42_retrained_masked_quantile.npy"
        )
        manifold_paths = [
            OUT
            / (
                "e19_r42_retrained_learned_manifold_"
                f"quantile_{seed}.npy"
            )
            for seed in (20261111, 20261112, 20261113)
        ]
        channel_key = "r42_retrained_channel_"
    else:
        masked_path = OUT / "e17_r52_M0_masked_quantile.npy"
        manifold_paths = [
            OUT
            / f"e19_r52_learned_manifold_quantile_{seed}.npy"
            for seed in (20261111, 20261112, 20261113)
        ]
        channel_key = "r52_external_channel_"
    masked = np.load(masked_path)
    manifold = np.mean(
        np.stack([np.load(path) for path in manifold_paths]),
        axis=0,
    )
    channel_values = load_channel_quantiles(channel_key)
    hazard = None
    if block_name == "r42_original":
        hazard = np.load(
            OUT / "s3p_hazard_r42_original_components.npz",
            allow_pickle=False,
        )["hazard"]
    return {
        "data": data,
        "users": data["users"].astype(str),
        "dates": pd.to_datetime(data["dates"]),
        "masked": masked,
        "manifold": manifold,
        "channels": channel_values,
        "hazard": hazard,
    }


@lru_cache(maxsize=3)
def load_channel_quantiles(
    channel_key: str,
) -> dict[str, np.ndarray]:
    """Load one block's five channel arrays once per process."""

    with np.load(
        OUT / "channel_evidence_arrays.npz",
        allow_pickle=False,
    ) as store:
        return {
            name: store[channel_key + name + "_quantile"]
            for name in CHANNELS
        }


def feature_lookup(
    data: dict,
    path: Path,
) -> dict[tuple[str, str], dict]:
    frame = load_user_day_frame(path).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    wanted = [
        "observed_event_count",
        "after_hours_logon_count",
        "other_pc_logon_count",
        "usb_connect_count",
        "file_copy_count",
        "external_recipient_count",
        "after_hours_email_count",
        "external_http_count",
    ]
    columns = [
        column
        for column in wanted
        if column in frame.columns
    ]
    frame = frame[["user_id", "date", *columns]].copy()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return {
        (str(row.user_id), row.date): {
            column: float(getattr(row, column))
            for column in columns
            if pd.notna(getattr(row, column))
        }
        for row in frame.itertuples(index=False)
    }


def main() -> None:
    queue = pd.read_csv(
        QUEUE_FILE,
        dtype={"user_id": str},
    )
    families = {
        "r42": load_r42_cached(),
        "r52": load_r52_cached(),
    }
    feature_files = {
        "r42": (
            ROOT / "processed_r42" / "r4.2_user_day_features.csv"
        ),
        "r52": (
            ROOT / "processed_r52" / "r5.2_user_day_features.csv"
        ),
    }
    blocks = {}
    features = {
        "r42": feature_lookup(
            families["r42"], feature_files["r42"]
        ),
        "r52": feature_lookup(
            families["r52"], feature_files["r52"]
        ),
    }
    for block_name, family in (
        ("r42_original", "r42"),
        ("r42_retrained", "r42"),
        ("r52_external", "r52"),
    ):
        blocks[block_name] = load_block_evidence(
            block_name, families[family]
        )
        blocks[block_name]["feature_lookup"] = features[family]

    rows = []
    for row in queue.itertuples(index=False):
        block = blocks[str(row.data_block)]
        user = str(row.user_id)
        date_text = str(row.date)
        user_index = int(
            np.flatnonzero(block["users"] == user)[0]
        )
        time_index = int(
            np.flatnonzero(
                block["dates"].strftime("%Y-%m-%d") == date_text
            )[0]
        )
        channel_scores = {
            name: float(
                block["channels"][name][
                    user_index, time_index
                ]
            )
            for name in CHANNELS
        }
        top_channels = sorted(
            channel_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        masked = float(block["masked"][user_index, time_index])
        manifold = float(
            block["manifold"][user_index, time_index]
        )
        evidence_gap = abs(masked - manifold)
        if evidence_gap < 0.1:
            representation_note = "遮罩与用户流形接近"
        elif masked > manifold:
            representation_note = "遮罩差异更强"
        else:
            representation_note = "用户流形偏离更强"

        source = str(row.source)
        if source == "ts_main":
            reason_code = "TS_MAIN"
            reason = (
                "进入severity主路前K-q名。"
            )
        elif source == "iforest_rescue":
            reason_code = "IFOREST_RESCUE"
            reason = (
                "不在severity Top-K，"
                "由候选自身对齐IForest补位。"
            )
        else:
            reason_code = "TS_FALLBACK"
            reason = (
                "IForest独有候选不足，"
                "名额回退到severity主路。"
            )

        raw_features = block["feature_lookup"].get(
            (user, date_text), {}
        )
        hazard = None
        if block["hazard"] is not None:
            hazard = float(
                block["hazard"][user_index, time_index]
            )
        risk_band = str(row.queue_band)
        if risk_band == "critical":
            priority_note = "队列前三名，优先人工复核"
        elif risk_band == "elevated":
            priority_note = "队列中段，按审查容量处理"
        else:
            priority_note = "队列尾部，可用于观察趋势"

        evidence = {
            "data_block": row.data_block,
            "split": row.split,
            "date": date_text,
            "rank": int(row.rank),
            "user_id": user,
            "source": source,
            "reason_code": reason_code,
            "queue_band": risk_band,
            "severity": float(row.severity),
            "severity_percentile": float(row.risk_percentile),
            "triple_quantile": float(row.triple_quantile),
            "masked_quantile": masked,
            "manifold_quantile": manifold,
            "representation_gap": evidence_gap,
            "representation_note": representation_note,
            "top_channel_1": top_channels[0][0],
            "top_channel_1_score": top_channels[0][1],
            "top_channel_2": top_channels[1][0],
            "top_channel_2_score": top_channels[1][1],
            "memory": float(row.memory),
            "frequency": float(row.frequency),
            "iforest_score": float(row.iforest_score),
            "hazard_3d": hazard,
            "threshold_level": int(row.threshold_level),
            "priority_note": priority_note,
            "explanation": (
                f"{reason} "
                f"{representation_note}；"
                f"最强通道为{top_channels[0][0]}和"
                f"{top_channels[1][0]}。"
                "这是排序证据，不是因果结论。"
            ),
            "raw_observed_event_count": raw_features.get(
                "observed_event_count"
            ),
            "raw_after_hours_logon_count": raw_features.get(
                "after_hours_logon_count"
            ),
            "raw_file_copy_count": raw_features.get(
                "file_copy_count"
            ),
            "raw_external_recipient_count": raw_features.get(
                "external_recipient_count"
            ),
            "raw_external_http_count": raw_features.get(
                "external_http_count"
            ),
        }
        rows.append(evidence)

    cards = pd.DataFrame(rows)
    cards.to_csv(
        OUT / "s4_evidence_cards_v2ms_k10.csv",
        index=False,
        encoding="utf-8",
    )

    sample_rows = []
    for block_name in (
        "r42_original",
        "r42_retrained",
        "r52_external",
    ):
        block_cards = cards[cards["data_block"] == block_name]
        sample_rows.append(block_cards.head(2))
        rescue = block_cards[
            block_cards["source"] == "iforest_rescue"
        ].head(2)
        sample_rows.append(rescue)
    sample = pd.concat(sample_rows, ignore_index=True)

    lines = [
        "# S4 风险证据卡样例",
        "",
        "日期：2026-11-11",
        "",
        "模型：V2-M-S（severity-only风险）",
        "",
        "说明：下列证据卡只展示模型排序和可观察证据，不包含离线标签，"
        "不把证据解释为因果结论。",
        "",
    ]
    for row in sample.itertuples(index=False):
        hazard_text = (
            f"{row.hazard_3d:.3%}"
            if pd.notna(row.hazard_3d)
            else "未提供（该数据块未运行hazard模型）"
        )
        lines.extend(
            [
                f"## {row.date} / {row.user_id} / "
                f"{row.data_block}",
                "",
                f"- 队列：第{int(row.rank)}名，{row.queue_band}",
                f"- 来源：{row.source}（{row.reason_code}）",
                f"- severity：{row.severity:.4f}",
                f"- 三分数：masked={row.masked_quantile:.3f}，"
                f"manifold={row.manifold_quantile:.3f}",
                f"- 最强通道：{row.top_channel_1}"
                f"={row.top_channel_1_score:.3f}，"
                f"{row.top_channel_2}"
                f"={row.top_channel_2_score:.3f}",
                f"- 解释：{row.explanation}",
                f"- 未来3天hazard：{hazard_text}",
                "",
            ]
        )
    (OUT / "s4_evidence_card_samples.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    summary = {
        "model": "V2-M-S + S3 evidence",
        "primary_k": 10,
        "cooldown_days": 7,
        "rows": int(len(cards)),
        "labels_in_cards": False,
        "source_counts": cards["source"].value_counts().to_dict(),
        "queue_band_counts": cards["queue_band"].value_counts().to_dict(),
        "top_channel_counts": {
            block_name: Counter(
                cards[
                    cards["data_block"] == block_name
                ]["top_channel_1"]
            ).most_common()
            for block_name in (
                "r42_original",
                "r42_retrained",
                "r52_external",
            )
        },
        "hazard_coverage": {
            block_name: int(
                cards[
                    cards["data_block"] == block_name
                ]["hazard_3d"].notna().sum()
            )
            for block_name in (
                "r42_original",
                "r42_retrained",
                "r52_external",
            )
        },
        "outputs": {
            "evidence_cards": str(
                OUT / "s4_evidence_cards_v2ms_k10.csv"
            ),
            "samples": str(
                OUT / "s4_evidence_card_samples.md"
            ),
        },
        "limitations": [
            "证据是排序归因，不是因果结论",
            "hazard只对r4.2原模型提供",
            "离线标签没有写入证据卡",
            "r5.2仅作外部敏感性",
        ],
    }
    (OUT / "s4_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
