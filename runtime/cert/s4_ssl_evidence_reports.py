from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
BASE_CARDS = OUT / "s4_evidence_cards_v2ms_k10.csv"
DATASET = ROOT / "processed_r42" / "r4.2_sequence_dataset.npz"
R52_DATASET = ROOT / "processed_r52" / "r5.2_sequence_dataset.npz"
PLAIN = {
    "r42_original": OUT / "e17_r42_M0_plain.npy",
    "r42_retrained": OUT / "e19_r42_retrained_plain.npy",
    "r52_external": OUT / "e17_r52_M0_plain.npy",
}
TRAIN_END = {
    "r42_original": 158,
    "r42_retrained": 158,
    "r52_external": 103,
}


def cosine_distance_rows(
    query: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    query_norm = np.linalg.norm(query, axis=-1)
    reference_norm = np.linalg.norm(reference, axis=-1)
    denominator = np.maximum(
        query_norm[:, None] * reference_norm[None, :],
        1e-8,
    )
    similarity = (query @ reference.T) / denominator
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def build_ssl_features(
    block_name: str,
    users: np.ndarray,
    dates: pd.DatetimeIndex,
    active_mask: np.ndarray,
    cards: pd.DataFrame,
) -> dict[tuple[str, str], dict]:
    plain = np.load(PLAIN[block_name]).astype(np.float32)
    user_index = {
        str(user): index for index, user in enumerate(users)
    }
    date_index = {
        date.strftime("%Y-%m-%d"): index
        for index, date in enumerate(dates)
    }
    train_end = TRAIN_END[block_name]
    block_cards = cards[cards["data_block"] == block_name]
    requested: dict[int, set[int]] = {}
    for row in block_cards.itertuples(index=False):
        user = user_index[str(row.user_id)]
        time = date_index[
            pd.Timestamp(row.date).strftime("%Y-%m-%d")
        ]
        requested.setdefault(user, set()).add(time)

    nearest = {}
    velocity = {}
    curvature = {}

    for user, target_times in requested.items():
        train_times = np.flatnonzero(
            active_mask[user, : train_end + 1]
        )
        if len(train_times) == 0:
            continue
        references = plain[user, train_times]
        active_times = np.flatnonzero(active_mask[user])
        position_map = {
            int(time): position
            for position, time in enumerate(active_times)
        }
        for time_index in target_times:
            query = plain[user, time_index]
            distances = cosine_distance_rows(
                query[None, :], references
            )
            nearest[(user, time_index)] = float(
                distances.min()
            )
            position = position_map[time_index]
            if position == 0:
                continue
            previous = active_times[position - 1]
            first_velocity = (
                plain[user, time_index] - plain[user, previous]
            )
            velocity[(user, time_index)] = float(
                np.linalg.norm(first_velocity)
            )
            if position >= 2:
                previous_previous = active_times[position - 2]
                second_velocity = (
                    plain[user, previous]
                    - plain[user, previous_previous]
                )
                denominator = max(
                    float(
                        np.linalg.norm(first_velocity)
                        * np.linalg.norm(second_velocity)
                    ),
                    1e-8,
                )
                cosine = float(
                    np.dot(first_velocity, second_velocity)
                    / denominator
                )
                curvature[(user, time_index)] = (
                    0.5 * (1.0 - np.clip(cosine, -1.0, 1.0))
                )

    output = {}
    for row in block_cards.itertuples(index=False):
        user = user_index[str(row.user_id)]
        date_text = pd.Timestamp(row.date).strftime(
            "%Y-%m-%d"
        )
        time = date_index[date_text]
        components = {
            "masked_reconstruction": float(
                row.masked_quantile
            ),
            "personal_manifold": float(
                row.manifold_quantile
            ),
            "channel_counterfactual": float(
                row.top_channel_1_score
            ),
        }
        high_count = sum(
            value > 3.0 for value in components.values()
        )
        dominant = max(
            components, key=components.get
        )
        output[
            (str(row.data_block), str(row.user_id), date_text)
        ] = {
            "ssl_nearest_normal_distance": (
                None
                if (user, time) not in nearest
                else float(nearest[(user, time)])
            ),
            "ssl_velocity_l2": (
                None
                if (user, time) not in velocity
                else float(velocity[(user, time)])
            ),
            "ssl_curvature": (
                None
                if (user, time) not in curvature
                else float(curvature[(user, time)])
            ),
            "ssl_masked_strength": components[
                "masked_reconstruction"
            ],
            "ssl_manifold_strength": components[
                "personal_manifold"
            ],
            "ssl_counterfactual_strength": components[
                "channel_counterfactual"
            ],
            "ssl_high_view_count": int(high_count),
            "ssl_dominant_view": dominant,
            "ssl_view_agreement": bool(high_count >= 2),
        }
    return output


def main() -> None:
    cards = pd.read_csv(
        BASE_CARDS,
        parse_dates=["date"],
        dtype={"user_id": str},
    )
    info = {
        "r42_original": np.load(
            DATASET, allow_pickle=False
        ),
        "r42_retrained": np.load(
            DATASET, allow_pickle=False
        ),
        "r52_external": np.load(
            R52_DATASET, allow_pickle=False
        ),
    }
    block_features = {}
    for block_name, dataset in info.items():
        users = dataset["users"].astype(str)
        dates = pd.to_datetime(dataset["dates"].astype(str))
        active = dataset["active_mask"][:, :, 0].astype(bool)
        block_features.update(
            build_ssl_features(
                block_name, users, dates, active, cards
            )
        )
    feature_rows = []
    for row in cards.itertuples(index=False):
        key = (
            str(row.data_block),
            str(row.user_id),
            pd.Timestamp(row.date).strftime("%Y-%m-%d"),
        )
        fields = block_features.get(key, {})
        feature_rows.append(fields)
    features = pd.DataFrame(feature_rows)
    enriched = pd.concat(
        [cards.reset_index(drop=True), features],
        axis=1,
    )

    def explain(row: pd.Series) -> str:
        view_names = {
            "masked_reconstruction": "遮罩重建",
            "personal_manifold": "个人正常流形",
            "channel_counterfactual": "通道反事实",
        }
        dominant = view_names.get(
            str(row.get("ssl_dominant_view", "")),
            "多视图",
        )
        agreement = (
            "多个自监督视图同时偏高"
            if bool(row.get("ssl_view_agreement", False))
            else "仅部分自监督视图偏高"
        )
        return (
            f"主导自监督视图为{dominant}；{agreement}。"
            f"通道{row['top_channel_1']}和"
            f"{row['top_channel_2']}的遮挡影响最大；"
            "证据是模型解释，不是因果结论。"
        )

    enriched["ssl_evidence_explanation"] = enriched.apply(
        explain, axis=1
    )
    enriched.to_csv(
        OUT / "s4_ssl_evidence_cards_v2ms_k10.csv",
        index=False,
        encoding="utf-8",
    )

    sample = []
    for block in (
        "r42_original",
        "r42_retrained",
        "r52_external",
    ):
        block_data = enriched[
            enriched["data_block"] == block
        ]
        sample.append(
            block_data[
                block_data["queue_band"] == "critical"
            ].head(2)
        )
    sample = pd.concat(sample, ignore_index=True)
    lines = [
        "# S4 自监督证据卡样例",
        "",
        "日期：2026-11-11",
        "",
        "说明：自监督视图描述模型表征，不是恶意事实。",
        "",
    ]
    for row in sample.itertuples(index=False):
        hazard = (
            "NA"
            if pd.isna(row.hazard_3d)
            else f"{row.hazard_3d:.3%}"
        )
        lines.extend(
            [
                f"## {pd.Timestamp(row.date).strftime('%Y-%m-%d')} / "
                f"{row.user_id} / "
                f"{row.data_block}",
                "",
                f"- 队列：第{row.rank}名，{row.queue_band}",
                f"- 来源：{row.source}",
                f"- SSL主导视图：{row.ssl_dominant_view}",
                f"- 高视图数量：{row.ssl_high_view_count}",
                f"- 最近正常表示距离："
                f"{row.ssl_nearest_normal_distance}",
                f"- 表示速度：{row.ssl_velocity_l2}",
                f"- 表示曲率：{row.ssl_curvature}",
                f"- 通道：{row.top_channel_1}、"
                f"{row.top_channel_2}",
                f"- 3天hazard：{hazard}",
                f"- 解释：{row.ssl_evidence_explanation}",
                "",
            ]
        )
    (OUT / "s4_ssl_evidence_samples.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    summary = {
        "model": "V2-M-S + self-supervised evidence",
        "rows": int(len(enriched)),
        "blocks": {
            block: int(
                (
                    enriched["data_block"] == block
                ).sum()
            )
            for block in (
                "r42_original",
                "r42_retrained",
                "r52_external",
            )
        },
        "high_view_count_distribution": {
            str(int(key)): int(value)
            for key, value in enriched[
                "ssl_high_view_count"
            ].value_counts().sort_index().items()
        },
        "view_agreement_rate": float(
            enriched["ssl_view_agreement"].mean()
        ),
        "dominant_view_counts": enriched[
            "ssl_dominant_view"
        ].value_counts().to_dict(),
        "labels_in_cards": False,
        "outputs": {
            "cards": str(
                OUT / "s4_ssl_evidence_cards_v2ms_k10.csv"
            ),
            "samples": str(
                OUT / "s4_ssl_evidence_samples.md"
            ),
        },
        "limitations": [
            "主导视图是自监督表征归因，不是因果结论",
            "最近正常距离只在训练期参考集中计算",
            "轨迹证据不改变S3排序",
            "hazard仅r4.2原模型提供",
        ],
    }
    (OUT / "s4_ssl_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
