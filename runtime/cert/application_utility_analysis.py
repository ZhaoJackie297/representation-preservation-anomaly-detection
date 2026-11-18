from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"

# 应用情景权重在查看本轮结果前固定。
# 三个分量依次为：
# 1. 3 天事件召回：漏掉真实风险事件的代价；
# 2. 3 天场景宏平均：不同攻击场景是否都能覆盖；
# 3. 账户日 Precision@K：审查员每天看到的候选有多大比例有意义。
PROFILES = {
    "impact_first": {
        "event_recall_3d": 0.60,
        "scenario_macro_recall_3d": 0.25,
        "precision_at_k": 0.15,
    },
    "balanced": {
        "event_recall_3d": 0.50,
        "scenario_macro_recall_3d": 0.25,
        "precision_at_k": 0.25,
    },
    "workload_first": {
        "event_recall_3d": 0.40,
        "scenario_macro_recall_3d": 0.25,
        "precision_at_k": 0.35,
    },
}

# 实际部署通常先确定每日审查容量，而不是把所有 K 混在一起。
# K=10 是主预算；K=20 用于审查能力临时翻倍的敏感性检查。
FOCAL_BUDGETS = (10, 20)
PRIMARY_BUDGET = 10
SECONDARY_BUDGET = 20


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    rename = {
        "variant": "method",
    }
    frame = frame.rename(columns=rename)
    return frame


def normalize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].replace(
        {
            "r5.2": "r5.2_external",
            "r5.2_original_external": "r5.2_external",
        }
    )
    return frame


def collect_metrics() -> pd.DataFrame:
    frames = []

    base = normalize_columns(
        pd.read_csv(OUT / "quota_relationship_final_metrics.csv")
    )
    base = base[
        base["method"].isin(["ts_only", "iforest_only", "quota_model"])
    ].copy()
    base["dataset"] = base["dataset"].map(
        {"r4.2": "r4.2_original", "r5.2": "r5.2_external"}
    )
    base["method"] = base["method"].map(
        {
            "ts_only": "complete_ts2vec",
            "iforest_only": "complete_iforest",
            "quota_model": "fixed_q_0.4K",
        }
    )
    frames.append(base)

    retrain_base = pd.read_csv(
        ROOT
        / "processed_r42"
        / "innovation_sim"
        / "r4.2_quota_retrain_metrics.csv"
    )
    retrain_base["dataset"] = "r4.2_retrained"
    retrain_base["method"] = retrain_base["method"].map(
        {
            "ts_only": "complete_ts2vec",
            "iforest_only": "complete_iforest",
            "quota_model": "fixed_q_0.4K",
        }
    )
    frames.append(retrain_base)

    adaptive = pd.read_csv(
        OUT / "adaptive_count_quota_metrics.csv"
    )
    adaptive = normalize_dataset(adaptive)
    adaptive["method"] = "adaptive_q_count"
    frames.append(adaptive)

    augmented = pd.read_csv(
        OUT / "feature_augmented_iforest_final_metrics.csv"
    )
    augmented["method"] = "feature_augmented_iforest"
    augmented["dataset"] = augmented["split"].map(
        {
            "validation": "r4.2_original",
            "test": "r4.2_original",
            "external_test": "r5.2_external",
        }
    )
    frames.append(augmented)

    augmented_retrain = pd.read_csv(
        ROOT
        / "processed_r42"
        / "innovation_sim"
        / "feature_augmented_iforest_retrain_metrics.csv"
    )
    augmented_retrain["dataset"] = "r4.2_retrained"
    augmented_retrain["method"] = "feature_augmented_iforest"
    frames.append(augmented_retrain)

    ensemble = pd.read_csv(
        OUT / "iforest_view_ensemble_final_metrics.csv"
    )
    ensemble["dataset"] = "r4.2_original"
    external_mask = ensemble["split"] == "external_test"
    retrain_mask = ensemble["method"].str.startswith("retrained_")
    ensemble.loc[external_mask, "dataset"] = "r5.2_external"
    ensemble.loc[retrain_mask, "dataset"] = "r4.2_retrained"
    ensemble["method"] = ensemble["method"].replace(
        {
            "iforest_view_max": "two_view_max",
            "retrained_iforest_view_max": "two_view_max",
        }
    )
    frames.append(ensemble)

    raw_multiseed = pd.read_csv(
        OUT / "multiseed_raw_iforest_metrics.csv"
    )
    raw_multiseed = normalize_dataset(raw_multiseed)
    raw_multiseed["method"] = "raw_multiseed_adaptive"
    frames.append(raw_multiseed)

    columns = [
        "dataset",
        "split",
        "method",
        "k",
        "event_recall_3d",
        "event_recall_1d",
        "precision_at_k",
        "false_alerts_per_day",
        "scenario_macro_recall_3d",
    ]
    combined = pd.concat(
        [frame[columns] for frame in frames],
        ignore_index=True,
    )
    keys = ["dataset", "split", "method", "k"]
    duplicated = combined[combined.duplicated(keys, keep=False)]
    if not duplicated.empty:
        metric_columns = [
            "event_recall_3d",
            "event_recall_1d",
            "precision_at_k",
            "false_alerts_per_day",
            "scenario_macro_recall_3d",
        ]
        conflicting = duplicated.groupby(keys)[metric_columns].nunique(
            dropna=False
        )
        conflicting = conflicting[(conflicting > 1).any(axis=1)]
        if not conflicting.empty:
            raise ValueError(
                "同一评估单元出现互相冲突的重复指标：\n"
                f"{conflicting.to_string()}"
            )
    return combined.drop_duplicates(keys, keep="first")


def add_profile_utilities(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    for profile_name, weights in PROFILES.items():
        metrics[profile_name] = sum(
            metrics[column] * weight
            for column, weight in weights.items()
        )
    return metrics


def summarize_average(
    metrics: pd.DataFrame,
    ks: tuple[int, ...] | None,
) -> pd.DataFrame:
    selected = metrics
    if ks is not None:
        selected = metrics[metrics["k"].isin(ks)]
    return (
        selected.groupby(["dataset", "split", "method"], as_index=False)
        .agg(
            mean_event_recall_3d=("event_recall_3d", "mean"),
            mean_event_recall_1d=("event_recall_1d", "mean"),
            mean_precision_at_k=("precision_at_k", "mean"),
            mean_false_alerts_per_day=(
                "false_alerts_per_day",
                "mean",
            ),
            mean_scenario_macro=(
                "scenario_macro_recall_3d",
                "mean",
            ),
            impact_first=("impact_first", "mean"),
            balanced=("balanced", "mean"),
            workload_first=("workload_first", "mean"),
        )
    )


def build_rankings(
    average: pd.DataFrame,
    dataset: str,
    split: str,
    budget_name: str,
) -> list[dict[str, object]]:
    ranked_rows = []
    group = average[
        (average["dataset"] == dataset) & (average["split"] == split)
    ]
    for profile_name in PROFILES:
        sorted_group = group.sort_values(
            [profile_name, "mean_precision_at_k"],
            ascending=False,
        ).copy()
        sorted_group["rank"] = np.arange(1, len(sorted_group) + 1)
        for row in sorted_group.itertuples(index=False):
            ranked_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "budget": budget_name,
                    "profile": profile_name,
                    "method": row.method,
                    "utility": float(getattr(row, profile_name)),
                    "rank": int(row.rank),
                    "event_recall_3d": float(row.mean_event_recall_3d),
                    "scenario_macro_recall_3d": float(
                        row.mean_scenario_macro
                    ),
                    "precision_at_k": float(row.mean_precision_at_k),
                }
            )
    return ranked_rows


def summarize_rank_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    stability = (
        rankings.groupby("method", as_index=False)
        .agg(
            appearances=("rank", "size"),
            first_place=("rank", lambda values: int((values == 1).sum())),
            top_three=("rank", lambda values: int((values <= 3).sum())),
            mean_rank=("rank", "mean"),
            worst_rank=("rank", "max"),
            mean_utility=("utility", "mean"),
            min_utility=("utility", "min"),
        )
        .sort_values(
            ["mean_rank", "appearances"],
            ascending=[True, False],
        )
    )
    stability["top_three_rate"] = (
        stability["top_three"] / stability["appearances"]
    )
    return stability


def summarize_regret(rankings: pd.DataFrame) -> pd.DataFrame:
    frame = rankings.copy()
    best_by_block = frame.groupby(
        ["dataset", "split", "profile"]
    )["utility"].transform("max")
    frame["relative_utility"] = frame["utility"] / best_by_block
    frame["regret"] = 1.0 - frame["relative_utility"]
    regret = (
        frame.groupby("method", as_index=False)
        .agg(
            mean_relative_utility=("relative_utility", "mean"),
            worst_relative_utility=("relative_utility", "min"),
            mean_regret=("regret", "mean"),
            worst_regret=("regret", "max"),
            mean_rank=("rank", "mean"),
            top_three_rate=(
                "rank",
                lambda values: float((values <= 3).mean()),
            ),
        )
        .sort_values(
            ["mean_regret", "worst_regret"],
            ascending=[True, True],
        )
    )
    return regret


def frame_to_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records = frame.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, np.generic):
                record[key] = value.item()
    return records


def main() -> None:
    metrics = add_profile_utilities(collect_metrics())
    average = summarize_average(metrics, ks=None)
    primary = summarize_average(metrics, ks=(PRIMARY_BUDGET,))
    secondary = summarize_average(metrics, ks=(SECONDARY_BUDGET,))
    focal = summarize_average(metrics, ks=FOCAL_BUDGETS)

    metrics.to_csv(
        OUT / "application_utility_per_k.csv",
        index=False,
        encoding="utf-8",
    )
    average.to_csv(
        OUT / "application_utility_summary.csv",
        index=False,
        encoding="utf-8",
    )
    primary.to_csv(
        OUT / "application_utility_k10_summary.csv",
        index=False,
        encoding="utf-8",
    )
    secondary.to_csv(
        OUT / "application_utility_k20_summary.csv",
        index=False,
        encoding="utf-8",
    )
    focal.to_csv(
        OUT / "application_utility_focal_budgets_summary.csv",
        index=False,
        encoding="utf-8",
    )

    ranking_frames = []
    for budget_name, summary_frame in (
        ("K10", primary),
        ("K20", secondary),
        ("K10_K20", focal),
    ):
        for (dataset, split) in summary_frame[
            ["dataset", "split"]
        ].drop_duplicates().itertuples(index=False):
            ranking_frames.extend(
                build_rankings(
                    summary_frame,
                    dataset,
                    split,
                    budget_name,
                )
            )
    rankings = pd.DataFrame(ranking_frames)
    rankings.to_csv(
        OUT / "application_utility_rankings.csv",
        index=False,
        encoding="utf-8",
    )

    # 主稳定性结论只看 K=10，避免把实际部署不会采用的大预算
    # 和主预算混在一起，把排名“平均得更好看”。
    primary_rankings = rankings[rankings["budget"] == "K10"]
    stability = summarize_rank_stability(primary_rankings)
    stability.to_csv(
        OUT / "application_utility_k10_rank_stability.csv",
        index=False,
        encoding="utf-8",
    )
    k10_regret = summarize_regret(primary_rankings)
    k10_regret.to_csv(
        OUT / "application_utility_k10_regret.csv",
        index=False,
        encoding="utf-8",
    )

    secondary_rankings = rankings[rankings["budget"] == "K20"]
    k20_regret = summarize_regret(secondary_rankings)
    k20_regret.to_csv(
        OUT / "application_utility_k20_regret.csv",
        index=False,
        encoding="utf-8",
    )

    k10_order_by_profile = (
        primary_rankings.sort_values(
            ["dataset", "split", "profile", "rank"]
        )
        .groupby(["dataset", "split", "profile"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {"order": " > ".join(group["method"])}
            ),
            include_groups=False,
        )
    )
    profile_orders = (
        k10_order_by_profile.groupby(["dataset", "split"])["order"]
        .nunique()
        .to_dict()
    )
    profile_rankings_identical = all(
        count == 1 for count in profile_orders.values()
    )

    summary = {
        "profiles": PROFILES,
        "instruction": (
            "Weights are fixed application scenarios and are not optimized "
            "on validation, test, or external labels. K10 is the primary "
            "operating budget; K20 is a sensitivity check; the K grid is "
            "not used for model selection."
        ),
        "primary_budget": PRIMARY_BUDGET,
        "secondary_budget": SECONDARY_BUDGET,
        "k10_rankings": frame_to_records(
            primary_rankings.sort_values(
                [
                    "dataset",
                    "split",
                    "profile",
                    "rank",
                ]
            )
        ),
        "k10_rank_stability": frame_to_records(stability),
        "k10_regret": frame_to_records(k10_regret),
        "profile_rankings_identical_at_k10": (
            profile_rankings_identical
        ),
        "outputs": {
            "per_k": str(OUT / "application_utility_per_k.csv"),
            "summary": str(OUT / "application_utility_summary.csv"),
            "k10_summary": str(
                OUT / "application_utility_k10_summary.csv"
            ),
            "k20_summary": str(
                OUT / "application_utility_k20_summary.csv"
            ),
            "focal_budgets_summary": str(
                OUT / "application_utility_focal_budgets_summary.csv"
            ),
            "rankings": str(OUT / "application_utility_rankings.csv"),
            "k10_rank_stability": str(
                OUT / "application_utility_k10_rank_stability.csv"
            ),
            "k10_regret": str(
                OUT / "application_utility_k10_regret.csv"
            ),
            "k20_regret": str(
                OUT / "application_utility_k20_regret.csv"
            ),
        },
    }
    (OUT / "application_utility_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("K=10 应用效用排名（每个数据块的前三名）", flush=True)
    top_three = primary_rankings[primary_rankings["rank"] <= 3]
    print(
        top_three.sort_values(
            ["dataset", "split", "profile", "rank"]
        ).to_string(index=False),
        flush=True,
    )
    print("\nK=10 跨数据块与情景的排名稳定性", flush=True)
    print(stability.to_string(index=False), flush=True)
    print("\nK=10 相对最优方案的平均与最差遗憾", flush=True)
    print(k10_regret.to_string(index=False), flush=True)
    print(
        "\n三种固定应用权重下，各数据块的方法排序是否完全一致："
        f"{profile_rankings_identical}",
        flush=True,
    )


if __name__ == "__main__":
    main()
