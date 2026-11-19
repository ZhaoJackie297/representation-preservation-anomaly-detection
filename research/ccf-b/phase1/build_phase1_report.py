from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-dir", type=Path, required=True)
    parser.add_argument("--r42-coverage-dir", type=Path, required=True)
    parser.add_argument("--clue-coverage-dir", type=Path, required=True)
    parser.add_argument("--risk-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fmt(value: object, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def compact_manifest(manifest: dict[str, object]) -> dict[str, object]:
    output = {
        key: value
        for key, value in manifest.items()
        if key not in {"selected_users", "target_users"}
    }
    if "selected_users" in manifest:
        output["selected_user_count"] = len(manifest["selected_users"])
    if "target_users" in manifest:
        output["target_user_count"] = len(manifest["target_users"])
    return output


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    digits: int = 6,
) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(fmt(value, digits=digits) for value in row)
            + " |"
        )
    return lines


def treatment_table(
    summary: pd.DataFrame,
    control: str,
    treatment: str,
) -> pd.DataFrame:
    selected = summary[
        (summary["score_type"] == "point")
        & summary["base_method"].isin([control, treatment])
    ].copy()
    columns = [
        "signal_type",
        "strength",
        "pr_auc_mean",
        "event_recall_k10_mean",
    ]
    left = selected[selected["base_method"] == control][columns].rename(
        columns={
            "pr_auc_mean": "control_pr",
            "event_recall_k10_mean": "control_k10",
        }
    )
    right = selected[selected["base_method"] == treatment][columns].rename(
        columns={
            "pr_auc_mean": "treatment_pr",
            "event_recall_k10_mean": "treatment_k10",
        }
    )
    pivot = left.merge(
        right,
        on=["signal_type", "strength"],
        validate="one_to_one",
    )
    output = pd.DataFrame(
        {
            "signal": pivot["signal_type"],
            "strength": pivot["strength"],
            f"{control}_pr": pivot["control_pr"],
            f"{treatment}_pr": pivot["treatment_pr"],
            "delta_pr": (
                pivot["treatment_pr"] - pivot["control_pr"]
            ),
            f"{control}_k10": pivot["control_k10"],
            f"{treatment}_k10": pivot["treatment_k10"],
            "delta_k10": (
                pivot["treatment_k10"] - pivot["control_k10"]
            ),
        }
    )
    return output.sort_values(["signal", "strength"], kind="stable")


def context_table(
    pairs: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    selected = pairs[
        pairs["comparison"].isin(
            [
                f"{method}_context_minus_point",
                f"{method}_combined_minus_point",
            ]
        )
        & (pairs["metric"].isin(["pr_auc", "event_recall_k10"]))
    ].copy()
    if selected.empty:
        return selected
    selected["aggregation"] = selected["comparison"].str.extract(
        r"_(context|combined)_minus_point"
    )[0]
    return (
        selected.groupby(
            ["signal_type", "aggregation", "metric"],
            as_index=False,
        )
        .agg(
            mean_delta=("mean_delta", "mean"),
            positive_seed_configs=("positive_seeds", "sum"),
            n_seed_configs=("n", "sum"),
        )
        .sort_values(
            ["signal_type", "aggregation", "metric"],
            kind="stable",
        )
    )


def treatment_stats(
    pairs: pd.DataFrame,
    comparison: str,
    *,
    metric: str,
    score_type: str,
) -> dict[str, float | int]:
    selected = pairs[
        (pairs["comparison"] == comparison)
        & (pairs["metric"] == metric)
        & (pairs["score_type"] == score_type)
    ]
    return {
        "settings": int(len(selected)),
        "positive_settings": int((selected["mean_delta"] > 0).sum()),
        "all_seed_positive_settings": int(
            (selected["positive_seeds"] == selected["n"]).sum()
        ),
        "mean_delta": float(selected["mean_delta"].mean()),
    }


def main() -> None:
    args = parse_args()
    temporal_summary = pd.read_csv(args.temporal_dir / "summary.csv")
    temporal_pairs = pd.read_csv(
        args.temporal_dir / "paired_comparisons.csv"
    )
    r42_summary = pd.read_csv(args.r42_coverage_dir / "summary.csv")
    r42_pairs = pd.read_csv(
        args.r42_coverage_dir / "paired_comparisons.csv"
    )
    clue_summary = pd.read_csv(args.clue_coverage_dir / "summary.csv")
    clue_pairs = pd.read_csv(
        args.clue_coverage_dir / "paired_comparisons.csv"
    )
    risk_summary = pd.read_csv(args.risk_dir / "summary.csv")
    risk_pairs = pd.read_csv(args.risk_dir / "paired_vs_instant.csv")
    manifests = {
        "temporal": compact_manifest(
            json.loads(
                (args.temporal_dir / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        ),
        "r42_coverage": compact_manifest(
            json.loads(
                (args.r42_coverage_dir / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        ),
        "clue_coverage": compact_manifest(
            json.loads(
                (args.clue_coverage_dir / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        ),
    }

    temporal = treatment_table(
        temporal_summary,
        "T3_instance_soft",
        "T4_full_soft",
    )
    r42_coverage = treatment_table(
        r42_summary,
        "A0-official",
        "A1-Cov",
    )
    clue_coverage = treatment_table(
        clue_summary,
        "A0-official",
        "A1-Cov",
    )
    temporal_context = context_table(
        temporal_pairs,
        "T4_full_soft",
    )
    clue_context = context_table(
        clue_pairs,
        "A1-Cov",
    )
    temporal_pr = treatment_stats(
        temporal_pairs,
        "T4_full_soft_minus_T3_instance_soft",
        metric="pr_auc",
        score_type="point",
    )
    temporal_k10 = treatment_stats(
        temporal_pairs,
        "T4_full_soft_minus_T3_instance_soft",
        metric="event_recall_k10",
        score_type="point",
    )
    r42_coverage_pr = treatment_stats(
        r42_pairs,
        "A1-Cov_minus_A0-official",
        metric="pr_auc",
        score_type="point",
    )
    clue_coverage_pr = treatment_stats(
        clue_pairs,
        "A1-Cov_minus_A0-official",
        metric="pr_auc",
        score_type="point",
    )
    baselines = summary_baselines(temporal_summary, clue_summary)

    lines = [
        "# Phase 1 受控异常与风险实验结果",
        "",
        "日期：2026-11-19",
        "",
        "## 1. 证据定位",
        "",
        "本阶段使用 CERT r4.2 与 CLUE-LDS 的真实正常行为作为母体，",
        "构造固定 seed 的点尖峰、持续偏移、缓慢漂移和账号切换事件。",
        "数据 seed 与模型 seed 分离，模型没有在注入事件上训练或选参。",
        "",
        "**这是受控机制实验，不是新的真实数据集，也不是外部验证。**",
        "",
        "## 2. Temporal-soft 桥接",
        "",
        "主比较为 `T4_full_soft - T3_instance_soft`，同一事件、同一强度、",
        "同一评分方式。表中使用 point residual 分数。",
        "",
        *markdown_table(
            temporal,
            [
                "signal",
                "strength",
                "delta_pr",
                "delta_k10",
            ],
        ),
        "",
        f"合计：{temporal_pr['positive_settings']}/"
        f"{temporal_pr['settings']} 个信号-强度设置的 PR-AUC 差值为正，"
        f"{temporal_pr['all_seed_positive_settings']} 个设置为 3/3 seeds 同向，"
        f"平均差值 {temporal_pr['mean_delta']:.6f}。K=10 只有 "
        f"{temporal_k10['positive_settings']} 个设置为正，"
        f"{temporal_k10['settings'] - temporal_k10['positive_settings']}"
        " 个为非正。",
        "",
        "解释：T4 在持续偏移和缓慢漂移上的 PR-AUC 差值方向稳定为正，",
        "但 K=10 事件召回没有同步提高。点尖峰上 PR-AUC 也可能提高，",
        "固定预算事件召回却出现负向。因此只能支持“temporal-soft 改变",
        "了持续信号的局部几何”，不能支持“temporal-soft 已有稳定",
        "端到端固定预算收益”。",
        "",
        "## 3. r4.2 Coverage 对照",
        "",
        *markdown_table(
            r42_coverage,
            [
                "signal",
                "strength",
                "delta_pr",
                "delta_k10",
            ],
        ),
        "",
        f"r4.2 受控点级 PR-AUC：{r42_coverage_pr['positive_settings']}/"
        f"{r42_coverage_pr['settings']} 个信号-强度设置为正，"
        f"{r42_coverage_pr['all_seed_positive_settings']} 个设置为 3/3 "
        f"seeds 同向，平均差值 {r42_coverage_pr['mean_delta']:.6f}。",
        "",
        "这种近乎全正的受控结果，加上后文 iForest 的强表现，说明注入器更",
        "偏向高幅度、单行可分的表格信号。它可用于识别边界，不能作为",
        "A1-Cov 已证明优于官方训练的确认性证据。",
        "",
        "## 4. CLUE 受控 Coverage 对照",
        "",
        *markdown_table(
            clue_coverage,
            [
                "signal",
                "strength",
                "delta_pr",
                "delta_k10",
            ],
        ),
        "",
        f"CLUE 受控点级 PR-AUC：{clue_coverage_pr['positive_settings']}/"
        f"{clue_coverage_pr['settings']} 个设置为正，"
        f"{clue_coverage_pr['all_seed_positive_settings']} "
        "个设置为 3/3 seeds 同向，"
        f"平均差值 {clue_coverage_pr['mean_delta']:.6f}。",
        "",
        "CLUE 受控数据上 A1-Cov 的点级指标多数提高。该方向与真实",
        "CLUE-LDS 10-seed 的 Coverage 负结果相反，说明当前注入器主要",
        "制造了高幅度、低歧义、tabular 可分的信号，不能据此推翻真实负结果。",
        "",
        "## 5. 点级、上下文和组合信号",
        "",
        "### 5.1 T4 temporal 模型",
        "",
        *markdown_table(
            temporal_context,
            [
                "signal_type",
                "aggregation",
                "metric",
                "mean_delta",
                "positive_seed_configs",
                "n_seed_configs",
            ],
        ),
        "",
        "### 5.2 CLUE A1-Cov",
        "",
        *markdown_table(
            clue_context,
            [
                "signal_type",
                "aggregation",
                "metric",
                "mean_delta",
                "positive_seed_configs",
                "n_seed_configs",
            ],
        ),
        "",
        "结论：不存在一个对全部信号都最优的分数。点信号应由 point",
        "残差主导；持续偏移和漂移的 PR-AUC 可由 context/combined 提高，",
        "但对孤立点使用上下文平滑会降低排序和召回。",
        "",
        "## 6. 统一 iForest 基线",
        "",
        *markdown_table(
            baselines,
            [
                "dataset",
                "signal",
                "iForest_pr",
                "best_neural_pr",
                "best_neural_method",
                "iForest_k10",
                "best_neural_k10",
            ],
        ),
        "",
        "iForest 在受控的高幅度点尖峰和漂移上经常强于神经表征。",
        "这直接说明注入器不能作为“深度方法优越”的证据；它是基线",
        "敏感性审计，而不是新主结果。",
        "",
        "## 7. S3 风险控制",
        "",
        *markdown_table(
            risk_summary,
            [
                "method",
                "pr_auc_mean",
                "roc_auc_mean",
                "event_recall_3d_mean",
                "median_contact_time_days",
                "false_alerts_per_day_mean",
            ],
        ),
        "",
        "相对 instant 的配对结果：",
        "",
        *markdown_table(
            risk_pairs[
                risk_pairs["metric"].isin(
                    [
                        "event_recall_3d",
                        "median_contact_time_days",
                        "roc_auc",
                        "false_alerts_per_day",
                    ]
                )
            ],
            [
                "method",
                "metric",
                "mean_delta",
                "positive_seeds",
                "negative_seeds",
                "n",
            ],
        ),
        "",
        "EWMA 和 CUSUM 明显损伤 3 天事件召回。单步传播仅带来很小的",
        "ROC/Precision 变化，事件召回不提高，提前量仅约 0.15 天，",
        "误报下降远低于 10%。S3 的运营收益门槛未通过。",
        "",
        "## 8. S4 案例",
        "",
        "固定规则生成的成功/失败案例位于：",
        "",
        "`research/ccf-b/phase1/results/risk_cases_v1/"
        "S4_CASES_generated.md`",
        "",
        "案例只作排序证据展示，不作攻击归因。",
        "",
        "## 9. 冻结结论",
        "",
        "1. temporal-soft 的局部几何作用可以保留，端到端固定预算收益未确认；",
        "2. point/context 必须按异常形态分别报告，统一融合不成立；",
        "3. 受控 A1-Cov 正向不能覆盖真实 CLUE 负结果；",
        "4. iForest 在受控幅度信号上很强，必须在论文中如实报告；",
        "5. EWMA/CUSUM 当前没有运营收益，风险章节只能保留理论和小幅",
        "   propagation 排序结果。",
        "",
        "## 10. 运行参数",
        "",
        "```json",
        json.dumps(manifests, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


def summary_baselines(
    temporal_summary: pd.DataFrame,
    clue_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset, summary in (
        ("CERT-CAS", temporal_summary),
        ("CLUE-CAS", clue_summary),
    ):
        selected = summary[
            (summary["score_type"] == "point")
            & (summary["strength"] == 2.0)
        ]
        for signal in sorted(selected["signal_type"].unique()):
            signal_rows = selected[selected["signal_type"] == signal]
            iforest = signal_rows[signal_rows["base_method"] == "iForest"]
            neural = signal_rows[signal_rows["base_method"] != "iForest"]
            if iforest.empty or neural.empty:
                continue
            best_pr = neural.sort_values(
                "pr_auc_mean",
                ascending=False,
            ).iloc[0]
            best_k10 = neural.sort_values(
                "event_recall_k10_mean",
                ascending=False,
            ).iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "signal": signal,
                    "iForest_pr": float(iforest.iloc[0]["pr_auc_mean"]),
                    "best_neural_pr": float(best_pr["pr_auc_mean"]),
                    "best_neural_method": str(best_pr["base_method"]),
                    "iForest_k10": float(
                        iforest.iloc[0]["event_recall_k10_mean"]
                    ),
                    "best_neural_k10": float(
                        best_k10["event_recall_k10_mean"]
                    ),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
