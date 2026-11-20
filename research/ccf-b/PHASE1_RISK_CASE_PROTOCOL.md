# Phase 1：S3 风险积累与 S4 案例协议

日期：2026-11-19

## 1. 输入

只读取已经锁定的 r4.2 10-seed Coverage-PAI 分数：

```text
<path-to-cert-runtime>\external\
  pilot_r42_10seed_scores_v1\seed_*/A1-Cov_scores.npz
```

风险层不重新训练表征，不使用测试标签选择参数。用户关系只来自训练期
SoftCLT 软权重矩阵，取每行 top-10 邻接并重新归一化。

## 2. 风险模型

固定四个因果对照：

```text
instant
EWMA(alpha=0.2)
CUSUM(drift=0.5)
propagation(beta=0.2, top_m=10)
EWMA + propagation
```

用户无活动时状态清零。所有状态只依赖当前和过去分数。K=10、冷却期 7 天，
与主实验完全一致。

## 3. 评价

- 点级：PR-AUC、ROC-AUC；
- 事件级：canonical 3 天首次命中率；
- 运营：Precision@K、每活动日误报数、首次接触时间和提前量；
- 10 seeds 均报告均值和标准差。

S3 的成功门槛不因本次结果改动。只有：

```text
至少提高 1 天提前量
或减少 10% 误报
```

才可写成风险层有独立运营收益。

## 4. S4 案例

案例使用第一个配置 seed 的 `EWMA+propagation`。选择规则在看到结果前固定：

1. 按 `incident_id` 字典序排序；
2. 取前两个 3 天内命中事件；
3. 取第一个 3 天内未命中事件；
4. 无论结果好坏都保留。

证据卡只展示排序信号和标准化特征，不把模型证据写成恶意事实或因果关系。

## 5. 运行

```powershell
python research/ccf-b/phase1/risk_case_study.py `
  --output-dir "<path-to-cert-runtime>\external\phase1_risk_cases_v1"
```

输出：

```text
metrics_by_seed.csv
summary.csv
s4_case_cards.csv
S4_CASES_generated.md
protocol.json
result.json
```

## 6. 解释边界

- r4.2 test 是 development test；
- 风险结果不能替代 r5.2 锁定复验；
- S4 是固定预算排序证据，不是攻击归因；
- 如果 EWMA/CUSUM 没有达到门槛，必须如实写成未支持。

## 7. 执行结果

2026-11-19 已完成 r4.2 10-seed：

- instant：3 天事件召回 `13.9/24`；
- EWMA：`8.3/24`；
- CUSUM：`3.7/24`；
- propagation：`13.9/24`，仅 ROC 小幅提高且接触时间提前约 0.15 天；
- EWMA+propagation：`7.5/24`。

因此 EWMA/CUSUM 和 EWMA+propagation 未通过事件召回门槛。单步 propagation
只保留为排序层面的探索性结果，不写成运营收益已成立。
