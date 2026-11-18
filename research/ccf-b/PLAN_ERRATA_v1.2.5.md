# CCF-B 计划勘误 v1.2.5

日期：2026-11-18。

本勘误记录 CLUE-LDS 数据门禁与真实图 Phase 0 的实测结果。

详细结果见 `CLUE_LDS_PHASE0_RESULTS.md`。

## 1. 数据状态升级

CLUE-LDS 已完成：

- `clue.zip` 下载；
- 官方 MD5 校验；
- `clue.json` 解压与 SHA-256；
- 50,522,931 条事件全量 DuckDB 审计；
- 固定 seed 的 20 对账号劫持注入；
- 838 用户、1,911 天、52 维用户日数据；
- 画像先验；
- 组/分享辅助关系图；
- A3/A5a/A5b 真实图 10-seed 覆盖；
- 一个真实用户子集的软权重训练 step。

数据状态从：

```text
pending_download_and_protocol_audit
```

升级为：

```text
core_data_and_phase0_coverage_passed_full_external_matrix_pending
```

这不等于论文外部有效性已经成立。

## 2. 真实覆盖结果

838 用户、top-m=10：

| 指标 | 结果 |
| --- | ---: |
| A3 覆盖 | 85.95%–88.25% |
| A5a 覆盖 | 80.01%–80.08% |
| A5b 覆盖 | 85.49%–86.88% |
| 覆盖 batch | 948–961 |
| 零曝光用户 | 0 |
| 不同批次表 | 10/10 |
| 相邻 seed Jaccard | 0.2184–0.2216 |

结论：

- 80% 真实图覆盖可行；
- A5a 是覆盖瓶颈；
- 名义 520 batch 不够，真实 `B_stage` 至少约 948；
- A5b 覆盖率必须单独报告，不能省略。

## 3. 画像先验风险

CLUE-LDS 画像先验：

- 105 维画像；
- 归一化行熵均值 5.9308；
- 中位数 6.4906；
- 最大均匀熵 6.7298。

因此 `P` 接近平坦，top-m 关系仍有排序信息，但直接软权重信息较弱。

新增要求：

- 每个 stage 报告 `P_m` 行熵；
- 报告 beta 分布和实际 KL；
- 若 `P_m` 接近 `ln(m)`，明确标记“画像锚信息弱”；
- 不得把 top-m 的稳定性能自动解释为强画像先验。

## 4. 组织关系图边界

训练期组/分享图：

- 非零关系边 2,164；
- 连通用户 270；
- 孤立用户 568 / 838。

因此：

- 组/分享图只作辅助关系参照；
- 不把它直接当主画像先验；
- 主图仍使用行为画像；
- 不可用组织关系覆盖不足解释主方法全部收益或失败。

## 5. 训练器接线

新增 `CoverageSoftTS2Vec`：

- 显式 batch table 训练；
- 不再随机 shuffle；
- 同批用户对才读取软权重；
- 每步记录 `active_pair_mass`；
- 支持逐阶段优化器重置。

CPU 真实子集 smoke：

```text
16 users x 256 days x 52 features
realized pairs = 240
active pair mass = 16.0
loss = 5.898293
```

只证明接线正确，不证明检测收益。

## 6. 当前 Gate 状态

已通过：

- 外部数据访问和哈希；
- schema；
- 用户日；
- 标签注入；
- 画像图；
- A3/A5a/A5b 10-seed 覆盖；
- soft loss 接线 smoke。

尚未通过：

- A1-Cov vs A0-matched pilot；
- 训练期内部异常选参；
- 完整 A3/A4 训练；
- 固定预算 Top-K 评价；
- 外部确认统计；
- 与 R0 的端到端优势。

## 7. 下一步

顺序固定：

1. 为 CLUE-LDS 建立 A1-R0、A1-Cov、A0-matched 的训练入口；
2. 完成训练期内部合成异常验证；
3. 先跑 1 个 seed 的 A1-Cov vs A0-matched；
4. 若双终点未达到最小正向效应，停止 AdaGraph；
5. 若通过，再进入 A3/A4 和 10-seed；
6. CERT r4.2 pilot 仍按原计划独立执行。
