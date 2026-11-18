# CCF-B 计划勘误 v1.2.6

日期：2026-11-18。

本勘误在 v1.2.5 数据和覆盖结果基础上，冻结先验信息阈值、标签独立性、CLUE-LDS 功效和 pilot 判据。

## 1. 平坦先验判定规则

正式定义 `P_m` 行熵：

```text
H_m(u) = -sum_{v in S_m(u)} P_m(u,v) log P_m(u,v)
```

对每个 m，均匀上界为 `log(m)`。

在查看 A3/A5b 性能前冻结：

```text
flat_threshold = 0.95 * log(m)
```

满足以下任一条件，则标记 `FLAT_PROFILE_PRIOR`：

1. `mean_u H_m(u) > flat_threshold`；
2. 至少 80% 用户满足 `H_m(u) > flat_threshold`。

采用双条件是因为 CLUE-LDS 的熵呈双峰：

- 平均熵 2.1689，略低于 m=10 的阈值 2.1875；
- 但 89.98% 用户高于该阈值；
- 只用均值会漏掉“多数用户平坦、少数高度尖锐”的混合分布。

若 `FLAT_PROFILE_PRIOR` 触发：

- A3/A5b 的“画像锚是否有机制贡献”结论记为 **不可检验**；
- 不记为 A3 失败；
- A3/A4 的端到端性能比较仍可运行；
- 必须在结果表并列报告 `P_m` 熵、beta 和实际 KL。

CLUE-LDS 当前已触发该标记。

## 2. 标签协议措辞

统一改写为：

> 基于官方切换算法实现，由本项目固定 seed 构造的半合成账号劫持。

不再使用：

- “官方受控注入”；
- “官方异常标签”；
- “真实账号劫持攻击”。

CLUE-LDS 原始数据只有正常行为；标签由本项目执行切换算法产生。

## 3. 标签 seed 与训练 seed 分离

冻结：

```text
label_seed = 20261101
training_seeds = 20261116..20261125
```

标签 seed 不参与模型训练、调参或批次表生成。

## 4. 注入相似度的独立性

账号劫持 pair 的选择距离是：

```text
0.7 * action Jaccard
+ 0.3 * average-events-per-active-day ratio
```

它只使用原始事件类型和活跃日统计。

画像先验 `P` 使用：

- 52 维用户日特征；
- median/q90；
- 活跃比例；
- 标准化后的欧氏距离；
- 高斯核。

两者不是同一个距离。因此当前 pair 选择与画像先验不构成直接循环依赖。

为防止审稿人仍认为两者相关，补充：

- 审计 pair similarity 与 F4 画像距离的 Spearman 相关；
- 若 `|rho| >= 0.5`，必须增加独立度量生成对照或重新生成标签；
- 该审计只看距离关系，不看模型结果。

正式审计结果：

```text
unique_pairs = 24
Spearman(label similarity, profile affinity) = -0.1991304347826087
status = PASS_DISTANCES_DISTINCT
```

复现命令和明细见 `clue_lds/audit_label_prior_independence.py` 与 `clue_lds/label_prior_independence_20261118.json`。

## 5. 5,389 到 838 用户筛选规则

处理规则冻结为：

1. 只保留 `uidType=name`；
2. 所有 48 个注入目标用户必须保留；
3. 其余用户必须至少在训练期出现一次；
4. 非目标用户按训练期事件数降序；
5. 最多保留 1,000 用户；
6. 实际只有 838 个 name 用户满足训练期活动条件，因此得到 838；
7. 任何注入目标用户训练期全空则整体失败，不回填。

该筛选不使用异常标签、评价期检测结果或 A4 输出。

## 6. CLUE-LDS 事件与聚类

冻结标签协议下最多存在 24 个合格用户对。

正式数据使用：

```text
24 pairs
48 user-level events
```

主统计单位：

- 注入用户对；
- 一个 pair 的两个目标事件作为同一簇；
- bootstrap 整对重采样；
- 事件级重采样只作敏感性。

20-pair 的 40 事件只作为旧开发结果，不进入确认性统计。

## 7. CLUE-LDS 功效

在 24 个有效 pair 簇、80% 功效、单侧 alpha 0.025 下：

```text
MDE(recall delta) = 22.97-29.53 percentage points
```

即使忽略当前标签门禁，理论上：

| pair clusters | MDE 范围 |
| ---: | ---: |
| 40 | 17.6–24.7 pp |
| 60 | 14.4–21.4 pp |
| 100 | 10.9–17.9 pp |

因此：

- CLUE-LDS 对 5–10 个百分点的小/中等效应功效不足；
- 不显著不能直接解释为方法失败；
- CLUE-LDS 的定位调整为“有限功效外部效应复验”；
- B 类结论不能仅依赖 CLUE-LDS 的 p 值；
- 详细模拟见 `CLUE_LDS_POWER_ANALYSIS.md`。

## 8. CLUE-LDS pilot 最小效应

CLUE-LDS 上不得沿用 CERT 的 `0.0005` PR-AUC 门槛。

冻结：

### PR-AUC endpoint

先用 A0-matched 得到 `PR0`，再定义：

```text
positive_PR =
  Delta_PR >= max(0.001, 0.05 * PR0)
```

该阈值在使用 A1/A3/A4 结果前计算。

### 事件召回 endpoint

24 pairs 下共有 48 个事件：

```text
positive_recall =
  Delta_recall >= 3 / 48 = 0.0625
```

即至少 3 个事件。

### 双 seed 方向

CLUE pilot 必须使用 2 个独立训练 seed：

- 两个 seed 在同一 endpoint 上都必须为正；
- 两个 seed 的均值必须达到该 endpoint 的阈值；
- 单 seed 只作实现 smoke，不能停止或继续 A3/A4；
- pilot 不用于调参。

停止：

```text
若 A1-Cov vs A0-matched 在两个 endpoint 上都未达到 positive threshold，
则停止 CLUE 上的 AdaGraph 主实验。
```

只有一个 endpoint 达标时：

- 标为 `FRAGILE_SINGLE_ENDPOINT`；
- 不以 CLUE 单数据宣布胜利；
- 仍可在 CERT r4.2 上做独立方向性 pilot。

## 9. 当前数据重生成

v1.2.6 使用：

```text
label_seed=20261101
pairs=24
events=48
```

CLUE 用户日、画像、A3/A5a/A5b 图和覆盖结果已按新标签重新生成。最终处理产物的 SHA-256 清单见 `clue_lds/artifact_hashes_20261118.json`。

旧 `label_seed=20261116 / 20 pairs` 结果只保留为开发记录。

## 10. 论文解释边界

若 CLUE-LDS 结果不显著：

- 不写“AdaGraph 无效”；
- 写“24 个有效簇下数据只能可靠排除约 23–30 pp 或更大的召回差异”；
- 同时报告效应量、区间、方向一致性和 MDE；
- 结合 r5.2、原 CERT pilot、LANL 和机制结果作整体判断。

若先验被标记为平坦：

- 不写“A5b 证明画像无用”；
- 写“在当前 `P_m` 信息量下，A3/A5b 机制差异不可检验”。

## 11. 当前冻结后的下一步

文档和审计冻结后，按以下顺序推进：

1. 实现 `A0-matched / A1-Cov / A1-R0` 的完整训练入口；
2. 在训练期内部生成三类合成异常并完成无标签选参验证；
3. 用 2 个独立训练 seed 跑 CLUE pilot；
4. 只有 pilot 判据满足后才运行 A3/A4 的外部矩阵；
5. 不以 pilot 结果调参，也不把 CLUE pilot 改写成最终外部确认。
