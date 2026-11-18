# 对齐导师课题的升级方案 v3

日期：2026-11-18

本版是 `ALIGNED_UPGRADE_PLAN_v2.md` 的收缩版，回应审阅中的四个问题：

1. Gate B 必须数值化；
2. 集体异常没有清晰真实标签；
3. 融合器和风险模型不能只靠工程拼接；
4. 12 周内不能同时把三层都做大。

最终定位：

> **第一层机制研究为主，第三层风险积累理论为第二主线，第二层只保留点级和上下文级轻量扩展。**

## 1. 数据与标签事实

已核对：

- r4.2 共 70 个 incident，scenario 1/2/3 的 incident 数分别为 30/30/10；
- 每个 scenario 的 incident 基本对应不同的 `insider_id`；
- r5.2 共 99 个 incident，scenario 1/2/3/4 分别为 29/30/10/30；
- 同一 scenario 下的多个用户事件窗口可以重叠，但数据没有显式标注“跨用户协同”；
- CLUE-LDS 是 24 个构造用户对、48 个半合成目标事件；
- r6.2 只有 5 个事件。

因此：

- **不把 CERT scenario 当作 collective ground truth**；
- CLUE-LDS 的 pair 可以作为“成对半合成事件簇”，但不能写成真实协同攻击；
- 集体异常只作为探索性、人工注入或 pair-level 分析，不进入主 Gate；
- r6.2 不进入显著性检验。

## 2. 最终三层结构

### 2.1 第一层：覆盖感知自监督表征（必做）

保留 v5：

- TS2Vec；
- instance-soft；
- 欧氏表征库；
- 固定预算评价。

加入：

- A3/A5a/A5b 覆盖 batch；
- 官方 `dup_matrix` 的 `2B-2` 候选事实；
- `W_a` 尺度、`pi_a` 质量、`q_a` 关系三通道；
- `A1-Cov-rownorm`、`A1-Cov-uniform`、`A1-Cov-permq`；
- `A0-A3batch`；
- 画像先验平坦性审计。

第一层输出：

```text
用户日表征 z_{u,t}
W/pi/q 分布
三通道匹配效应
S1/S2 表征库
```

### 2.2 第二层：点级和上下文级异常（保留）

#### 点级

```text
s_point(u,t) = distance(z_{u,t}, B_u^train)
```

#### 上下文级

使用固定窗口 `W`：

```text
s_context(u,t) =
  distance(z_{u,t-W+1:t}^mask, z_{u,t-W+1:t}^plain)
```

至少比较两个窗口尺度。

#### 融合

只在训练期内部的无标签合成异常上选择融合器；真实测试标签不参与。

合成异常固定为：

- 幅度尖峰；
- 多通道突发；
- 低幅缓慢漂移；
- 每个机制 3 个强度；
- 只在训练期的 inner-validation 中生成；
- 合成器和真实事件不混用。

融合规则预注册：

```text
point-only
context-only
point + context
```

主融合端点只使用 PR-AUC 和 K=10 事件召回，其他指标为探索性。

### 2.3 集体异常（降级为探索性）

保留两种明确标注：

1. CLUE-LDS 的同一注入 pair；
2. 训练期人工构造的协同异常。

不保留：

- 把 CERT scenario 直接叫 collective；
- 用测试期关系/群体信息构造 `N(u)`；
- 让 collective 成为主 Gate。

集体异常报告只放在 Appendix / exploratory results：

- pair-level recall；
- pair-cluster bootstrap；
- 失败案例。

## 3. 第三层：风险积累与预警

### 3.1 基础模型

在训练期 median/MAD 标准化后：

```text
Z_t = alpha * s_t + (1-alpha) * Z_{t-1}
```

这是 EWMA 控制图形式。

备选 CUSUM：

```text
C_t = max(0, C_{t-1} + s_t - k)
alert when C_t >= h
```

群体传播只作为扩展：

```text
R_group(u,t) = aggregate_{v in N(u)} Z_v,t
R_final(u,t) = beta * Z_u,t + (1-beta) * R_group(u,t)
```

`N(u)` 只能由训练期画像图或训练期关系图构造，禁止使用测试期信息。

### 3.2 可检验理论

在正常分数近似零均值、异常期均值漂移 `mu`、方差近似稳定的假设下，EWMA 的
检测延迟和误报率存在经典折中。实际分数重尾且自相关，因此论文必须写：

> 理论给出高斯/独立同分布条件下的基准预测；实际偏离用训练期分数和回放事件评估。

预注册预测：

- `alpha` 增大，响应更快但误报/方差更高；
- 训练期正常分数量化阈值决定最小可接受误报率；
- 风险传播会增加邻居的基线抬升，应报告额外误报增量。

### 3.3 提前量定义

提前量统一定义为：

```text
lead_time = event_start_date - first_alert_date
```

数据规则：

- CERT：使用 incident 的 `first_observed_date`；
- CLUE-LDS：使用该 pair 的 `change_date`；
- r6.2：只报告 5 个事件逐事件结果，不作显著性；
- 没有明确起始日的数据不做提前量声称。

## 4. Gate B：数值化机制判据

Gate B 只看预先声明的主端点：

- PR-AUC；
- K=10 事件召回。

### 比较

```text
A1-Cov vs A0-matched
A1-Cov vs A1-Cov-rownorm
A1-Cov vs A1-Cov-uniform
A1-Cov vs A1-Cov-permq
A0-matched vs A0-A3batch
```

### 数值判据

对每个声称有效的机制比较，必须同时满足：

```text
Delta_PR >= max(0.0005, 0.05 * PR_baseline)
Delta_K10 >= max(0.02, 1/N_events)
```

并满足：

- 10 个 seed 中至少 8 个同向；
- 用户/事件簇配对 bootstrap 95% CI 不含 0；
- 不能只凭一个数据集的单个指标通过；
- A0-A3batch 必须作为批次组成诊断单独报告。

事件数不足时，不报告确认性 p 值，只报告效应、区间和失败事件。

### 解释

- Cov 相对 rownorm 通过：尺度通道有贡献；
- Cov 相对 uniform/permuq 通过：关系通道有贡献；
- Cov 相对 A0-A3batch 不通过：批次组成可能解释大部分收益；
- 若平坦先验仍存在：画像锚机制标 `METHOD_NOT_SUPPORTED`。

## 5. Gate C：多尺度判据

主比较：

```text
point + context vs point-only
```

在以下两个主端点中：

- 至少一个端点达到 Gate B 的数值阈值；
- 另一个端点不得劣于预注册非劣界；

其他窗口尺度、collective、附加融合策略全部标探索性。

集体异常不进入主 Gate。

## 6. Gate D：风险判据

以固定 K=10 和冷却期 7 天为预算：

- 风险模型相对瞬时分数至少提高 1 天提前量，或减少 10% 误报；
- 不能以明显增加复核成本换取；
- 失败案例必须保留；
- r6.2 只报告成本、延迟和逐事件命中，不做显著性。

如果提前量/风险曲线的收益只出现在单一数据集，结论必须标为探索性。

## 7. 实验缩减

### 必做

- 第一层全部控制：A0-official、A0-matched、A0-A3batch、A1-Cov、rownorm、uniform、permuq；
- 第一层 10 seed；
- 第二层 point-only、context-only、point+context；
- 第三层 instant、EWMA、EWMA+propagation、CUSUM 最小对照。

### 选做

- collective：只在 CLUE pair 和训练期人工注入上做；
- r6.2：2k/4k 压力、成本、5 事件；
- SMD/PSM/SMAP/MSL：只做 scoring-side 附录。

### 基线缩减为 5 个

```text
TS2Vec
SoftCLT T3/T4
NNCLR-style
Anomaly Transformer
iForest
```

其他基线只在审稿意见要求时补。

## 8. 主要实验表

### 表 1：权重和覆盖

- `W_a`、`pi_a`、`q_a`；
- 行熵、KL、active pair mass；
- A3/A5a/A5b 覆盖。

### 表 2：机制分解

- Cov vs rownorm/uniform/permuq/A0-A3batch；
- 主端点、CI、seed 方向。

### 表 3：多尺度

- point-only、context-only、point+context；
- PR、K10、窗口召回。

### 表 4：风险

- instant、EWMA、EWMA+propagation、CUSUM；
- 提前量、误报、复核成本、失败案例。

### 表 5：压力与外部

- r6.2 2k/4k；
- CLUE pair 有限复验；
- SMD/PSM/SMAP/MSL scoring-side。

## 9. 论文贡献最终写法

1. 覆盖感知自监督表征及实际边曝光诊断；
2. 尺度/质量/关系三通道机制分解与匹配对照；
3. 点级 + 上下文级多尺度异常证据（集体只作探索性）；
4. EWMA/CUSUM 风险积累、传播和预警评价；
5. 现有数据、r6.2 压力和 CLUE 有限复验的完整边界报告。

## 10. 下一步

1. rownorm、permuq 和逐 batch 逐锚点 uniform 已接入；下一步在真实数据上跑控制 smoke；
2. 生成全部数据的 W/pi/q、熵、KL、覆盖统计；
3. 实现 EWMA/CUSUM 和提前量定义；
4. 运行 r4.2 2-seed pilot；
5. pilot 通过后跑 10-seed；
6. 再跑 r5.2、CLUE、r6.2 和 scoring-side。

本方案保证的是研究可完成、结论可解释和失败可写作；不保证 A3/A4 正向，
也不保证 CCF-B 录用。
