# 对齐导师课题的升级方案 v2

日期：2026-11-18

## 0. 定位

本方案把研究从“SoftCLT/AdaGraph 的单点方法”升级为导师课题的三层问题：

```text
覆盖感知自监督时序表征
        ->
多尺度异常交互识别
        ->
动态风险演化与提前预警
```

现有 v5、三通道分析、覆盖采样器和欧氏表征库全部保留，但只作为第一层
“表征与机制”的证据，不再冒充整个课题。

论文暂定名：

> **Coverage-Aware Self-Supervised Temporal Representation for
> Multi-Scale Anomaly Detection and Risk Warning**

中文：

> **面向多尺度异常交互识别与风险预警的覆盖感知自监督时序表征方法**

目标不是保证 CCF-B 录用，而是保证交付一篇有完整问题、方法、实验和边界的
研究论文。

## 1. 最终科学问题

导师课题给出的三个约束是：

1. 用户行为连续、情境依赖、群体关联；
2. 异常既可能是单点突变、上下文异常，也可能是跨用户协同异常；
3. 检测必须落到风险演化和提前预警，而不是只输出一个异常分数。

因此最终研究问题是：

> 在用户行为先验不完整、关系图稀疏且训练 batch 很小的情况下，如何学习
> 可泛化的用户时序表征，并将点、上下文、集体异常证据融合为可追溯的风险过程？

## 2. 数据边界

只使用已冻结数据：

- CERT r4.2：开发、无标签选参、方向性 pilot；
- CERT r5.2：已见数据锁定复验；
- CERT r6.2：2,000/4,000 用户压力、成本、5 事件逐事件审计；
- CLUE-LDS：24 pairs/48 events，有限功效外部效应复验；
- SMD：scoring-side 外部验证；
- PSM/SMAP/MSL：连续时序 scoring robustenss。

不使用：

- LANL、OpTC、ChimeraLog；
- SPEDIA/TWOS 作为确认性证据；
- r1 作为有标签确认集；
- r6.2 五个事件作显著性检验。

## 3. 三层方法架构

### 3.1 第一层：覆盖感知自监督表征

骨干：TS2Vec。

保留 v5 的：

- 时序编码器；
- instance-soft / temporal contrast；
- 欧氏表征库；
- 固定预算评价。

新增诊断与修复：

- A3/A5a/A5b 共同覆盖 batch table；
- `2B-2` 非对角候选的尺度/质量/关系三通道分析；
- `A1-Cov-rownorm`、`A1-Cov-uniform`、`A1-Cov-permq`；
- 画像先验平坦性门禁；
- active pair mass 和边曝光统计。

本层输出不是一个“更强的 loss”声明，而是：

```text
表征 z_{u,t}
质量/关系/批次归因结果
可用的 S1/S2 表征库
```

### 3.2 第二层：多尺度异常检测

#### 点异常

单用户单日表征偏离正常参考：

```text
s_point(u,t) =
  distance(z_{u,t}, B_u^train)
```

其中 `B_u^train` 是训练期用户参考库，必要时与全局参考库取较小分位数。

#### 上下文异常

用固定长度窗口 `W`，比较掩码/未掩码表示，或比较窗口表示与窗口库：

```text
s_context(u,t) =
  distance(z_{u,t-W+1:t}^{mask}, z_{u,t-W+1:t}^{plain})
```

上下文异常必须保留至少两个窗口尺度，避免只捕捉单日尖峰。

#### 集体异常

用关系图邻域与群体一致性：

```text
s_collective(u,t) =
  distance(z_{u,t}, weighted_neighborhood_centroid(u,t))
```

同时计算用户相对群体的 rank、邻居同步变化和共享热点。

#### 多尺度融合

每个分数只在训练期正常数据上做 median/MAD 标准化：

```text
tilde s_k = (s_k - median_train_k) / MAD_train_k
```

融合分数：

```text
s_multi = fuse(tilde s_point, tilde s_context, tilde s_collective)
```

融合器只在训练期内部合成异常上选择；真实异常标签只用于评价。

第二层必须分别报告：

- point PR-AUC / ROC-AUC；
- context-window recall；
- collective event recall；
- 固定 K=5/10/20 事件命中；
- 首次命中延迟。

### 3.3 第三层：风险演化与预警

把日异常分数转成风险状态：

```text
R_{u,t} =
  alpha * R_{u,t-1}
  + (1-alpha) * s_multi(u,t)
```

增加群体风险传播：

```text
R_group(u,t) = max_{v in N(u)} R_{v,t}
R_final(u,t) = beta * R_{u,t} + (1-beta) * R_group(u,t)
```

候选参数只在训练期内部合成异常上选择，不在测试事件上调参。

预警队列：

- 每天在活跃用户中取 Top-K；
- 同一用户冷却 7 天；
- 设置低/中/高分级；
- 保存分数来源、主导通道、邻居证据、首次升级时间和复核状态。

第三层评价：

- 事件召回；
- 预警提前量；
- 首次命中延迟；
- 每用户误报告警；
- 风险曲线稳定性；
- 每用户复盘成本；
- 成功和失败案例。

## 4. 创新点

### 创新点 1：实际曝光关系表征

形式化完整图与训练中实际实现图的差异，并将覆盖门禁纳入自监督训练。
这不是增加一个图模块，而是修正“图上定义了但从未进入梯度”的实现混杂。

### 创新点 2：三通道机制分解

对官方 SoftCLT 实例软损失做尺度/质量/关系分解：

```text
l_a = W_a H(t_a, p_a)
```

并用 `rownorm/uniform/permuq/A0-A3batch` 控制实验识别收益来源。

三通道恒等式本身是分析工具，不单独当作算法创新。

### 创新点 3：多尺度证据融合

统一点异常、上下文异常和集体异常的定义、打分和证据输出，而不是只报告
单一点级 PR-AUC。

### 创新点 4：动态风险演化与预警

把异常分数变成积累、传播、衰减的风险过程，并以提前量、误报负担和复核成本
评价，而不是只以单日排序评价。

### 创新点 5：边界与复现

明确报告：

- 画像先验平坦；
- 权重未逐行归一化；
- CLUE 外部功效不足；
- r5.2 非盲测；
- r6.2 五事件只作压力审计。

## 5. v5 在新论文中的位置

| v5 内容 | 新论文位置 |
| --- | --- |
| TS2Vec 表征 | 第一层骨干 |
| instance-soft | 第一层机制审计 |
| 欧氏表征库 | 第二层 scoring 主路径 |
| 固定 K、冷却期 | 第三层运营评价 |
| CERT/SMD/CLUE 工程 | 数据与复现基础 |
| PAI/magG/T2 | 组件审计和基线 |
| v5 结果 | R0 对照与 preliminary evidence |

因此 v5 不浪费，但不再作为“完整创新论文”的唯一主体。

## 6. RQ 与实验矩阵

### RQ1：覆盖感知表征是否比随机 batch 更可解释？

比较：

```text
A0-official
A0-matched
A0-A3batch
A1-Cov-rownorm
A1-Cov-uniform
A1-Cov-permq
A1-Cov
```

### RQ2：三通道中是哪一通道解释 v5 的收益？

报告：

- `W_a` 分布；
- `pi_a` 分布；
- `KL(q_a || u)` 和行熵；
- 回归斜率和 R²；
- 逐 seed 效应。

### RQ3：多尺度检测是否优于单点检测？

比较：

- point only；
- context only；
- collective only；
- point+context；
- point+context+collective；
- 欧氏库 / 余弦库 / mask-only。

### RQ4：风险演化是否改善提前预警？

比较：

- 瞬时分数；
- 指数积累；
- 有衰减积累；
- 有群体传播；
- 分级阈值和冷却策略。

### RQ5：在现有数据上是否稳健且可复现？

- r4.2 10 seeds；
- r5.2 锁定复验；
- CLUE pair-cluster 有限复验；
- r6.2 2k/4k 压力、成本和失败案例；
- SMD/PSM/SMAP/MSL scoring-side 泛化。

## 7. 基线

至少包括：

- TS2Vec；
- SoftCLT T3/T4；
- TNC；
- TS-TCC；
- NNCLR-style；
- FNC/Incremental False Negative Detection 适配；
- Anomaly Transformer；
- DCdetector；
- OmniAnomaly；
- iForest（非时序基线）。

无法公平运行的基线必须标记为适配版本，不得冒称官方实现。

## 8. 统计与判停

- 10 个正式 seed；
- 先计算 seed 配对差；
- 用户/事件簇 bootstrap；
- 单侧 alpha=0.025；
- 预注册 Holm 家族；
- 报均值、标准差、效应量和 95% CI；
- 不显著只写“当前数据不可区分”；
- r6.2 五事件不进入显著性；
- CLUE 只报告效应、区间和 MDE。

### Gate A：表征与覆盖

- 三图覆盖率 `>=80%`；
- 零曝光用户为 0；
- batch hash 随 seed 不同；
- 实际训练长度不少于覆盖表长度。

### Gate B：机制

- A1-Cov 与 rownorm/uniform/permuq 至少有一组可解释差异；
- A0-A3batch 必须区分批次携带关系；
- 若画像先验平坦，画像机制记 `METHOD_NOT_SUPPORTED`。

### Gate C：多尺度

- 多尺度至少一个指标优于 point-only；
- 集体异常必须有独立结果，不藏在总 PR-AUC 中；
- 事件命中不能靠牺牲所有误报成本获得。

### Gate D：风险预警

- 风险模型相对瞬时分数提高提前量或降低误报；
- 至少报告一个失败事件；
- r6.2 的成本和延迟可复现。

## 9. 结果对应的三种论文写法

### A. 正向结果

覆盖感知 + 三通道 + 多尺度 + 风险演化均有稳定收益，按 CCF-B 方法/实证论文投。

### B. 机制中性结果

关系通道不可识别，但覆盖、质量/尺度和多尺度风险框架成立，写成诊断与边界研究。

### C. 负结果

覆盖和机制差异均不显著，写成“小批次关系软对比的失效条件与评测修正”，降级到 CCF-C/方法学期刊，不伪装 B 类正向结果。

## 10. 执行顺序

1. 实现 `rownorm`、`permuq` 和逐锚点 `uniform`；
2. 生成所有数据的 `W/pi/q`、熵、KL、覆盖统计；
3. r4.2 两 seed pilot，只做控制分解；
4. pilot 通过后 r4.2 10 seed；
5. 锁定参数跑 r5.2；
6. CLUE 两 seed 有限复验；
7. r6.2、SMD/PSM/SMAP/MSL 压力与 scoring 泛化；
8. 补深度基线和安全/运营实验；
9. 写论文。

## 11. 最低可交付包

即使正向性能不出现，最终仍交付：

- 边曝光理论和覆盖协议；
- 三通道公式与数值验证；
- 现有数据上的完整机制分解；
- 点/上下文/集体多尺度评价；
- 风险积累、传播、衰减和提前量；
- r6.2 成本和失败案例；
- 可复现 artifact 和明确边界声明。

这就是本方案的“保证工作”定义。
