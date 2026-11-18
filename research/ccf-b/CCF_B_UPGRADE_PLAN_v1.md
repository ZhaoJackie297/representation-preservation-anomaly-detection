# CCF-B 升级计划 v1

日期：2026-11-18

数据范围修正（v1.2.8）：LANL、OpTC 和 ChimeraLog 不可得，数据包已冻结；
本文计划只使用 CERT、CLUE-LDS 和 SMD/PSM/SMAP/MSL，不再把真实网络外部确认
作为执行依赖。

结论先说：**没有任何方案可以保证 CCF-B 录用**。可以确定的是：如果继续把
`AdaGraph-SoftCLT` 的画像锚当作必然有效的主创新，当前证据不足；如果改成
“覆盖感知的关系软对比学习：诊断、修复与边界”的方法学/实证论文，并把
`r5.2 + CLUE-LDS + LANL 或等价外部证据` 做成完整证据链，才是现阶段最有希望的
CCF-B 路线。

## 1. 当前成果能支撑什么，不能支撑什么

### 已支撑

- TS2Vec/SoftCLT 的随机小 batch 会让全图大部分边永远不进入实例软对比梯度；
- 共同覆盖批次表在 CERT 和 CLUE-LDS 上可以实现，CLUE 真实图需要
  948–961 个 stage batch；
- 10 个 seed 的 CLUE 批次表 hash 不同，零曝光用户为 0；
- 欧氏表征库在已有 CERT 结果中可以改善固定预算事件命中；
- 项目已有 CERT r4.2/r5.2、SMD、CLUE-LDS、PAT/PAI 组件审计和基础复现数据；
- A0/A1 训练入口、覆盖采样器、事件/用户簇评价接口已经具备；
- r4.2/r5.2/CLUE 的画像先验审计已完成。

### 已被证据否定或不能支撑

- 不能声称：画像锚在 CERT/CLUE-LDS 上已经被验证；
- 不能声称：A4 一定优于 A1+、A9、NNCLR 或 FNC；
- 不能声称：CLUE-LDS 是独立高功效外部确认；
- 不能声称：r6.2 的五事件结果具有显著性；
- 不能声称：r5.2 是盲测；
- 不能声称：只看一个数据集的小幅 PR-AUC 提升即可达到 CCF-B。

### 关键审计数字

| 审计 | 结果 | 状态 |
| --- | --- | --- |
| CLUE canonical | 24 pairs、48 user events、565 isolated users | 数据门禁通过 |
| CLUE 外部功效 | 24 个 pair，K10 MDE 约 23.0–29.5 pp | `UNDERPOWERED` |
| CERT r4.2 画像先验 | 主/top-kNN/m=5/10 均平坦 | `METHOD_NOT_SUPPORTED` |
| CERT r5.2 画像先验 | 主/top-kNN/m=5/10 均平坦 | `METHOD_NOT_SUPPORTED` |
| r5.2 99-event 严格双终点 Gate | 代理 MDE 约 6.7–15.0 pp，取决于相关性和事件簇 | `UNDERPOWERED` |

### Phase 0：软权重理论审计

官方 SoftCLT 的 `inst_CL_soft` 没有对每行权重归一化；v5 的 `densify` 权重
对角为 1、非对角通常不超过 0.5，`B=16` 时行总权重可以达到约 8.5。数值审计
表明原始权重、逐行归一化和 hard identity 会给出不同损失水平。详见
`SOFTCLT_WEIGHT_AUDIT.md`。

因此，后续实验必须先明确：

- self/quality channel 与 off-diagonal/relation channel 的质量占比；
- 是否逐行归一化；
- 目标熵和行 KL；
- 权重尺度是否被 `lambda_` 或学习率吸收。

这是第一步，不需要训练；未完成前不得把软对比收益解释成“关系信息贡献”。

### 与你提出的五步计划整合

| 你的步骤 | 在主计划中的位置 | 产物 |
| --- | --- | --- |
| 核实 SoftCLT 权重/理论 | Phase 0 | `SOFTCLT_WEIGHT_AUDIT.md` + toy JSON |
| 收窄贡献 | 主贡献与写作边界 | 三件事：质量/关系分解、批次覆盖、对照归因 |
| 无训练量描述统计 | RQ1/Q4 前置审计 | 质量占比、目标熵、行 KL、覆盖率 |
| 少量 A0/A1/Uniform/A5b pilot | Gate B | 先 r4.2 pilot，再 r5.2 锁定复验 |
| 按结果写作 | 三种叙事分支 | 关系有效、质量主导、覆盖/负结果三种稿法 |

因此五步计划不是替代 CCF-B 总计划，而是它的入口和判停层。

因此论文的核心主张必须从：

> 画像锚 AdaGraph 一定提升异常检测。

改成：

> 关系软对比的收益首先取决于关系边是否实际进入训练，以及先验是否具有
> 可识别的信息量；不满足这两个条件时，更强的图更新算法不会自动产生收益。

这不是降级，而是把负面机制结果转成可验证的科学命题。

## 2. 最终选择的研究主张

### 主标题方向

建议使用以下之一，按目标刊会调整：

1. **Realized Exposure Matters: Coverage-Aware Soft Relational Contrastive
   Learning for Temporal User Anomaly Detection**
2. **When Does Relational Contrast Help? Diagnosing Batch Realization and
   Prior Informativeness for User Behavior Anomaly Detection**

### 主贡献

1. **Diagnosis：实际曝光缺口**

   形式化定义完整关系图与训练中实际实现的图，推导随机 batch 下任意边的
   曝光概率：

   ```text
   p = b(b-1) / [N(N-1)]
   E[exposure] = n_b * p
   ```

   证明在 16 用户 batch 下，绝大部分全图边不会进入 SoftCLT 的实例损失。

2. **Method：覆盖感知训练协议**

   在每个 stage 构造 A3/A5a/A5b 联合覆盖表，设置 8 个设计槽位和 8 个随机
   填充槽位，硬门禁 80% 覆盖、零曝光用户、seed 唯一 hash，并匹配
   A0-matched、A1-Cov、A1+、A9 的实际 stage 预算。

3. **Boundary：先验可识别性与可靠性门禁**

   对画像先验计算 `P_m` 行熵，若主带宽和预注册 kNN 带宽都接近均匀，则把画像
   机制标为 `METHOD_NOT_SUPPORTED`，不允许用端到端提升替代机制证明。

   当前 CERT r4.2/r5.2 和 CLUE-LDS 都触发该边界，这一负结果是论文的一部分，
   不是隐藏的附录。

4. **Decomposition：关系内容、软权重和批次共现分离**

   至少包含：

   - A0-official；
   - A0-matched；
   - A0-A3batch；
   - A1-Cov-uniform；
   - A1-Cov-A5a；
   - A1-Cov；
   - A1-sparse vs A1-sparse-EMA；
   - A3/A4。

   这样即使 A4 没有超过 A1+，论文仍能回答“收益来自关系、软权重还是共同批次”。

5. **Scoring：固定预算下的双端异常评分**

   不把 PAI/magG 当作必然有效的核心组件。以 S2 欧氏表征库作为有证据的
   scoring path，PAI-A1/magG/T2 只作为预注册组件审计。报告 PR-AUC、ROC-AUC、
   K=5/10/20、事件命中、首次命中延迟和每用户复核成本。

### 明确不写入主贡献的内容

- “我们提出了全新的基础算法”（没有足够新基础算法证据）；
- “画像锚在 CERT/CLUE 上有效”（已触发平坦先验）；
- “CLUE-LDS 是真实攻击攻击标签”（它是本项目固定 seed 的半合成账号劫持）；
- “r5.2 是盲测”（R0 已见）；
- “A4 全面超过所有基线”（目前没有 10-seed 完整矩阵）。

## 3. 论文定位与目标刊会

### 首选：安全/应用型 CCF-B 或相当刊物

优先选择重视真实数据、运营预算、可复现性和安全评价的 venue，而不是纯 ML
方法刊：

- Computers & Security（必须先按最新 CCF 目录核实当前等级）；
- ACSAC、ESORICS、RAID 等安全会议（按最新目录和当年 track 核实）；
- 有 empirical/system track 的安全或数据管理 venue。

理由：本项目的相对优势是“覆盖实现 + 固定预算 + 可追溯证据 + 多数据审计”，
不是单点模型结构创新。

### 不作为第一选择

- 只接受纯新算法的顶会/顶刊：当前核心机制在 CERT/CLUE 上不可识别；
- 只看单数据集 PR-AUC 的数据挖掘 track；
- 依赖 CLUE-LDS 小样本显著性的安全 venue。

### 目标等级判定

| 条件 | 定位 |
| --- | --- |
| 覆盖诊断 + 分解控制 + r5.2 + 两个外部/异构数据 + 深度基线 + 安全评价 | CCF-B 冲刺 |
| 覆盖诊断 + r5.2 + CLUE 但外部功效不足 | CCF-C 上沿或方法学论文 |
| 只有实现和单数据集结果 | 不投 B |

## 4. 数据策略

### 开发与验证

- CERT r4.2：开发、无标签选参、方向性 pilot；
- CERT r5.2：锁定参数后的一次性版本复验，不称盲测；
- CERT r6.2：五事件压力审计，不进入显著性。

### 外部数据

- CLUE-LDS：canonical 24 pairs/48 events，只作有限功效外部效应复验；
- LANL：必须尽快取得 fence token，先审计 redteam 时间、事件簇、用户过滤和
  干净前缀；若没有全局干净前缀，则作 per-user local baseline 或外部压力测试；
- SMD：只验证 scoring side，不用于用户图/AdaGraph 机制；
- PSM/SMAP/MSL：只作连续时序 scoring robustness，不作跨用户关系确认。

### 最低外部证据组合

为冲 B，至少需要：

```text
CERT r4.2 (development)
+ CERT r5.2 (locked replication)
+ CLUE-LDS (limited external replication)
+ LANL or an auditable non-CERT generator family (real-network external)
```

如果 LANL 无法在截止日前取得，不要把 CLUE 的低功效结果包装成“外部确认”；
自动降级到 CCF-C 或方法学实证路线。

## 5. 方法实现路线

### Phase 0：已完成的接线

保留：

- `CommonCoverageBatchSampler`；
- `CoverageSoftTS2Vec`；
- `dual_end_entry.py`；
- CLUE canonical 数据/标签/图/覆盖；
- A0/A1 smoke；
- 事件和固定预算评价函数。

### Phase 1：补齐可区分控制

先实现并测试：

- `A0-A3batch`：硬实例负样本，批次仅由 A3 边选择；
- `A1-Cov-uniform`：A1-Cov 同批次，支撑集均匀权重；
- `A1-Cov-A5a`：A1-Cov 权重，A5a 批次；
- A1-sparse vs A1-sparse-EMA。

这些控制先于 A3/A4 大规模实验，否则无法解释批次选择效应。

### Phase 2：覆盖与训练矩阵

所有正式变体：

- `B_stage=max(B_base, required_coverage_batches)`；
- 每个 stage 重新生成覆盖表；
- A1+ 连续优化器、逐 stage 覆盖刷新；
- A9 逐 stage 优化器重置和覆盖刷新；
- 实际步数少于覆盖表长度时记为 `INVALID_COVERAGE_BUDGET`。

### Phase 3：可靠性与边界

- 记录 `P_m` 熵、beta、KL、clip fraction；
- 画像平坦时不声称 anchor 机制；
- EMA 教师必须满足 NCA@10 和有效秩质量门禁；
- `BETA_SOLVE_FAILED`、`EMA_DISTANCE_SATURATED`、
  `EMA_TEACHER_NOT_QUALIFIED` 的 run 不进入机制结论。

## 6. 预注册实验矩阵

### RQ1：随机 batch 中关系边实际曝光多少？

- r4.2、r5.2、CLUE-LDS 的 A3/A5a/A5b 覆盖率；
- 10-seed batch hash 和 Jaccard；
- 边曝光分布、零曝光用户；
- 成功门禁：每图 `>=80%`，零曝光用户为 0。

### RQ2：覆盖本身是否带来收益？

比较：

```text
A0-official -> A0-matched
```

若 A0-matched 的变化不解释训练分布变化，停止 A3/A4。

### RQ3：收益来自批次关系、软权重，还是图更新？

比较：

```text
A0-matched -> A0-A3batch
A0-A3batch -> A1-Cov
A1-Cov -> A1-Cov-uniform
A1-Cov -> A1-Cov-A5a
A1-Cov -> A3 -> A4
```

每个差值报告效应、区间和 seed 方向，不把 A4 vs A0-official 作为方法增益。

### RQ4：什么先验条件下关系软对比才可能有帮助？

- 主带宽和 kNN 带宽的 `P_m` 熵；
- beta/KL；
- 邻域保持 NCA@10；
- 有效秩；
- 平坦先验与端到端结果的分层报告。

如果所有数据都平坦，主结论是“当前先验设计不可识别”，不是“软对比无用”。

### RQ5：固定预算安全运营是否有价值？

- PR-AUC、ROC-AUC；
- K=5/10/20 事件命中；
- 首次命中延迟；
- 每用户/每组织的复核成本；
- 误报热点的稳定性；
- 成功案例和明确失败案例。

### RQ6：是否公平优于已有方法？

至少运行：

- TS2Vec；
- SoftCLT T3/T4；
- TNC；
- TS-TCC；
- NNCLR-style；
- FNC/Incremental False Negative Detection 适配；
- Anomaly Transformer；
- DCdetector；
- OmniAnomaly；
- iForest（作为非时序外部基线）。

所有方法使用相同切分、相同评价、相同算力/调参预算；无法公平运行的基线
明确写 `LAN-style`，不得冒充官方实现。

## 7. 统计与成功判据

### 主终点

- PR-AUC；
- K=10 固定预算事件召回，冷却期 7 天。

### 统计

- 10 个训练 seed；
- 先计算 seed 配对差，再对用户/事件簇做 10,000 次 bootstrap；
- 事件簇不能拆成独立事件；
- 单侧 alpha=0.025；
- 预先声明的比较族使用 Holm；
- 报告均值、标准差、逐 seed 方向、95% CI；
- 不把双层 bootstrap 写成确认性 p 值；
- 不显著的结果只写“当前数据不可区分”，不写“方法无效”。

### 三个硬 Gate

**Gate A：实现**

- 覆盖 A3/A5a/A5b 均 `>=80%`；
- 零曝光用户 `=0`；
- 10 个 seed 的 batch hash 全部不同；
- 实际步数等于或超过该 stage 覆盖表长度。

**Gate B：分解**

至少回答：

- A0-A3batch 是否解释 A1-Cov 的收益；
- A1-Cov-uniform/A5a 是否解释软权重或批次关系；
- A1-sparse 与 A1-sparse-EMA 的差异是否存在。

若不能回答，论文最多是系统实现报告，不是 CCF-B 方法贡献。

**Gate C：外部与功效**

- r5.2 按锁定参数一次性复验；
- CLUE 只作有限功效复验；
- LANL 通过数据门禁后才作真实网络外部证据；
- 正式事件/用户簇功效模拟在查看 A4 结果前完成；
- 若 MDE 大于业务最小效应，标记 `UNDERPOWERED`，不伪造显著性。

## 8. 明确的时间表

按 12 周冲刺，团队 24 小时轮班可执行：

| 周 | 工作 | 交付 |
| --- | --- | --- |
| 1 | 冻结 v1.2.7、补齐控制和测试 | 控制代码、单元测试、计划 hash |
| 2 | r4.2 1–2 seed pilot；先跑 A0/A1/控制 | pilot 表、覆盖日志、失败分析 |
| 3 | r4.2 10 seed；正式功效模拟 | 主开发表、MDE、分解结果 |
| 4 | 选参锁定，准备 r5.2；LANL 审计 | 超参清单、D_ext 门禁报告 |
| 5 | r5.2 一次性锁定复验 | r5.2 主表和机制表 |
| 6 | CLUE 两 seed pilot；SMD scoring side | 外部有限功效报告、SMD 结果 |
| 7 | LANL（若可用）；补深度基线 | LANL 结果或明确降级 |
| 8 | 消融、敏感性、低慢攻击、成本分析 | 安全/运营实验表 |
| 9 | 案例研究、失败案例、复现脚本 | 证据包和 artifact |
| 10 | 论文重写为 RQ 驱动结构 | 完整初稿 |
| 11 | 内部红队审稿、统计复核 | 审稿回复矩阵 |
| 12 | 投稿或 CCF-C 降级 | 最终稿件/技术报告 |

## 9. 论文结构

1. Introduction：实际曝光缺口与两个负面结果；
2. Related Work：TS2Vec、SoftCLT、关系对比、内部威胁、固定预算评价；
3. Problem Formulation：完整图、实现图、覆盖、预算；
4. Coverage-Aware Method：采样器、匹配控制、可靠性门禁、scoring；
5. Theory：曝光命题、Pinsker 偏差界、覆盖偏差上界；
6. Experimental Protocol：数据、标签、时间、簇、种子、统计；
7. Results：RQ1–RQ6；
8. Threats to Validity：平坦先验、半合成标签、功效不足、批次选择；
9. Conclusion。

## 10. 判停与降级规则

必须预先执行：

- 覆盖门禁失败：`IMPLEMENTATION_BLOCKED`；
- 数据时间/标签/许可失败：`DATA_GATE_FAILED`；
- 画像主/kNN 先验均平坦：`METHOD_NOT_SUPPORTED`，不写 anchor 成功；
- 事件簇 MDE 高于业务最小效应：`UNDERPOWERED`，不写“无效”；
- r5.2 严格双终点不可检出：保留严格 Gate，但只能报告效应和区间；
- LANL 未通过：撤回 CCF-B 外部有效性承诺；
- 控制实验不能分离关系/软权重/批次：定位为方法学/实证或 CCF-C。

## 11. 最终可接受的三条结论

### A. 最好结果

覆盖感知协议、分解控制和双端 scoring 在 r5.2 与至少一个可审计外部数据上
稳定优于匹配基线，机制边界清楚，安全 venue 可投。

### B. 中性但可发表

覆盖修复是必要的，A4 没有稳定超过 A1+/A9，但论文严谨证明“关系边是否进入
训练”会改变结论，并给出大量实测和负面机制结果。定位方法学/实证论文。

### C. 不可继续包装

如果覆盖和分解控制都无法站住，或者外部数据全部不可审计，则停止 CCF-B
承诺，保留 r4.2/r5.2 的可复现实验和技术报告，不把结果写成 B 类成果。

## 12. 唯一推荐的下一步

立即执行以下顺序，不并行启动 A3/A4 大规模训练：

1. 完成 A0-A3batch、A1-Cov-A5a、A1-Cov-uniform 的正式数据接线；
2. 在 r4.2 跑 1–2 seed，只回答批次/权重分解；
3. 用真实事件和用户簇做 Gate 功效模拟；
4. 若 Gate A/B 有信号，再跑 r4.2 10 seed；
5. 依据 r4.2 结果锁定后跑 r5.2；
6. CLUE 只作有限功效外部复验；
7. 并行争取 LANL；没有可审计外部数据就不承诺 B。

这条路不能保证审稿人一定给 B，但它把“可失败的研究”变成了“即使主方法不赢，
仍能交付科学结论和方法学贡献”的路线。这是当前所有已有成果中唯一有希望
支撑 CCF-B 的方案。
