# CCF-B Research Program

版本：v1.2.8
状态：研究目标已从“CCF-C 上沿”切换为“冲击 CCF-B”。

v5 冻结为 R0 基线。后续工作只围绕一个可证伪的新方法展开，不再把 SoftCLT、训练期行为画像、欧氏库和 PAI 继续组合成更大的方法包。

本文件只定义研究问题、方法主线、硬门槛和启动顺序。具体实验变体见 `EXPERIMENT_MATRIX.md`，冻结判据见 `PREREGISTRATION.md`，执行排期见 `WEEKLY_PLAN.md`。

v1.2.3 的两个硬结论：

- CERT r1 没有官方恶意标签，不能作为确认集；现有外部数据也尚未通过主图方法确认门禁；
- 当前随机 16 用户 batch 会让大多数图边永远不进入 SoftCLT 梯度，必须先冻结共同覆盖批次协议。

因此当前状态是“方法候选 + 可执行门禁”，不是“B 类有效性已证明”。

Phase 0 规格、A3/A5a/A5b 联合覆盖、A1 版本拆分、pilot 最小效应和边曝光命题见 `PLAN_ERRATA_v1.2.2.md`；CLUE-LDS 替代 ChimeraLog 的协议见 `PLAN_ERRATA_v1.2.4.md`；真实数据和覆盖结果见 `PLAN_ERRATA_v1.2.5.md`；平坦先验、独立标签 seed、24 对功效和 CLUE pilot 判据见 `PLAN_ERRATA_v1.2.6.md`；CERT 先验审计、批次控制、功效分类和 v1.2.7 Gate 修订见 `PLAN_ERRATA_v1.2.7.md`；数据封闭和论文重定位见 `PLAN_ERRATA_v1.2.8.md`。

CLUE-LDS 真实图 10-seed 已完成 80% 三图覆盖，需要 948–961 个 stage batch；详见 `CLUE_LDS_PHASE0_RESULTS.md`。完整训练和外部确认矩阵尚未完成。

CLUE-LDS 当前冻结为 24 个用户对、48 个事件，标签 seed=`20261101`。24 个有效簇下 80% 功效 MDE 约为 23.0–29.5 个百分点，因此它只能作“有限功效外部效应复验”，不能承诺检出中等增益。画像 top-10 行熵检查已触发 `FLAT_PROFILE_PRIOR`，A3/A5b 的画像机制贡献只能记为“不可检验”，不能记为“证明画像无效”。

注入 pair 相似度与画像关系相似度的 Spearman 为 −0.199（绝对值低于冻结阈值 0.5），当前标签距离没有触发循环依赖失败条件；复现见 `clue_lds/label_prior_independence_20261118.json`。

**当前科学状态：** CERT r4.2/r5.2 和 CLUE-LDS 的画像 top-m 先验均触发 `FLAT_PROFILE_PRIOR`，画像锚机制记为 `METHOD_NOT_SUPPORTED`；CLUE-LDS 和当前 r5.2 严格 Gate 记为 `UNDERPOWERED`。这不是数据门禁失败，也不是方法失败，具体状态见 `STATUS_REGISTRY.md`。

相关证据：`CERT_PROFILE_PRIOR_AUDIT.md`（主带宽和 kNN 替代带宽）和
`CERT_GATE_POWER_PROXY.md`（r5.2 事件级功效代理）。

## 目标定位

默认主目标为安全方向的 B 类期刊，优先考虑 `Computers & Security` 一类接受系统、实证和运营评价的刊物，最终以最新 CCF 目录和投稿周期为准。

会议路线需要在深度基线、规避分析和真实案例全部完成后启动。

## 研究问题

TS2Vec 的硬实例对比会把行为相近的正常用户当作负样本。固定训练期行为画像距离只能提供静态规则，无法随表示学习调整；纯表示近邻图又会自我强化。

本文研究的问题是：

> 能否用稳定画像先验约束表示反馈，在有限信任域内学习用户软关系图，从而降低假负样本偏差，同时避免异常用户被吸收到正常表示核心？

## 方法：AdaGraph-SoftCLT

### 1. 画像先验

用训练期鲁棒行为画像构造用户关系先验 `P`。CERT 为 47 维，CLUE-LDS 为 105 维；画像只使用行为和时间特征，不使用 LDAP 部门、角色或团队字段。

### 2. EMA 表示距离

用第一阶段模型的 EMA 教师表示计算用户距离，并对 EMA 表示停止梯度。距离按行中位数归一化：

```text
d_tilde(u,v) = d_EMA(u,v) / median_{v != u} d_EMA(u,v)
```

若行中位数过小，使用预先定义的数值下限，避免温度失效。v1.2.7 进一步冻结
`d_tilde <= 10`；clip 比例超过 5% 时标 `EMA_DISTANCE_SATURATED`。教师进入
第二阶段前还必须满足 NCA@10 不低于 A1+ 的 95%、有效秩比不低于 0.90，否则标
`EMA_TEACHER_NOT_QUALIFIED`。

### 3. 支撑集

对每个用户 `u`，按以下分数选择 top-m 邻居：

```text
score(u,v) = P(u,v) * exp(-d_tilde(u,v) / tau)
```

在选出的支撑集 `S_m(u)` 上，将画像先验 `P` 重新归一化为 `P_m`。所有 KL 计算都在该支撑集上完成。

正式图边必须通过 `BATCH_REALIZATION_PROTOCOL_v1.2.md` 的共同覆盖批次表进入模型；否则支撑集只是离线矩阵，不进入实例对比梯度。

### 4. 信任域更新

对每个用户行求解：

```text
min_{W in simplex} <W, d_tilde> - tau * H(W)
s.t. KL(W || P_m) <= epsilon
```

在支撑集上得到：

```text
W_beta(u,v) ∝ P_m(u,v)^beta *
               exp(-(1 - beta) * d_tilde(u,v) / tau)
beta = lambda / (tau + lambda), 0 <= beta < 1
```

KL(W || P_m) 随 beta 单调下降。二分搜索 beta，使其满足 KL 上界；如果 beta=0 已满足约束，则直接取 beta=0。

- `beta=0` 表示纯表示图；
- `beta` 增大表示更多保留画像先验；
- `beta` 接近 1 时回到画像图。

行熵已经由目标中的 `tau * H(W)` 控制，不再额外增加行熵约束。行熵只作为监控指标。

### 5. 两轮更新

- A1-R0：R0 的原始 A1，随机 shuffle、稠密画像权重、单阶段 `B`。
- A1-Cov：固定画像权重，但重放共同覆盖批次表，单阶段 `B_stage`。
- A0-A3batch：硬实例负样本，但共同批次表仅由 A3 边选择，测量批次本身携带的画像关系信息。
- A1-Cov-A5a：固定画像软权重，但批次表由 A5a 边选择，分离批次形状和软权重。
- A1-Cov-uniform：与 A1-Cov 同一批次表，支撑集内权重取均匀值，分离权重形状和软权重存在。
- A1-sparse-EMA：支撑集与 AdaGraph 相同，但每行权重固定为 `P_m`；它是 `epsilon=0` 的真实画像端点。
- A3：画像图训练 `B` 步，更新一次图，再从当前权重继续训练 `B` 步。
- A4：在 A3 基础上再更新一次图，从当前权重继续训练 `B` 步。
- A1+：固定画像图，按与 A3/A4 相同的阶段重新生成覆盖表并连续训练到相同总步数。
- A9：固定画像图，按与 A4 相同的三段预算继续训练，但图不更新。

A3/A4/A9 每个阶段重置优化器状态；A1+ 不重置优化器，使用与总训练预算对应的单一 warmup-cosine 调度。A1+ 的每阶段覆盖表仍与同 seed/stage 的其他变体重放，避免把总覆盖差异误写成算力收益。

主模型邻居总质量固定为 `rho=1.0`，敏感性网格为 `rho={0.25,0.5,1.0,2.0}`。
A3-hard 的每个支撑邻居取 `1/m`，使软硬图的邻居总质量可比。

支撑集与权重使用完整 2×2：A1-sparse-EMA、A3-epsInf、A3-noAnchor，以及新增 A1-repSupport-P。主因果比较使用 A1-Cov，A1-R0 只用于 R0 复现。

## 与已有工作的区别

- SoftCLT：实例距离固定，来自原始数据空间。
- NNCLR：用当前表示支持集最近邻构造正样本，没有画像锚。
- PCL、SwAV：依赖原型或聚类分配。
- Incremental False Negative Detection、False Negative Cancellation：动态检测或抵消假负样本，没有稳定用户先验和信任域约束。
- TGMC 等图对比内部威胁方法：构造异构图和专用图编码器；AdaGraph 只校正 TS2Vec 训练中的实例软权重。

AdaGraph 的新颖性必须限定为：

> 画像锚定、EMA 表示反馈和 KL 信任域约束下的实例软权重图学习。

不能宣称首次将图或对比学习用于内部威胁检测。

## 理论目标

当前 Proposition 1 只是标准加权 InfoNCE 梯度。B 类版本至少需要：

1. 将 InfoNCE 梯度分解为真负样本、假负样本和正样本；
2. 给出假负样本偏差与错误软权重之间的上界；
3. 给出 KL 信任域下表示图相对画像图的漂移界；
4. 给出 AdaGraph 优于固定画像图和纯表示图的条件。

## Gate 1：方法有效性

共同主终点：

- PR-AUC；
- `K=10` 固定预算事件召回。

开发与方向性诊断：CERT r4.2。

锁定参数复验：CERT r5.2，一次运行，不宣称盲测。

外部路线已冻结：`D_app=CLUE-LDS` 只作有限功效外部效应复验；LANL、OpTC 和 ChimeraLog 不再作为后续依赖。SMD/PSM/SMAP/MSL 只作 scoring-side 泛化。

预注册 10 个种子。主推断明确为“条件于当前 10 个训练模型，在用户/事件簇重采样下的配对效应”；双层 bootstrap 只作训练随机性敏感性。

通过 Gate 1 需要：

- 图边覆盖门禁通过；
- A1-Cov 相对 A0-matched 至少一个主终点达到预注册最小正向效应；若两个主终点均未达到，立即停止 AdaGraph 并披露批次覆盖局限；
- r4.2 pilot 通过，但只作方向性判断；
- A4 在 r5.2 锁定复验中相对 A1+/A9 至少一个主终点优效，另一主终点满足 v1.2.7 临时非劣界；当前统计状态仍为 `UNDERPOWERED`，不能承诺该 Gate 一定能区分中等效应；
- A4 相对 A7/A8b 至少一个主终点优效，且两个终点均非劣；
- A4 相对 A1-Cov 有端到端正向收益；
- A4 相对 A5a/A5b 至少一个共同主终点在关系控制家族中通过 Holm 校正，并要求 8/10 seed 方向一致；
- A3 相对 noAnchor 和 noEMA 的机制家族至少各有一个主终点通过 Holm 校正的单侧效应判据；若 `FLAT_PROFILE_PRIOR` 仍为真，涉及画像锚的结论改记 `METHOD_NOT_SUPPORTED`；
- NCA@10 和 ER_ratio 满足预注册非劣界。

共同覆盖批次表必须同时覆盖 A3、A5a 和 A5b，batch 16 中固定 8 个覆盖设计槽位和 8 个随机填充槽位。任一方法覆盖率低于 80% 时，其比较记为 `INVALID_IMPLEMENTATION`。10 个正式 seed 必须使用 10 张不同的批次表，批次表哈希和用户对 Jaccard 必须报告。

Pilot 报告按 `A0-official → A0-matched → A0-A3batch → A1-Cov-uniform/A1-Cov-A5a → A1-Cov → A3/A4` 分解，禁止把 A4 对 A0-official 写成方法增益。

预注册主比较固定为：

1. A4 vs A1-Cov；
2. A4 vs A1+；
3. A4 vs A7；
4. A4 vs A8b；
5. A4 vs A9。

其他消融标记为机制分析或探索性分析。

A4 vs A1-Cov 只表示同批次暴露下的端到端收益，不能用于组件归因。A1-R0 单独保留为 R0 复现参照。画像锚与 KL 信任域由同一个 `beta` 控制，H2 评价的是其他组件存在时的边际贡献和剂量响应，不再宣称三个组件的独立作用。H4 的交互项只作探索性分析。

外部 B 类有效性不再承诺真实网络外部确认；当前论文定位为方法学/实证研究，r5.2 再稳也不能替代独立外部确认。

CLUE-LDS 提供应用级半合成账号劫持；SMD/PSM/SMAP/MSL 提供 scoring-side 外部验证。LANL 不可得，因此不承诺真实企业网络外部确认，也不再用“未来外部数据”作为实验依赖。

第三周末若仍无通过门禁的外部数据，停止条件性 B 主张，转为 C 类上沿或“批次共现与覆盖修正”的方法学/实证研究。

## Gate 2：外部与实验完整性

- TS-TCC、TNC、NNCLR、False Negative Cancellation 和至少一个内部威胁序列基线；
- OmniAnomaly、DCdetector、Anomaly Transformer；
- r5.2 使用 r4.2 锁定的超参数；
- 新增一个连续时序数据集，仅验证打分端和稳健性；
- 所有方法记录调参空间、试验次数、训练时间和推理时间。

## Gate 3：安全证据

- 正式威胁模型；
- 原始行为通道上的 low-and-slow 与扰动实验；
- 相同总攻击量下的单日注入与多日递增对照；
- S4 与当前 T4 表示连接；
- 按固定规则选择成功、延迟和失败案例；
- 5/10/20 分钟/条审查时间的显式场景假设与敏感性分析。

## 数据边界

- r4.2 用于开发、pilot 和无标签选参，不作最终盲测确认。
- r5.2 作为已见数据的锁定参数复验，不宣称为盲测。
- r1 无官方恶意场景标签，只可用于无标签工程诊断，不能计算 PR-AUC、K=10 或确认性 p 值。
- SPEDIA 仅 17 个 `User` 值、25 天且首日有异常，不作主用户图确认集。
- TWOS 为五天受控竞赛，只作外部压力测试。
- ChimeraLog/OpTC/LANL 均不追求，数据包冻结。
- CLUE-LDS 是当前应用级有限外部复验：真实正常云行为 + 基于官方切换算法、由本项目固定 seed 构造的半合成账号劫持。标签 seed=`20261101`，与训练 seed 分离。
- LANL/OpTC/ChimeraLog 不再等待，数据包冻结。
- 现有数百 GB 同源数据只能做稳定性、敏感性和复现，不能冒充独立外部验证。
- SMD 没有跨用户图结构，训练端固定为 A0，仅用于 S 系列和检测基线。
- PSM、SMAP、MSL 只用于评分端和稳健性验证。
- r6.2 用于 2,000/4,000 用户规模、覆盖、成本、5 个事件逐事件命中和失败案例压力审计；不进入显著性检验。

数据边界和不可用原因见 `DATASET_AND_POWER_AUDIT_v1.2.md`；不再把未来数据获取写成实验依赖。

## 标签与调参纪律

- 图更新不使用真实异常标签；
- 超参数只在训练期内部验证中选择；
- 统一使用训练期内部合成异常验证；固定候选表、每个方法族最多 20 次调参训练；
- LDAP 部门、角色和团队只用于事后组织结构参照；
- A4 和 A9 使用相同初始化、阶段重置和调度；
- A7、A8a、A8b 及深度基线使用与主方法相同的调参预算；
- “异常用户进入正常核心比例”只作事后诊断，不能触发早停。

## 论文实验问题

- RQ1：固定画像关系是否稳定，并与组织结构参照一致？
- RQ2：画像锚、EMA 和信任域是否分别降低假负样本偏差？
- RQ3：AdaGraph 与欧氏打分是否存在交互收益？
- RQ4：固定预算、冷却期和证据报告是否改善运营可用性？
- RQ5：面对 low-and-slow 与原始行为扰动是否稳健？

## 启动顺序

1. 冻结 R0 的代码、配置、种子、切分和数据哈希；
2. 按 `PREREGISTRATION.md` 冻结主比较、A1-Cov vs A0-matched 最小效应停止规则和当前可冻结阈值；
3. 完成 `BATCH_REALIZATION_PROTOCOL_v1.2.md` 的采样器、A3∪A5a∪A5b 联合边覆盖、50% 随机填充、seed 差异测试和覆盖率 smoke run；
4. CLUE-LDS 的数据、标签 seed、用户日、图协议、平坦先验和 24-pair 功效已审计；A0/A1 入口 smoke 已通过，下一步补齐分解控制和训练期内部合成异常验证；
5. 在 r4.2 跑 `A0-official、A0-matched、A1-Cov、A1-sparse-EMA、A1+、A3、A5a、A5b、A7` 的 1–2 种子 pilot，并先检查 A1-Cov vs A0-matched；pilot 只决定继续或停止，不用于调参；
6. pilot 和覆盖门禁通过后跑 10 种子 r4.2；
7. 冻结最终超参数，在 r5.2 做一次性锁定复验；
8. 完成 CLUE-LDS 两 seed pilot 后再决定是否运行 A3/A4 外部矩阵；不再补充真实网络外部证据；
9. Gate 1 通过后再运行深度基线、S×A 交叉和安全实验。

文献条目见 `LITERATURE_REVIEW.md` 和 `references_ccf_b.bib`。只有已经核实的条目可以进入论文。

