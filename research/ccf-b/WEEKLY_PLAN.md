# 八周执行计划 v1.2.8

前提：

- 团队可轮班推进，主算力为 RTX 5060；
- v1.2.6 已完成 CLUE-LDS canonical 审计；v1.2.7 已补 CERT 平坦先验、Gate 功效代理和状态分类；v1.2.8 已冻结现有数据范围；
- CLUE-LDS 的数据、标签和图门禁已通过，但完整模型矩阵尚未开始；LANL、OpTC、ChimeraLog 不再等待；
- CLUE-LDS 真实 10-seed 图覆盖已通过，A0/A1 入口和 soft trainer smoke 已完成；正式模型矩阵和新增分解控制尚未完成；
- CERT r4.2/r5.2 与 CLUE-LDS 的主/kNN 画像先验均触发 `FLAT_PROFILE_PRIOR`；画像锚机制当前为 `METHOD_NOT_SUPPORTED`；
- 八周计划以“先过实现和数据门禁，再跑主矩阵”为顺序。

## 第 1 周：门禁、冻结与采样器

### 数据

- 完成 CERT r1 无标签结论归档；
- 归档 CLUE-LDS 的 MD5/SHA-256、canonical 24-pair/48-event、用户筛选、平坦先验和距离独立性审计；旧 20/40/568 保留为历史；
- 归档 CERT r4.2/r5.2 主带宽、kNN 带宽、m=5/10 的画像先验审计；
- 将 LANL、OpTC、ChimeraLog 标为 `NOT_PURSUED`；现有数据只做稳定性、敏感性和复现；
- 将 SMD/PSM/SMAP/MSL 明确限制为 scoring-side 外部验证。

### 实现

- 冻结 R0 的 code、配置、种子、切分和数据哈希；
- 独立采样器、合成 smoke 和 CLUE-LDS 真实覆盖已完成，见 `PHASE0_SMOKE_RESULTS.md` 与 `CLUE_LDS_PHASE0_RESULTS.md`；
- 实现 `CommonCoverageBatchSampler`；
- 单元测试：
  - A3∪A5a∪A5b 联合边覆盖；
  - 8 个设计槽位 + 8 个随机填充槽位；
  - 用户不重复；
  - 同 seed 可复现；
  - 不同 seed 的批次表不同；
  - 所有变体共享同一批次表；
- 在 r4.2 跑采样器 smoke run。
- 验证已接入的 A0-A3batch、A1-Cov-A5a、A1-Cov-uniform 在正式覆盖表上的行为。

### 协议

- 冻结 A1-sparse-EMA、A3-hard `1/m`、邻居质量 `rho`；
- 冻结 `A1-Cov vs A0-matched` 的最小效应和双终点停止规则；CLUE-LDS 使用独立阈值，并要求 2 个 seed 同方向；
- 冻结批次携带关系信息的分解顺序，以及 `A1-sparse vs A1-sparse-EMA` 对照；
- 记录 CLUE-LDS 的 `FLAT_PROFILE_PRIOR`、24-pair MDE 和标签/画像距离审计结果；
- 冻结事件不足规则、v1.2.8 临时非劣界和双层 bootstrap；
- 归档 r5.2 Gate 功效代理；正式事件/用户簇功效结果出来前不宣称 Gate 可检出中等效应。
- 记录实际 `B_stage`；
- 固定 12 候选表。

交付：

- 数据门禁报告；
- pair coverage 报告；
- sampler 单元测试；
- v1.2.8 hash。

## 第 2 周：非约束性 pilot

在 r4.2 跑 1–2 seed：

```text
A0-official
A0-matched
A1-Cov
A1-sparse-EMA
A1+
A3
A5a
A5b
A7
```

检查：

- active pair mass；
- 每用户边曝光；
- 教师-学生距离；
- 有效秩与各向同性；
- KL 和 beta 是否有非空变化；
- 训练损失和显存。

第一比较为 `A1-Cov vs A0-matched`：

- 正向阈值：`Delta_PR>=+0.0005`，`Delta_K10>=+max(0.02,1/N_events)`；
- 两个终点都未达到阈值，停止 AdaGraph；
- 在 R0 论文中披露共同批次覆盖和图边落批局限；
- 2 seed 时同一终点必须两个 seed 都超过阈值；
- 只有一个终点达标时，标为 `FRAGILE_SINGLE_ENDPOINT`，先在 10-seed r4.2 复验；
- pilot 不用于调参。

Pilot Go/No-Go 明确为**非约束性启动判断**。它只决定是否进入第二批，不作确认性推断。r4.2 测试期结果一旦用于该判断，r4.2 永久标记为已见开发数据。

CLUE-LDS 的数据、协议和首版训练入口审计已完成；不再等待 LANL/OpTC/ChimeraLog。CLUE-LDS 的正式矩阵、新增分解控制和两 seed pilot 尚未完成，不能写成外部确认已通过。

交付：

- pilot 表；
- 覆盖与稳定性报告；
- 是否进入第二批的工程决定。

## 第 3 周：r4.2 十种子与锁定准备

运行：

```text
A1-R0
A1-Cov
A1-repSupport-P
A2
A4
A8a
A8b
A9
A3-noAnchor
A3-noEMA
A3-epsInf
A3-product
A3-hard
```

要求：

- 10 seed；
- 同 seed/stage 共享覆盖批次表；
- A1+ 和 A9 实际步数匹配；
- A1-R0 与 A1-Cov 分开报告，不能合并；
- 补齐 A1-repSupport-P 与 A3-epsInf 的 2×2 支撑/权重单元；
- 结果只作方向性和工程复验，不进入确认性 p 值；
- 完成固定候选表选参；
- 冻结 r5.2 将使用的全部超参数。

交付：

- r4.2 十种子训练端主表；
- 选参分布；
- 实际 GPU 小时；
- 锁定超参数清单。

第三周末外部数据判定：

- CLUE-LDS 核心数据与 Phase 0 已通过，保留条件性 B 路线；但其 24 个 pair 只能提供有限功效复验，A3/A4 外部矩阵尚未开始；
- LANL/OpTC/ChimeraLog 已标 `NOT_PURSUED`，不能补写“已完成的真实网络外部确认”；
- 若完整模型结果和外部效应均不足，切到 C 类上沿或“批次共现与覆盖修正”的方法学/实证定位。

## 第 4 周：r5.2 一次性锁定复验

- 超参数不得根据 r5.2 真实标签修改；
- 运行主比较与机制小家族；
- 输出 PR-AUC、K=10、均值±标准差、配对区间；
- 事件级指标按 99 个事件解释；
- 记录“已见数据、非盲测”；
- 若出现致命失败，先检查实现和批次覆盖，不直接改阈值。

同时并行接入一个可复现序列基线，优先选择可在 CERT 上公平运行的实现。

交付：

- r5.2 锁定复验表；
- 失败分析；
- 外部数据门禁状态更新。

并行使用 r6.2 做压力审计：

- 2,000 vs 4,000 用户切片；
- 覆盖、内存、训练/编码时间和每用户告警成本；
- 5 个事件逐事件命中、首次延迟和失败案例；
- 只作描述性压力证据，不进入显著性检验。

## 第 5 周：有限外部复验与统计

- 完成 CLUE-LDS A0-A3batch/A1-Cov-A5a/A1-Cov-uniform 入口和训练期内部合成异常验证；
- 用 2 个独立训练 seed 跑 CLUE-LDS pilot，先看 A1-Cov vs A0-matched；单 seed 只作实现 smoke；
- 以 pair 整簇做 bootstrap，使用 `positive_PR=max(0.001,0.05*PR0)` 和 `positive_recall=3/48`，不沿用 CERT 阈值；
- 不再适配 ChimeraLog、LANL、OpTC；
- CLUE-LDS 只作有限功效外部效应复验；SMD/PSM/SMAP/MSL 只作 scoring-side 验证；
- 在 A4 结果可见前冻结业务最小有意义效应；
- 用每条数据自身结构做 MDE/80% 功效模拟；CLUE-LDS 当前 MDE 为 23.0–29.5 pp；
- 只有完整模型矩阵和外部数据门禁同时通过后，才冻结最终 Holm 家族；
- 完成用户/事件簇主 bootstrap 和双层敏感性；
- 完成 A1-Cov/A4 × S1/S2 四单元；
- 完成 SMD/PSM/SMAP/MSL 的 scoring-side 稳健性。

交付：

- CLUE-LDS 两 seed pilot 与 scoring-side 外部验证状态；
- 最终统计计划；
- S×A 主交叉；
- 外部 MDE。

若 CLUE-LDS 复验功效不足：

- 不进入“B 类外部确认”；
- 不把 r5.2 包装成盲测；
- 启动 C 类或“方法学/实证研究”改写预案。

## 第 6 周：基线、安全与运营证据

### 基线

- OmniAnomaly；
- DCdetector；
- Anomaly Transformer；
- TS-TCC；
- 一个内部威胁序列基线；
- 有可复现代码时加入 TGMC 类方法。

统一记录：

- 输入适配；
- 调参预算；
- 官方代码版本；
- 训练时间、显存和失败原因。

### 安全

- 正式威胁模型；
- 原始行为通道 low-and-slow；
- 相同总攻击量下单日与多日曲线；
- 幅度、通道、时间、缺失扰动；
- 表示层扰动只作敏感性；
- S4 证据卡接入 T4；
- 固定规则选择成功、延迟、失败案例；
- 5/10/20 分钟/条作为明确场景假设。

交付：

- 基线主表；
- 安全检查表；
- 案例卡；
- 运营成本敏感性。

## 第 7 周：理论、写作和贡献收口

- 将 Theorem/Proposition 限定为可证明内容：
  - 软权重 InfoNCE 梯度分解；
  - 假负样本偏差上界；
  - KL 信任域漂移界；
  - 加性噪声或最小间隔下的有限样本风险界；
- 明确图只在共同覆盖批次中实现；
- 按 RQ1–RQ5 重写实验；
- 删除无关历史模块；
- 主文区分：
  - r4.2 开发；
  - r5.2 已见复验；
  - CLUE-LDS 有限功效外部复验；
  - SMD/PSM/SMAP/MSL scoring-side 泛化；
- 贡献表述以条件、算法性质或可复现新现象为主，不以组件数量为主。

交付：

- CCF-B 结构初稿；
- 定理与实验对应表；
- 数据/口径表。

## 第 8 周：红队、复现与投稿判断

- 按审稿人问题检查：
  - r1 是否仍被误写为确认集；
- 非劣界是否仍使用旧固定 0.0005/两事件界；
  - 事件少于 20 是否仍报 p 值；
  - A1-sparse-EMA 是否完成；
  - A3-hard 是否保持 `1/m`；
  - 是否报告 active pair mass；
  - 是否把 r5.2 称盲测；
  - 是否把 Chimera/SPEDIA/TWOS 写成已确认；
- 复跑关键 run；
- 冻结 artifact、hash 和附录；
- 建立回复审稿意见的结果索引；
- 核查所有文献与许可证；
- 根据现有数据与主结果作投稿分级：
  - `B-ready`：实现、r5.2、CLUE 有限复验、scoring-side 外部数据、统计和安全全部通过；
  - `C-ready`：实现和锁定复验通过，但外部泛化或机制结果不足；
  - `stop`：图边覆盖、加性噪声、算力归因或异常核心警示失败。

交付：

- 可投稿版本或明确的 C 类版本；
- artifact 清单；
- 最终风险声明。

## 停止条件

任一情况出现时暂停 B 组实验：

- 图边覆盖率 `<80%` 或有用户零曝光；
- A0-matched 与 A0-official 的差异解释不了训练分布变化；
- A4 在 r5.2 相对 A1+/A9 至少一个终点优效且另一终点满足临时非劣界；若功效模拟判定不可检出，标 `UNDERPOWERED`；
- A4 相对 A7/A8b 超过绝对非劣界；
- A3 与 A5b 无差异且 PeerMass 无信号；
- `epsilon=0` 与 A1-sparse-EMA 仍有系统性差异，说明支撑集定义未受控；
- 有效秩持续下降；
- 异常用户进入正常核心率持续恶化；
- 深度基线无法公平运行；
- 外部数据许可证不允许复现实验。

现有数据无法支持真实网络外部确认，不否定方法工程结果，但当前只能按方法学/实证论文评级。
