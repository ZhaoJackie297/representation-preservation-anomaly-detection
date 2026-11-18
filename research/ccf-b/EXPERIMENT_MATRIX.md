# CCF-B 实验矩阵

版本：v1.2.7

日期：2026-11-18。

本矩阵只定义待执行实验。`D_ext` 未过数据门禁、图边覆盖未过实现门禁时，确认性矩阵不得开跑。

## A. 训练端方法

| 代号 | 实例关系来源 | 预算 | 作用 |
| --- | --- | ---: | --- |
| A0-official | 官方随机 batch + 硬实例负样本 | B | 原版外部参照 |
| A0-matched | 共同覆盖 batch + 硬实例负样本 | `B_stage` | 排除批次暴露变化 |
| A0-A3batch | 仅 A3 边选择的共同覆盖 batch + 硬实例负样本 | `B_stage` | 分离批次携带的画像关系 |
| A1-R0 | 原始随机 shuffle、稠密画像软权重 | B | R0 复现基线 |
| A1-Cov | 训练期行为画像软权重、共同覆盖 batch | `B_stage` | 主因果基线 |
| A1-Cov-A5a | 行为画像软权重、A5a 覆盖 batch | `B_stage` | 分离批次形状 |
| A1-Cov-uniform | 与 A1-Cov 同批次、支撑集均匀权重 | `B_stage` | 分离关系权重形状 |
| A1-sparse | 按 `P` 选 top-m，权重取 `P_m` | B | 画像稀疏化 |
| A1-sparse-EMA | 按 `P*exp(-d/tau)` 选 top-m，权重取 `P_m` | B | `epsilon=0` 画像端点 |
| A1-repSupport-P | 表示 top-m 支撑集、权重取归一化 `P_m` | 2B | 支撑无锚、权重有锚 |
| A1+ | 固定画像图，逐阶段重放覆盖表，匹配 A3/A4 | 2B/3B | 等算力/等覆盖对照 |
| A2 | 无 EMA、无画像锚 | 2B | EMA×锚 no/no |
| A3 | 画像锚 + EMA + KL | 2B | 单轮 AdaGraph |
| A4 | AdaGraph 两轮更新 | 3B | 主候选 |
| A3-noAnchor | 只按表示选支撑集和权重 | 2B | 真正移除锚 |
| A3-epsInf | mixed 支撑集、纯表示权重 | 2B | 支撑有锚、权重无锚 |
| A3-noEMA | 当前表示替代 EMA | 2B | EMA 边际贡献 |
| A3-product | 替代乘积融合 | 2B | 融合规则对照 |
| A3-hard | 邻居权重 `1/m` | 2B | 受控软/硬对照 |
| A5a | 度保持随机图 | 2B | 随机稀疏图 |
| A5b | 画像节点置换图，重放共同批次表并单独报告覆盖率 | 2B | 错误画像锚 |
| A6 | 仅画像中位数维度 | B | 附录画像消融 |
| A7 | NNCLR 近邻正样本 | B | 近邻对比 |
| A8a | Incremental False Negative Detection | B | 动态假负检测 |
| A8b | False Negative Cancellation | B | 假负抵消 |
| A9 | 固定图，三段续训、逐段覆盖表和优化器重置 | 3B | 算力/重启/覆盖对照 |

### 端点与质量

- `epsilon=0` 对应 A1-sparse-EMA；
- `epsilon=infinity` 对应固定混合支撑集上的纯表示权重；
- A3-noAnchor 的支撑集只由表示距离决定；
- A3-hard 每个邻居取 `1/m`，邻居总质量与软图一致；
- 主模型邻居总质量 `rho=1.0`；
- `rho={0.25,0.5}` 进入敏感性；
- top-m 正式主网格 `{5,10}`；
- `{20,50}` 仅在覆盖和算力门禁通过后作敏感性。

## B. 图边落批矩阵

所有主对照共享同一 seed/stage 的用户 batch 表：

| 检查 | 目的 |
| --- | --- |
| A0-official vs A0-matched | 测量共同批次本身的影响 |
| A0-A3batch vs A0-matched | 测量批次携带的 A3 关系信息 |
| A1-Cov vs A1-Cov-uniform / A1-Cov-A5a | 分离软权重形状、批次形状和画像关系 |
| A1-Cov vs A3 vs A4 | 在共同批次暴露下比较图更新 |
| A3∪A5a∪A5b 联合覆盖 | 三类图都不能因未落批而假阴性 |
| active_pair_mass_A3/A5a/A5b | 确认三类软权重确实进入 `inst_CL_soft` |
| coverage_A3/A5a/A5b | 三类目标边均至少覆盖 80% |
| zero-realized-edge users | 必须为 0 |
| median edge exposure | 必须 `>=1` |
| batch-table hash by seed | 10 个 seed 必须全部不同 |
| random filler fraction | batch 16 中固定 8 个随机填充用户 |

若覆盖失败：

- 状态记为 `IMPLEMENTATION_BLOCKED`；
- 不进入 A3/A4 结论；
- 先调整批次表、m 或预算；
- 不把失败 run 解释为方法无效。

### A1-Cov vs A0-matched 停止规则

Pilot 首看：

```text
Delta_PR = PR_AUC(A1-Cov) - PR_AUC(A0-matched)
Delta_K10 = RecallK10(A1-Cov) - RecallK10(A0-matched)
```

Pilot 正向阈值：

```text
Delta_PR >= +0.0005
Delta_K10 >= +max(0.02, 1/N_events)
```

若两个终点都未达到阈值，停止 AdaGraph，在 R0 论文披露批次覆盖局限。2 seed 时，同一 endpoint 必须两个 seed 均超过阈值。若只有一个 endpoint 达标，后续标为 `FRAGILE_SINGLE_ENDPOINT`，必须先在 r4.2 10-seed 稳定复现。Pilot 不用于调参。

这里是 CERT pilot 启动阈值；确认性非劣界使用 v1.2.7 的
`-min(0.0005,0.05*PR_A1plus)` 和 `-max(0.01,1/N_events)`，两者不能混用。

详见 `BATCH_REALIZATION_PROTOCOL_v1.2.md`。

## C. 无标签选参

选择分数：

```text
J = mean synthetic PR-AUC across 3 mechanisms x 3 strengths
```

固定规则：

- 只使用训练期内部、首个真实恶意事件之前的数据；
- 固定 S2 欧氏打分；
- 12 个候选先写入候选表；
- 每方法族最多 20 次训练；
- 报告所有候选 `J` 分布；
- 同分依次比较 RankMe 和较低图漂移；
- 前 4 名补 2 seed；
- 冻结后在完整训练期重训 10 个正式 seed；
- 附录最多 40 次训练，报告主选点是否仍在前 3、top-4 集合 Jaccard、Spearman 秩相关和方法排序结论是否改变；
- synthetic J 与 r4.2 真实 PR-AUC 的秩相关只作协议诊断。

真实 validation/test 标签不参与选择。

## D. 数据矩阵

| 数据 | 作用 | 当前限制 |
| --- | --- | --- |
| CERT r4.2 | 开发、pilot、无标签选参、方向性诊断 | 已见；不作最终确认 |
| CERT r5.2 | 锁定参数复验 | R0 已见；不是盲测 |
| CERT r6.2 | 五事件压力审计 | 不进入显著性 |
| CERT r1 | 无标签工程诊断 | 无官方恶意标签；不作确认 |
| SPEDIA | 会话级辅助压力测试 | 17 个 `User` 值、25 天、首日异常 |
| TWOS | 小样本真实感/规避压力测试 | 五天受控竞赛 |
| CLUE-LDS | 首要应用级外部效应复验候选 | 真实正常云行为 + 本项目固定 seed 的半合成账号劫持；24 pairs、48 事件、数据/图门禁已通过，但只有 24 个有效簇，MDE 约 23.0–29.5 pp |
| LANL | 首要真实外部候选 | 红队事件构念、事件簇、时间切分和图适配待审计 |
| ChimeraLog | 托管恢复后备选 | 当前下载未就绪 |
| OpTC | 待审计候选 | 用户日、标签和训练期未确认 |
| SMD | 打分端和检测基线 | 训练端固定 A0 |
| PSM/SMAP/MSL | 连续时序打分端稳健性 | 不验证用户图 |

`D_ext` 最低要求：

- 非 CERT 同生成体系；
- 至少 100 用户；
- 至少 20 事件；
- 至少 30 天干净训练期；
- 标签、用户映射、许可证和切分可审计。

## E. 打分端实验

| 代号 | 距离 | 作用 |
| --- | --- | --- |
| S0 | mask L1 | 官方 TS2Vec 分数 |
| S1 | 余弦表征库 | 方向距离 |
| S2 | 欧氏表征库 | 幅度保留主方法 |
| S3 | 欧氏 + magG | 单点幅度 |
| S4 | 欧氏 + T2 | 漂移 |
| S5 | 完整 PAI | 探索组合 |

Gate 1 前主交叉：

| 训练端 | 打分端 |
| --- | --- |
| A1-Cov | S1 |
| A1-Cov | S2 |
| A4 | S1 |
| A4 | S2 |

探索交互：

```text
Delta_interaction =
  (M_A4_S2 - M_A4_S1) -
  (M_A1_S2 - M_A1_S1)
```

H4 不进入 Gate 1。

## F. 基线

### CERT 用户行为

- iForest；
- LSTM-AE / DeepLog-style；
- LAN 或 LAN-style；
- TS2Vec；
- TNC；
- TS-TCC；
- NNCLR；
- Incremental False Negative Detection；
- False Negative Cancellation；
- TGMC 类方法，在有可复现代码时加入。

### SMD/连续时序

- OmniAnomaly；
- DCdetector；
- Anomaly Transformer；
- TimesNet。

基线规则：

- 使用相同数据切分和评价；
- 超参数只在 r4.2 inner-validation 选择；
- 锁定后直接迁移到 r5.2 / `D_ext`；
- 记录官方代码、适配差异、训练时间、显存和搜索次数；
- 自研 LAN 替代统一标记 `LAN-style`；
- 不使用测试标签调参。

## G. 主统计

共同主终点：

- PR-AUC；
- K=10 固定预算事件召回，冷却期 7 天。

正式 seed：

```text
20261116 至 20261125
```

主推断：

- 条件于 10 个已训练模型；
- seed 配对平均；
- 固定每日 Top-K 名单；
- 用户/事件簇配对 bootstrap 10,000 次；
- 多用户事件整体重采样；
- 用户日不独立。

双层 bootstrap：

- 先重采样 seed，再重采样用户/事件簇；
- 仅作训练随机性敏感性；
- 不自动升级为确认性 p 值。
- 必须说明它不能证明“任取一次训练都稳定”。
- 同时报告逐 seed 配对方向和同号 seed 数，例如 `8/10`，作为描述性证据。

事件不足：

- `N_events<20`：不报告确认 p 值；
- 只报告效应、区间、逐场景方向。

v1.2.7 临时非劣界：

- PR-AUC：`-min(0.0005, 0.05*PR_A1plus)`；
- K=10：`-max(0.01,1/N_events)`；
- 旧 `-0.0005` 和两事件界只保留作历史量级审计。

旧 `-0.0005` 在现有 A1 尺度上约为：

- r4.2 原 mask：5.60%；
- r4.2 S2 欧氏库：3.55%；
- r5.2 S2 欧氏库路径：6.60%。

因此它是约 3.55%–6.60% A1 基线的固定绝对容差，不是随极小基线变化的相对百分比。

业务可接受损失：只有至少一个主终点相对 A7/A8b 获得正向收益时，才接受
v1.2.7 临时界内的损失；K=10 默认不得损失超过一个事件。

### 比较家族

候选固定集：

1. A4 vs A1-Cov；
2. A4 vs A1+；
3. A4 vs A7；
4. A4 vs A8b；
5. A4 vs A9；
6. A0-A3batch vs A0-matched；
7. A1-Cov vs A1-Cov-uniform；
8. A1-Cov vs A1-Cov-A5a；
9. A1-sparse vs A1-sparse-EMA；
10. A3 vs A3-noAnchor；
11. A3 vs A3-noEMA。

第 6–9 项是分解诊断，不属于方法优越性家族；A0-official vs A0-matched
仍是批次实现诊断。

r5.2 锁定复验预先使用三个独立 Holm 家族：

| 家族 | 比较 | 检验数 |
| --- | --- | ---: |
| 主复验 | A4 vs A1-Cov、A1+、A7、A8b、A9 | 10 |
| 机制 | A3 vs A3-noAnchor、A3-noEMA | 4 |
| 关系控制 | A4 vs A5a、A5b | 4 |

每个家族分别校正，不跨家族合并。`D_ext` 的最终 Holm 范围只能在事件数和 MDE 冻结后确定。

## H. 机制指标

- `PeerMass`；
- Top-10 组织结构一致率；
- 同部门/同角色加权召回；
- `NCA@10`；
- `ER_ratio`；
- `P` 与 `P_m` 行熵；
- beta 分布；
- KL 使用量；
- 教师-学生距离；
- active pair mass；
- 异常用户进入正常核心率，只作事后诊断。

机制 Gate：

- NCA@10 的 CI 下界 `>=-0.02`；
- ER_ratio 的 CI 下界 `>=0.90`；
- H2a/H2b 至少一个主终点通过预注册 Holm；
- 仅符号为正不构成机制通过。

## I. 三批执行

| 批次 | 数据与 seed | 变体 |
| --- | --- | --- |
| Phase 0 | smoke | sampler、A0-matched、覆盖报告 |
| Pilot | r4.2，1–2 seed | A0-official、A0-matched、A0-A3batch、A1-Cov-uniform、A1-Cov-A5a、A1-Cov、A1-sparse-EMA、A1+、A3、A5a、A5b、A7 |
| 第二批 | r4.2，10 seed | A1-R0、A1-Cov、A1-repSupport-P、A2、A4、A8a、A8b、A9、A3 机制消融 |
| 锁定复验 | r5.2，10 seed，超参冻结 | 主比较与机制小家族 |
| CLUE pilot | CLUE-LDS，2 seed | 训练期内部合成异常验证；A0-matched、A1-R0、A1-Cov；A1-Cov vs A0-matched |
| 外部确认 | `D_ext`，10 seed | 门禁通过后冻结 Holm |
| 扩展 | Gate 1 后 | 深度基线、S×A、安全、案例 |

CLUE-LDS 的 pilot 不沿用 CERT 的阈值：`positive_PR = max(0.001, 0.05*PR0)`，`positive_recall=3/48`，并且必须在 2 个独立训练 seed 上同方向达标。24 个 pair 的簇 bootstrap 必须整 pair 重采样。

## J. 安全与运营

### Low-and-slow

- 相同总攻击量；
- 单日一次性 vs 多日递增；
- 首次命中延迟、K 事件召回、误报数量和持续时间。

### 扰动

- 主实验在原始行为通道；
- 径向幅度缩放；
- 通道替换；
- 时间平移；
- 随机通道缺失；
- 表示层扰动只作敏感性。

### 运营

- 每日审查条数；
- 有效告警率；
- 冷却节省；
- 每发现一个事件的审查条数；
- 5/10/20 分钟/条明确为场景假设。

### 案例

固定选择：

- 成功：K=10 命中中官方事件编号最小；
- 延迟：K=10 命中中首次命中延迟最大；
- 失败：K=10 未命中中官方事件编号最小。

至少一个失败或困难案例。

## K. 结果表边界

所有结果表必须标注：

- 数据角色；
- 是否盲测/已见；
- 超参数来源；
- B_stage；
- 覆盖门禁状态；
- seed 数；
- 统计单位；
- 口径版本。

不得混表：

- P0 旧生产协议；
- P1 E0/E1 原始 S2；
- P2 E2 内部；
- P3 E3 组件审计；
- P4 缺失修复后联合；
- P5 r5.2；
- P6 r6.2 匹配 T0。
