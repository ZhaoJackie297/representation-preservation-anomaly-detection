# AdaGraph-SoftCLT 预注册表

版本：v1.2.8

冻结日期：2026-11-18

本预注册晚于 R0。新增组件和比较按“候选方法验证”处理，R0 已使用的数据不重新宣称为盲测。所有偏离必须写入版本化勘误。

当前状态：

- 方法公式可冻结；
- 统计对象和事件不足规则可冻结；
- 图边覆盖实现必须通过门禁；
- 数据包已冻结；LANL、OpTC、ChimeraLog 标为 `NOT_PURSUED`；
- `D_app=CLUE-LDS` 只作有限功效外部效应复验；SMD/PSM/SMAP/MSL 只作 scoring-side 泛化；
- CLUE-LDS 的低信息画像先验和 24 个有效 pair 已预注册为“有限功效外部效应复验”边界；
- CERT r4.2/r5.2 与 CLUE-LDS 的主带宽和 kNN 替代带宽均触发 `FLAT_PROFILE_PRIOR`，画像锚机制状态为 `METHOD_NOT_SUPPORTED`；
- CLUE-LDS 外部效应和 r5.2 严格双终点 Gate 状态为 `UNDERPOWERED`；
- 现有数据不宣称真实网络外部有效性；CCF-B 仅按方法学/实证路线条件性争取。

### CLUE-LDS canonical 对账

v1.2.5 中的 `label_seed=20261116 / 20 pairs / 40 events / 568 isolated users`
是开发历史，不再是 canonical。v1.2.6/v1.2.7 canonical 为：

```text
label_seed=20261101
24 pairs / 48 events
565 isolated users
label SHA-256=2441744DED393381C9B42D3E1EFB53B11137810D8F94611FCF76BC0C67D248CA
```

逐文件哈希见 `clue_lds/artifact_hashes_20261118.json`。使用旧数字的 run 标为
`STALE_CLUE_DEVELOPMENT_DATA`。

## 1. 数据地位

| 数据 | 地位 | 规则 |
| --- | --- | --- |
| CERT r4.2 | 开发、pilot、方向性诊断 | 可做实现调试和无标签选参；不提供最终盲测或外部确认 |
| CERT r5.2 | 锁定参数复验 | 2,000 用户、104 天干净训练期、99 个事件；R0 已见，不称盲测 |
| CERT r6.2 | 五事件压力审计 | 不进入显著性检验 |
| CERT r1 | 无标签工程诊断 | 官方 answer key 无恶意标签；禁止 PR-AUC、K=10 和确认 p 值 |
| SPEDIA | 会话/事件级辅助压力测试 | 17 个 `User` 值、25 天且首日有异常；不作主图确认 |
| TWOS | 外部小样本压力测试 | 五天受控竞赛；不作确认性分析 |
| CLUE-LDS | 首要应用级外部候选 | 真实正常云行为 + 本项目固定 seed 的半合成账号劫持；24 pairs、48 事件和真实图覆盖已通过 |
| LANL | 不追求 | 数据包冻结，不能作为依赖 |
| ChimeraLog | 不追求 | 数据包冻结，不能作为依赖 |
| OpTC | 不追求 | 数据包冻结，不能作为依赖 |
| SMD | 打分端验证 | 训练端固定 A0，不运行 AdaGraph 用户图 |
| PSM/SMAP/MSL | 打分端外部验证 | 不用于 AdaGraph 用户图结论 |

外部数据门禁仅适用于未来重新开放的 `D_ext`；当前不追求该门禁：

1. 非 CERT r1–r6.2 同一生成家族；
2. 至少 100 个可建模用户；
3. 至少 20 个可评价事件；
4. 至少 30 天无标签参与的正常训练期；
5. 标签、用户映射、切分和许可证可审计；
6. 在查看 A4 结果前冻结数据哈希和评价协议。

外部证据采用两层：

- `D_app=CLUE-LDS`：应用级半合成账号劫持确认；
- 当前不依赖 `D_net`。

不得把 CLUE-LDS 的账号劫持写成真实恶意内部人员攻击；不得把 LANL 写成 CERT 同构。

CLUE-LDS 进入确认前至少要求：

- 下载文件 MD5 为 `9e318370f96b68077667e9cdc05f26a5`；
- 固定异常注入 seed=`20261101`，不得与训练 seed 重合；
- 24 个切换 pairs，共 48 个用户级事件；
- 每个受影响用户切换后至少 30 个活跃日；
- 切换时间只落入冻结评价窗口；
- 训练期不得看到切换后身份；
- 用户图只由训练期组、共享目录、文件分享或画像构建。

CLUE-LDS 用户筛选冻结为：

1. 只保留 `uidType=name`；
2. 所有注入目标用户保留；
3. 其余用户至少在训练期出现一次；
4. 非目标用户按训练期事件数排序；
5. 上限 1,000 用户；
6. 实际满足条件 838 用户；
7. 任一目标用户训练期全空则整体失败。

### 平坦先验规则

对每个用户和支撑集计算：

```text
H_m(u) = -sum P_m(u,v) log P_m(u,v)
flat_threshold = 0.95 * log(m)
```

满足以下任一条件即标记 `FLAT_PROFILE_PRIOR`：

- 平均 `H_m` 大于阈值；
- 至少 80% 用户大于阈值。

触发后：

- A3/A5b 的画像机制贡献记为“不可检验”；
- 不记为 A3 失败；
- 端到端性能比较继续；
- 必须报告熵、beta 和实际 KL。

CLUE-LDS 当前已触发。

### 标签距离独立性

账号劫持 pair 的选择距离为：

```text
0.7 * action Jaccard
+ 0.3 * average-events-per-active-day ratio
```

画像先验使用 52 维用户日特征聚合后的标准化欧氏距离；两者不是同一个距离。

冻结审计规则：

```text
rho = Spearman(label_pair_similarity, -profile_distance)
若 |rho| >= 0.5，则 pair 选择与画像先验相关性过强，必须增加独立度量对照或重新生成标签。
```

CLUE-LDS 当前审计结果为 `rho=-0.1991`，24 个 pair，状态为 `PASS_DISTANCES_DISTINCT`。该审计只检查距离关系，不查看模型结果。

## 2. 研究假设

- H1a：画像先验锚定的软图相对纯表示图降低假负样本偏差。
- H1b：AdaGraph 相对固定画像软权重取得端到端收益。
- H2a：在其他组件固定时，画像锚的边际贡献为正。
- H2b：在其他组件固定时，EMA 教师的边际贡献为正。
- H2c：画像关系信息不是随机稀疏加权或共同批次暴露造成的假收益。
- H3：AdaGraph 的收益不能由更长训练或重复优化器重启解释。
- H4：训练端 AdaGraph 与打分端欧氏库的交互为正；H4 为探索性。

H2 只研究其他组件存在时的边际贡献和剂量响应，不把画像锚与 KL 信任域拆成两个可独立识别的组件。

## 3. 训练变体

| 编号 | 定义 | 用途 |
| --- | --- | --- |
| A0-official | 官方 TS2Vec 随机 batch + 硬实例负样本 | 原始外部参照 |
| A0-matched | 共同覆盖 batch + 硬实例负样本 | 排除批次暴露变化 |
| A0-A3batch | 仅按 A3 边构建共同覆盖 batch + 硬实例负样本 | 分离批次携带的画像关系信息 |
| A1-R0 | R0 原始随机 shuffle、稠密画像软权重、单阶段 `B` | R0 复现基线 |
| A1-Cov | 固定画像软权重，重放共同覆盖批次表，单阶段 `B_stage` | 同批次暴露的主因果基线 |
| A1-Cov-A5a | 固定画像软权重，重放 A5a 覆盖表，单阶段 `B_stage` | 分离批次形状与软权重 |
| A1-Cov-uniform | 与 A1-Cov 同一批次表，支撑集内均匀权重 | 分离关系权重形状与软权重存在 |
| A1-sparse | 按 `P` 选 top-m 并重归一化，权重取 `P_m` | 画像稀疏化对照 |
| A1-sparse-EMA | 按 `P * exp(-d_tilde/tau)` 选 top-m，权重取 `P_m` | `epsilon=0` 的真实画像端点 |
| A1-repSupport-P | 按表示距离选 top-m，在该支撑集上归一化并取 `P_m` | 支撑无锚、权重有锚单元 |
| A1+ | 固定画像图，逐阶段重放同 seed/stage 覆盖表，连续训练到与 A3/A4 相同总步数 | 等算力/等覆盖对照 |
| A2 | 当前表示、无 EMA、无画像锚，两阶段 `2B` | EMA×锚的 no/no 单元 |
| A3 | 画像锚 + EMA + KL 信任域，一次更新，`2B` | AdaGraph 单轮核心 |
| A4 | AdaGraph 两轮更新，`3B` | 主候选方法 |
| A3-noAnchor | top-m 只按 `d_tilde` 选，权重为 `exp(-d_tilde/tau)`，`2B` | 真正移除画像锚 |
| A3-epsInf | mixed 支撑集、纯表示权重 | 支撑有锚、权重无锚单元 |
| A3-noEMA | 用当前表示替代 EMA，但保留画像锚和信任域，`2B` | EMA 边际贡献 |
| A3-product | `P_m * exp(-d_tilde/tau)` 归一化，不满足 KL 缩放 | 替代融合规则 |
| A3-hard | 与 A3 同支撑集，邻居权重 `1/m`，其余为 0 | 受控软/硬权重对照 |
| A5a | 度保持随机图 + 共同覆盖 batch | 排除随机稀疏加权 |
| A5b | 错误画像锚 `P_perm(u,v)=P(pi(u),pi(v))` | 排除画像关系信息 |
| A6 | 仅画像中位数维度 | 画像构造消融，只放附录 |
| A7 | NNCLR 近邻正样本 | 近邻对比基线 |
| A8a | Incremental False Negative Detection 适配 | 动态假负检测基线 |
| A8b | False Negative Cancellation 适配 | 假负抵消基线 |
| A9 | 固定画像图，三段 `1B` 权重续训、逐段覆盖表重生成与逐段优化器重置 | 排除重复训练/调度收益 |

### 支撑集端点

支撑集与权重采用完整 2×2：

| 单元 | 支撑集 | 权重 | 变体 |
| --- | --- | --- | --- |
| S1,W1 | mixed `P*exp(-d/tau)` | `P_m` | A1-sparse-EMA / `epsilon=0` |
| S1,W0 | mixed `P*exp(-d/tau)` | `exp(-d/tau)` | A3-epsInf |
| S0,W0 | 表示 top-m | `exp(-d/tau)` | A3-noAnchor |
| S0,W1 | 表示 top-m | 在表示支撑集上归一化 `P_m` | A1-repSupport-P |

所有剂量响应图必须报告 `P` 行熵、`P_m` 行熵、每行 beta 分布、实际 KL 和可行 beta 比例。若 `P_m` 接近 `ln(m)`，必须明确指出锚在支撑集内信息很弱。

### 邻居质量

- 主模型 `rho=1.0`；
- 敏感性 `rho in {0.25,0.5,1.0}`；
- `A3-hard` 邻居总质量固定为 1，每个支撑邻居权重为 `1/m`；
- 记录每个用户实际非零正关系项数和 `1+rho` 的有效正关系质量。

## 4. 训练实例与图边落批

### 训练单位

- TS2Vec 输入形状为 `(n_users, n_timestamps, n_features)`；
- 每个训练实例是一个用户的时间序列；
- 一个 batch 是 16 个用户；
- SoftCLT 的实例软权重只在使用 `soft_labels[idx][:, idx]` 时作用于同批用户。

### 覆盖门禁

正式 A3/A4 使用 `CommonCoverageBatchSampler`：

1. 在每个 stage 开始时形成 `E_cov=E_A3 ∪ E_A5a ∪ E_A5b`；
2. batch 16 中固定 8 个覆盖设计槽位和 8 个同 seed、同 stage 的随机填充槽位；
3. 用固定 hash 排序，按贪心覆盖选择设计槽位；
4. 所有匹配变体重放同一 seed/stage 的 16 用户批次表；
5. A0-official 保留原版随机 shuffle，A0-matched 使用共同批次表；
6. 分别记录 A3、A5a、A5b 的 `intended_support_edges`、`realized_support_edges`、`active_pair_mass`、每用户边曝光；
7. 正式门禁：A3、A5a、A5b 覆盖率均 `>=80%`，没有用户零曝光，用户中位曝光 `>=1`；
8. 每次图更新后重新生成该 stage 目标边和覆盖率，不得沿用上一 stage；
9. `B_stage=max(B_base, required_coverage_batches)`；A1+ 与 A9 匹配实际步数，并在每个新 stage 重新生成同 seed/stage 的覆盖表。
10. 任何实际训练步数少于其 stage 覆盖表长度的 run 标为 `INVALID_COVERAGE_BUDGET`。

违反门禁时状态为 `IMPLEMENTATION_BLOCKED`，不得把阴性结果解释为假设失败。

批次表必须随 seed 变化。10 个正式 seed 的 `batch_table_sha256` 必须全部唯一，并报告 seed 间已落批用户对集合的 Jaccard 中位数、最小值和最大值。复用同一 batch 表的 seed 标记为 `INVALID_SEED_RANDOMIZATION`。

预注册分解顺序：

```text
A0-official -> A0-matched -> A0-A3batch
-> A1-Cov-uniform / A1-Cov-A5a -> A1-Cov -> A3/A4
```

每一步单独报告。A4 vs A0-official 不得写成方法增益；A0-matched 下降也不自动构成 AdaGraph 有效性证据。

### Proposition：边曝光下界

随机无放回抽取大小为 `b` 的 batch 时，任意固定无向用户边的单步出现概率：

```text
p = b(b-1) / [N(N-1)]
```

`n_b` 步内期望曝光次数和至少曝光一次概率：

```text
lambda = n_b p
Pr(exposure >= 1) = 1 - (1-p)^n_b
```

若定向批次表保证一条目标边至少出现 `c` 次，则其曝光次数至少为 `c`。该命题只解释软图失效机制和覆盖必要性，不证明 AdaGraph 有效。

详细协议见 `BATCH_REALIZATION_PROTOCOL_v1.2.md`。

### 预算

- 基础预算 `B` 以实际优化步数计；
- A3 为 `2B`，A4 为 `3B`；
- A1+ 连续训练到匹配总预算，但不重置优化器；每个阶段使用同 seed/stage 的新覆盖表；
- A4 后两段和 A9 三段从上一段权重续训，逐段重置优化器并重新生成覆盖表；
- 同一 seed 下数据顺序、图初始化、模型初始化和共同批次表固定；
- 实际 `B_stage`、wall time 和 GPU 小时写入 run registry。

## 5. 无标签超参数选择

所有训练方法和基线使用同一选择协议：

1. 在训练期内再切出一段完全早于首个真实恶意事件的时间作为 inner-validation；
2. 从正常行为注入固定合成异常：幅度尖峰、多通道突发、低幅缓慢漂移；
3. 每机制固定 3 个强度，仅使用合成标签；
4. 选择分数：

```text
J = mean synthetic PR-AUC across 3 mechanisms x 3 strengths
```

5. `J` 固定使用 S2 欧氏表征库；
6. 候选表在开跑前冻结，不按结果新增点；
7. 每个方法族最多 20 次训练：12 个候选各 1 seed，前 4 名各补 2 seed；
8. 报告所有候选的 `J` 分布、淘汰顺序和同分规则，不只报告胜者；
9. 同分时依次比较 RankMe 有效秩、较低图漂移；
10. 冻结超参数后在完整训练期重训 10 个正式 seed；
11. 附录执行一次最多 40 次训练的放宽预算审计，报告主选点是否仍在前 3、top-4 集合 Jaccard、Spearman 秩相关和是否改变方法排序结论；
12. 在 r4.2 事后计算 synthetic `J` 与真实 PR-AUC 的 Spearman 相关，只作选参协议诊断，不用于重新选参。

固定候选表见 `TUNING_CANDIDATES_v1.2.md`。

真实 validation/test 标签不得参与模型选择、早停或图更新。

## 6. 主终点与估计对象

共同主终点：

- PR-AUC；
- `K=10` 固定预算事件召回，冷却期 7 天。

正式种子：

```text
20261116 至 20261125，共 10 个
```

主估计对象：

> 条件于当前训练的 10 个模型，在用户/事件簇重采样下，方法 A 与方法 B 的配对效应。

主推断流程：

1. 每个 seed 计算配对差值；
2. 10 个 seed 求平均；
3. 固定每个模型每日 Top-K 名单；
4. 对用户/事件簇做 10,000 次配对 bootstrap；
5. 用户日不独立重采样；
6. 多用户事件整体重采样。

双层 bootstrap 敏感性：

1. 先对 10 个 seed 有放回重采样；
2. 再在选中 seed 内对用户/事件簇重采样；
3. 重复 10,000 次；
4. 报告为训练随机性敏感性；除非确认集和计划明确将其升级，不取代主 p 值公式。

所有主比较同时报告逐 seed 配对方向、同号 seed 数、均值和标准差。8/10 同号是描述性稳健性，不替代 p 值；双层区间也必须说明其不能证明“任取一次训练都稳定”。

## 7. 显著性与事件不足规则

- 单侧 `alpha=0.025`；
- 优效 p 值：

```text
(1 + #{Delta* <= 0}) / (B + 1)
```

- 非劣 p 值：

```text
(1 + #{Delta* <= delta_margin}) / (B + 1)
```

- `delta_margin` 为绝对差值；
- 确认集事件数 `<20` 时，不报告确认性 p 值，只报告效应、95% 区间、逐场景方向和失败案例；
- 事件数 `>=20` 才执行预注册 Holm 家族；
- r6.2、SPEDIA、TWOS 和单个 Chimera 组织默认不进入确认性 p 值。

## 8. 外部确认与非劣界

### `D_ext`

v1.2.8 数据封闭后，本项目不追求新的真实网络 `D_ext`。CLUE-LDS 只作有限功效
外部效应复验，SMD/PSM/SMAP/MSL 只作 scoring-side 泛化；不得把二者合并成
“独立外部确认”。

`D_ext` 的数据哈希、标签和用户日门禁通过前：

- 最终外部 MDE 不冻结；
- 最终外部 Holm 家族不冻结；
- 只冻结候选比较集和审计规则；
- 所有外部结果标为 `pending_data_gate`。

当前 CLUE-LDS 已达到：

```text
CLUE_LDS_core_data_and_phase0_passed_full_external_matrix_pending
```

这不等于外部确认已通过；完整模型矩阵和最终确认性统计仍未运行。

`D_ext` 门禁通过后：

1. 先冻结数据、事件簇、用户簇和评价期；
2. 只运行 A1-Cov/A1-R0/A0 等基线并获得方差；
3. 在 A4 结果可见前定义业务最小有意义效应；
4. 用 `D_ext` 自身结构做 80% 功效模拟；
5. 冻结最终 MDE、非劣界、Holm 范围和排除规则；
6. 一次性运行确认性矩阵。

### 临时非劣界（v1.2.7）

- PR-AUC：`delta_margin_PR=-min(0.0005, 0.05*PR_A1plus)`；
- K=10 事件召回：`delta_margin_K10=-max(0.01, 1/N_events)`；
- 对 r5.2 的 99 个事件，默认只允许约一个事件步长，即 `1/99=0.010101...`；
- 这比 v1.2.6 的 `-max(0.02,2/N)` 更严格，不能再把两个事件损失当作默认容差。

旧 `0.0005` 的量级依据只用于审计，不再是 v1.2.7 的默认界：

| 当前 A1 附近口径 | PR-AUC | `0.0005` 占基线 |
| --- | ---: | ---: |
| r4.2 instance-soft，原 mask | 0.008922 | 5.60% |
| r4.2 instance-soft，S2 欧氏库 | 0.014091 | 3.55% |
| r5.2 S2 欧氏库路径 | 0.007572 | 6.60% |

最终 `D_ext` 和非劣界仍须按事件簇 MDE 复核；若需改变，只能新建看结果前冻结的勘误。当前 PR 的严格界是上述 `min(0.0005, 5% PR_A1plus)`，不是旧的固定 0.0005。

业务可接受损失预先定义为：

> 只有当 A4 相对 A7/A8b 至少一个主终点取得正向收益时，才接受 v1.2.7 临时非劣界内的损失；K=10 事件召回默认不得损失超过一个事件。没有主终点收益时，任何超过非劣界的损失都不接受。

这是双终点 trade-off 边界，不表示 AdaGraph 与更强基线等价。

### 主比较集

最终候选比较：

1. A4 vs A1-Cov；
2. A4 vs A1+；
3. A4 vs A7；
4. A4 vs A8b；
5. A4 vs A9；
6. A0-matched vs A0-official，批次实现诊断；
7. A0-A3batch vs A0-matched，批次携带画像关系诊断；
8. A1-Cov vs A1-Cov-uniform / A1-Cov-A5a，权重与批次分解；
9. A1-sparse vs A1-sparse-EMA，支撑集信息增量；
10. A3 vs A3-noAnchor；
11. A3 vs A3-noEMA。

前五项属于端到端/算力/已有方法比较；第 6–9 项是分解诊断，不能和主确认家族混合；第 10–11 项属于机制小家族。A0-matched 差异只用于批次实现诊断，不写成方法优越性。

### r5.2 锁定复验的三个独立 Holm 家族

r5.2 仍是已见数据，但为避免看到结果后改变检验范围，预先冻结以下三个互不合并的家族：

1. 主复验家族：A4 vs A1-Cov、A1+、A7、A8b、A9，`5 x 2 = 10` 个检验；
2. 机制家族：A3 vs A3-noAnchor、A3-noEMA，`2 x 2 = 4` 个检验；
3. 关系控制家族：A4 vs A5a、A5b，`2 x 2 = 4` 个检验。

每个家族分别做 Holm 校正。不能因为一个家族通过就重写另一个家族，也不能把诊断性的 A0-matched vs A0-official 加入其中。

`D_ext` 通过数据门禁后，按其自身事件数和 MDE 另建外部确认家族；外部家族不能沿用 r5.2 的 p 值。

## 9. Gate 1

以下条件全部满足才通过方法 Gate：

1. 共同覆盖批次门禁通过；
2. A1-Cov 相对 A0-matched 至少一个主终点达到预注册最小正向效应；
3. r4.2 pilot 方向性通过，但 pilot 只作非约束启动判断；
4. A4 在 r5.2 锁定复验中相对 A1+/A9 至少一个主终点优效，另一主终点满足 v1.2.7 临时非劣界；当前状态标记 `UNDERPOWERED`，不得据此承诺中等效应可检出；
5. A4 相对 A7/A8b 至少一个主终点优效，两个主终点均非劣；
6. A4 相对 A1-Cov 有端到端正向收益；
7. A4 相对 A5a/A5b 至少一个主终点有统计支持的收益；
8. A3 相对 noAnchor/noEMA 的机制小家族至少各有一个主终点通过 Holm；在 `FLAT_PROFILE_PRIOR` 下，涉及画像锚的机制结论改记 `METHOD_NOT_SUPPORTED`，不能强行判 pass/fail；
9. NCA@10 的 95% CI 下界不低于 `-0.02`；
10. ER_ratio 的 95% CI 下界不低于 `0.90`；
11. 不存在异常用户进入正常核心率持续恶化的事后警示。

若 CERT/CLUE-LDS 均保持 `FLAT_PROFILE_PRIOR`，即使端到端 Gate 通过，也不能宣称“画像锚驱动的 AdaGraph 获得验证”；B 类机制主张必须等待信息量足够的新先验/新数据集。

### A1-Cov vs A0-matched 最小效应与停止规则

Endpoint 在 pilot 中算“正向”必须同时满足方向和幅度：

```text
Delta_PR >= +0.0005
Delta_RecallK10 >= +max(0.02, 1/N_events)
```

2 个 pilot seed 时，每个 seed 必须在同一 endpoint 上超过该阈值；1 个 pilot seed 时，仅按幅度判断并标记 `PILOT_ONE_SEED`。

若两个主终点均未达到上述正向标准，立即停止 AdaGraph 主实验，不运行 A3/A4 的 10-seed 矩阵，并在 R0 论文中披露共同批次与图边落批的局限。

若只有一个主终点达标，可进入 10-seed r4.2 方向性复验，但标记为 `FRAGILE_SINGLE_ENDPOINT`；10-seed 下仍应报告同号 seed 数和区间，不能凭 pilot 结果进入 r5.2。

Pilot 只决定继续或停止，不用于选择超参数。

这里的 `+0.0005`/`+max(0.02,1/N)` 是 CERT pilot 的启动阈值，不是
v1.2.7 的确认性非劣界；不得与 r5.2/D_ext 的 `min(0.0005,5% PR_A1plus)` 和
`max(0.01,1/N)` 混用。

### CLUE-LDS pilot 的独立最小效应

CLUE-LDS 不得直接沿用 CERT r4.2 的 `0.0005` PR-AUC 门槛。其 pilot 判据在数据结果可见前单独冻结为：

```text
PR0 = A0-matched 的 PR-AUC
positive_PR = Delta_PR >= max(0.001, 0.05 * PR0)
positive_recall = Delta_recall >= 3 / 48 = 0.0625
```

`PR0` 只由 A0-matched 计算，不使用 A1/A3/A4 的结果。事件阈值对应至少 3 个用户级事件。

CLUE-LDS pilot 必须运行 2 个独立训练 seed：

- 两个 seed 在同一 endpoint 上必须均为正；
- 两个 seed 的均值必须达到该 endpoint 的阈值；
- 单 seed 只作实现 smoke，不能用来继续或停止 AdaGraph；
- pilot 不用于选择超参数。

只有两个 endpoint 都未达标时才停止 CLUE 上的 AdaGraph 主实验；只有一个达标时标记 `FRAGILE_SINGLE_ENDPOINT`，不得据此宣称外部确认成功。

按 pair 聚类，24 个有效簇在 80% 功效下的 MDE 约为 23.0–29.5 个百分点的事件召回差。因此 CLUE-LDS 只能定位为“有限功效外部效应复验”，不能把不显著解释为方法无效，也不能用它单独支撑中等效应。

外部 B 类 Gate 额外要求：

- `D_app=CLUE-LDS` 的数据、时间、标签和图门禁通过；
- CLUE-LDS 的预注册主要比较完成；
- SMD/PSM/SMAP/MSL 的 scoring-side 验证完成；
- 论文明确区分 r4.2 开发、r5.2 已见复验、CLUE-LDS 半合成账号劫持和 scoring-side 外部数据；
- 不声称 LANL、OpTC 或 ChimeraLog 获批；
- 当前外部 B 类承诺为条件性方法学稿，不是外部确认稿。

## 10. 机制分析

### FN 代理质量

LDAP 只用于评价：

```text
PeerMass(u) = sum_{v: same department and role} W(u,v)
```

先分别做 A1-R0、A1-Cov、A5a、A5b 的 PeerMass 基线。只有画像锚定方法对随机图和置换图均显示关系信号，才把 PeerMass 用作 H1a 代理终点。

报告：

- `PeerMass@top-m`；
- Top-10 组织结构一致率；
- 同部门和同部门同角色加权召回；
- A5a/A5b 对应值。

### 邻域保持

```text
NCA@10 =
  mean_u |N10_raw(u) intersect N10_repr(u)| /
         |N10_raw(u) union N10_repr(u)|
```

比较对象为同预算 A1+，95% CI 下界不低于 `-0.02`。

### 有效秩

```text
ER_ratio = ER_method / ER_A1plus
```

95% CI 下界不低于 `0.90`。

### 剂量响应

- epsilon 网格：`{0,0.001,0.01,0.05,0.1,0.2,0.5,infinity}`；
- 在 epsilon 轴上计算 Spearman 单调性；
- 报告 beta、有效温度、KL、`P`/`P_m` 行熵；
- 若 `P_m` 接近 `ln(m)`，明确标记锚信息弱；
- 若 epsilon 曲线非单调，不通过挑选个别点构造结论。

### EMA 可检验性

每阶段记录：

- 教师-学生参数余弦差；
- EMA 用户表示与学生表示的余弦距离；
- 阶段内有效更新常数；
- 若教师与学生近乎一致，H2b 标为“不可检验”。

## 11. 数据结构与实现规格

### 画像先验 `P`

- 使用训练期行为画像；CERT 为 47 维，CLUE-LDS 为 105 维；
- 排除部门、角色、团队和 LDAP 派生字段；
- 每维用训练用户中位数和 IQR 标准化；
- `D_profile` 为欧氏距离；
- `h=median(D_profile)`；
- `P_raw(u,v)=max(exp(-D_profile(u,v)^2/(2h^2)),1e-12)`，`u!=v`；
- 对角置 0，逐行归一化；
- 支撑集上的 `P_m` 再归一化。

预注册的替代带宽仅为审计用：

```text
h_knn = median_u d(u, NN_10(u))
```

主结果仍使用 `h=median(D_profile)`。替代带宽不能根据下游性能选择。CERT r4.2/r5.2
的主带宽和 kNN 带宽均触发 `FLAT_PROFILE_PRIOR`；详情见 `CERT_PROFILE_PRIOR_AUDIT.md`。

### EMA

- 每阶段开始时教师权重复制学生；
- 仅用训练期活跃日；
- 日表示先 L2 归一化，再对活跃日均值并归一化；
- `d_EMA=1-cosine`；
- 行中位数归一化，距离下限 `1e-6`；
- 归一化距离上界固定为 `10`，并记录 clip fraction；
- decay 搜索 `{0.99,0.995}`；
- 记录教师-学生距离；不满足有效步数条件的 decay 不进入正式模型；
- 教师必须满足 `NCA@10_teacher >= 0.95*NCA@10_A1plus` 且
  `ER_teacher/ER_A1plus >= 0.90`，否则标 `EMA_TEACHER_NOT_QUALIFIED`；
- clip fraction > 5% 时标 `EMA_DISTANCE_SATURATED`，不得用该阶段的机制结果作支持证据。

### 支撑集和 KL

- `score(u,v)=P(u,v)*exp(-d_tilde(u,v)/tau)`；
- 按 score 取 top-m；
- 在支撑集上计算 `P_m` 和 KL；
- beta 二分容差 `1e-6`，最多 64 次；
- `beta=0` 走独立端点分支；
- 无可行 beta 时标 `BETA_SOLVE_FAILED`，该 run 不得进入主比较，不能静默回退 `beta=1`；
- `epsilon=0` 必须产出 `P_m`，`epsilon=infinity` 必须产出纯表示权重。

### Pinsker 型偏差界

对同一支撑集上的行分布 `W_u` 与真实同类关系 `W*_u`，若
`KL(W_u || W*_u) <= delta`，且用户表示有界 `||x_u|| <= B`，则

```text
|| c_W(u) - c_W*(u) ||_2 <= C * sqrt(delta / 2)
```

其中 `C` 由特征空间直径决定；信任域 `KL(W_u||P_m)<=epsilon` 只能给出
`O(sqrt(epsilon))` 级的最坏漂移上界。该界不保证方向正确，也不能替代下游实验。

### W 到 SoftCLT

- 自权重 1；
- 非自邻接权重 `rho * W_beta(u,v)`；
- 主模型 `rho=1.0`；
- A3-hard 的邻居权重为 `rho/m`；
- 未进入支撑集或批次的对作为普通负样本；
- 只报告实际进入 `soft_labels[idx][:,idx]` 的 `active_pair_mass`。

## 12. 参数和候选表

### AdaGraph

- EMA decay：`{0.99,0.995}`；
- tau：`{0.05,0.1,0.2,0.5,1.0}`；
- epsilon：`{0,0.001,0.01,0.05,0.1,0.2,0.5,infinity}`；
- top-m 正式主网格：`{5,10}`；
- top-m `{20,50}` 只在覆盖门禁和算力审计通过后作敏感性；
- 学习率和 weight decay 共用固定搜索空间；
- 12 个候选由开跑前写入的固定表生成；
- 主模型 `rho=1.0`。

### A7

- NNCLR 的 `k` 使用与 AdaGraph 相同的 `{5,10}`；
- `{20,50}` 仅作敏感性；
- 使用共同覆盖批次表，避免正样本库与批次暴露不同。

### A8a/A8b

- 优先使用官方代码或可核对适配；
- 相同用户级负样本池和共同批次表；
- 附录记录相对原论文的输入、正负样本单位和训练步数改动。

## 13. 执行批次

### Phase 0：实现门禁

- `CommonCoverageBatchSampler`；
- 单元测试：A3∪A5a∪A5b 联合边覆盖、50% 随机填充、同 seed 可复现、不同 seed 表不同、用户不重复；
- pair coverage smoke run；
- A0-matched 实现；
- 画图/统计管线测试。

### Pilot

r4.2，1–2 seed：

```text
A0-official, A0-matched, A1-Cov, A1-sparse-EMA,
A1+, A3, A5a, A5b, A7
```

第一优先比较为 `A1-Cov vs A0-matched`。Pilot 只判断继续或停止，不作确认性推断，也不用于调参。

CLUE-LDS 入口验证与 pilot：

```text
CLUE-LDS, 2 training seeds
A0-matched
A1-R0
A1-Cov
```

- 先完成训练期内部 synthetic anomaly validation；
- 两个 seed 使用同一标签、用户、时间和评价协议；
- 主比较为 A1-Cov vs A0-matched；
- 判据使用 `max(0.001,0.05*PR0)` 和 `3/48`，不使用 CERT 的 `0.0005` 阈值；
- 主 bootstrap 按注入 pair 整簇重采样；
- pilot 只决定是否进入 A3/A4 外部矩阵，不用于调参或选择异常阈值。

### 并行任务

- 不再审计或等待 ChimeraLog、LANL、OpTC；
- 将现有数据整理成可审计的 scoring-side 和有限外部复验清单；
- 不因外部数据等待而跳过 Phase 0 和 A1/A0-matched pilot。

### 第二批

Pilot 和覆盖门禁通过后：

```text
r4.2, 10 seed
A1-R0, A1-Cov, A1-repSupport-P, A2, A4, A8a, A8b, A9,
A3-noAnchor, A3-noEMA, A3-epsInf, A3-product, A3-hard
```

### 锁定复验

- 超参数在 r4.2 inner-validation 冻结；
- r5.2 一次性运行；
- r5.2 标签不参与后续调参；
- 结果明确标记“已见数据复验，非盲测”。

### 外部确认

当前不执行新的 `D_ext` 外部确认。CLUE-LDS 只作有限功效外部效应复验；
SMD/PSM/SMAP/MSL 只作 scoring-side 泛化；辅助结果不能替代独立确认。

### 安全与基线

Gate 1 和方法锁定后再运行深度基线、S×A 交叉、安全扰动、案例和运营成本。

## 14. 标签、早停与排除

- 图更新、调参、早停不使用真实异常标签；
- synthetic anomaly 标签只来自固定生成器；
- LDAP 只用于评价；
- 异常用户进入正常核心比例只作事后诊断；
- 测试标签不得触发早停或选择模型；
- 失败 run 保留在 registry；
- 任何使用 test/validation 真实标签选择模型的 run 从主分析排除；
- 任何跳过共同覆盖批次门禁的 A3/A4 run 标为 `INVALID_IMPLEMENTATION`。
