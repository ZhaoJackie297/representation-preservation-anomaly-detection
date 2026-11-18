# MarketRisker：外部 AI 审查与复算交接说明

日期：2026-11-17  
项目：大学生创新训练计划“基于自监督时序学习的异常交互识别技术研究”  
当前研究主题：面向 TS2Vec 的“表征信息保全”方法适配  
本文用途：交给另一个具备数学、机器学习和统计审查能力的 AI，辅助核对理论、代码、结果和实验设计

---

## 0. 先给审查者的任务边界

请把工作分成四类，不要混写：

1. **已经由数据支持的事实**；
2. **实现上已经核对的公式与比较口径**；
3. **尚未证明、但可解释现有现象的机制假设**；
4. **需要新实验才能验证的结论**。

必须遵守：

- 不按“哪组结果更好看”选方法，按方法正确性选数；
- 测试集不得参与权重、阈值、K、温度、带宽或模型选择；
- 同配置、同协议、同版本的数才能直接比较；
- 不能把相关性自动写成因果；
- 不能把 3 个随机种子下的方向一致性写成统计显著；
- 不能声称“官方 PAI 权重最优”或“解压资料包即可复现”；
- 不得伪造未生成的文件或凭经验补写实验结论。

当前最需要审查的核心问题：

> 从表示空间几何上，为什么 instance-soft 有效，而 temporal-soft 在当前 CERT 日粒度任务上没有独立收益？现有解释是否成立，哪些地方需要更严格的推导或新实验？

---

## 1. 课题背景与研究定位

导师课题关注：

- 用户行为具有连续、情境依赖和群体关联特征；
- 异常可能表现为点异常、上下文异常或跨用户集体异常；
- 标注稀缺，需要通过自监督时序学习学习正常行为规律；
- 最终目标包括异常识别、风险演化和动态预警。

当前项目没有完成导师课题的全部内容，只聚焦底层的 S2 表征学习及后续审计：

1. 复现官方 TS2Vec 的核心表征学习；
2. 在实例对比中加入内容相关的软权重；
3. 在打分端比较余弦、欧氏表征库与 PAI 幅度融合；
4. 在 CERT 多版本和 SMD 上做时间外、固定预算和多种子实验。

本项目不声称发明 TS2Vec、SoftCLT 或 PAI 的基础算法。创新定位是：

> 面向用户行为数据的方法适配、机制解释、严格对照和可复现审计。

---

## 2. 数据与任务

### 2.1 CERT 日粒度用户行为

| 数据 | 用户数 | 天数 | 训练期 | 评价方式 |
| --- | ---: | ---: | --- | --- |
| r4.2 | 1,000 | 501 | 2010-01-02 至 2010-06-09，159 天 | validation + test |
| r5.2 | 2,000 | 517 | 2010-01-02 至 2010-04-15，104 天 | external test |
| r6.2 | 4,000 | 516 | 训练期 222 天；固定 1,000 用户子集训练 | 外部事件压力审计 |

r4.2 切分：

- train：索引 `0..158`
- validation：索引 `159..302`
- test：索引 `303..500`

训练期在第一个官方恶意事件之前结束。正常/恶意标签不参与 S1 标准化、训练样本选择或模型选择。

### 2.2 输入张量

每个版本形成：

```text
N × T × 27
```

特征包括：

- 登录/登出；
- 非工作时间登录；
- 其他主机登录；
- USB 连接/断开；
- 文件复制、压缩文件、可执行文件；
- 邮件、外部收件人、附件、邮件大小；
- HTTP、外部访问、唯一域名；
- 周末与星期周期编码。

计数特征使用 `log1p`。标准化只使用训练期非缺失值的均值和标准差。

### 2.3 47 维用户画像

实例软权重基于 23 个行为特征：

```text
23 个 median
+ 23 个 q90
+ 1 个训练期活跃比例
= 47 维用户画像
```

r4.2 的画像欧氏距离为对角标准化欧氏距离，可视为限制为对角协方差的马氏距离。

它不是 SoftCLT 原论文中的 DTW，只是针对 CERT 稀疏日计数数据的适配假设。

### 2.4 重要缺失值口径

S1 用 `-1` 表示：

- 当天没有登录；
- 当天没有登出；
- 无法形成有效活跃跨度。

旧画像代码曾执行：

```python
np.maximum(values, 0.0)
```

这会把 `-1` 误当零点。修正版为：

```text
-1 → NaN
逐用户 nanmedian / nanquantile
若某用户某特征完全缺失，才用训练期人群该特征中位数回填
```

修正后所有 T3/T4 结果均写入独立目录 `missing_safe_v2_20261117`，旧结果没有覆盖。

---

## 3. 从数据到检测分数

### 3.1 S1：数据准备

原始 CERT 日志聚合为用户日行为，构造 27 维时序张量和活跃掩码。

### 3.2 S2：自监督表征

基础方法是 TS2Vec：

- 因果 dilated CNN encoder；
- 层级时间对比；
- instance 对比；
- 遮罩一致性；
- 变长序列 padding 和 mask。

当前方法在 S2 接入 SoftCLT 的正样本软权重。

### 3.3 打分

主要分数包括：

1. `masked L1`：遮罩编码与普通编码的 L1 差异；
2. 余弦表征库：到训练期聚类中心最小的 3 个余弦距离均值；
3. 欧氏表征库：到训练期聚类中心最小的 3 个欧氏距离均值；
4. PAI 幅度融合：欧氏距离与 magG、T2 两个幅度组件融合。

所有 z-score、聚类和阈值统计只使用训练期。

### 3.4 评价

- 点级：PR-AUC、ROC-AUC；
- 固定预算：每日 Top-K，K ∈ {5,10,20}；
- 告警冷却：7 天；
- 事件命中：威胁实例是否被命中；
- r4.2：validation + test；
- r5.2：external test；
- r6.2：5 个威胁实例、78 个恶意账户日；
- SMD：28 台机器。

---

## 4. SoftCLT 实验矩阵

TS2Vec 原始层级损失：

\[
\mathcal L
=
\lambda\mathcal L_{\mathrm{inst}}
+
(1-\lambda)\mathcal L_{\mathrm{temp}}.
\]

当前五组模型：

| 模型 | instance 项 | temporal 项 | \(\lambda\) | 真实含义 |
| --- | --- | --- | ---: | --- |
| T0 | hard | hard | 0.5 | Native 层级对比参照 |
| T1 | 无 | hard | 0.0 | temporal-only hard |
| T2 | 无 | soft | 0.0 | temporal-only soft |
| T3 | soft | hard | 0.5 | instance-soft 单独效果 |
| T4 | soft | soft | 0.5 | 完整 SoftCLT |

### 4.1 必须注意的比较设计

- `T1 → T2`：固定没有 instance 项，只改变 temporal hard/soft；
- `T3 → T4`：固定 instance-soft，只加入 temporal-soft；
- `T2 vs T0` 同时改变两件事：移除 instance 项、temporal hard→soft；
- 因此不能用 `T2 vs T0` 单独证明“temporal-soft 有害”。

当前更准确的结论是：

> temporal-soft 不能替代跨用户 instance 对比，并且在 instance-soft 之上没有稳定独立收益。

---

## 5. SoftCLT 实际公式

### 5.1 instance-soft

对用户行为画像距离 \(d^X_{u,v}\)：

\[
w^I_{u,v}
=
\frac{2\alpha}
{1+\exp(\tau_I d^X_{u,v})}
.
\]

当前：

- `alpha = 0.5`
- 画像距离先非对角 min-max 到 `[0,1]`
- `tau_I = 1 / median(normalized_distance)`
- r4.2 修正版 `tau_I ≈ 4.8104`

非对角权重分位数为：

- 1%：`0.080`
- 50%：`0.269`
- 99%：`0.446`
- 最大：`0.500`

它是稠密但有梯度的核图，不是少数硬邻边。

### 5.2 temporal-soft

当前使用 sigmoid 时间核：

\[
w^T_{t,s}
=
\frac{2}
{1+\exp(\tau_T2^d|t-s|)}
\]

其中：

- \(d\) 是 TS2Vec 层级池化深度；
- \(2^d\) 随层级增大；
- 当前 `tau_T = 1.5`；
- 窗口和步长来自 TS2Vec 随机裁剪。

### 5.3 官方代码位置

```text
E:\MarketRisker\references\softclt_repo\softclt_ts2vec\models\soft_losses.py
E:\MarketRisker\references\softclt_repo\softclt_ts2vec\models\timelags.py
E:\MarketRisker\references\softclt_repo\softclt_ts2vec\soft_ts2vec.py
```

需要重点审查：

1. `inst_CL_soft` 和 `temp_CL_soft` 的 logits 排列是否正确；
2. `dup_matrix` 是否正确复制软权重；
3. \(\lambda=0\) 时是否完全移除 instance 项；
4. temporal hierarchy 中 `tau_T * 2**d` 的变化方向是否合理；
5. soft label 的 batch 切片是否为 `soft_labels[idx][:, idx]`；
6. 软权重是否在所有层级使用同一 user-pair matrix。

---

## 6. 当前理论假设：正样本图与表示几何

### 6.1 图视角

局部近似下，正样本加权对比项可看成图平滑能量：

\[
\mathcal E(Z)
\approx
\sum_{i,j}w_{ij}
\left\|z_i-z_j\right\|^2.
\]

完整 InfoNCE 还包含 log-sum-exp 负样本排斥和归一化，因此这只是机制近似，不是严格恒等式。

两种软权重的信息结构不同：

| 权重 | 使用内容 | 用户间边 | 用户内时间边 | 主要作用 |
| --- | --- | --- | --- | --- |
| instance-soft | 是，画像距离 | 是 | 否 | 修复相似用户之间的假负例 |
| temporal-soft | 否，只看索引差 | 否 | 是 | 用户内时间低通平滑 |

### 6.2 instance-soft 为什么可能有效

标准 instance 对比只把同一用户的两个增强视图作为正样本，其他用户全为负样本。CERT 中不同用户可能共享相同岗位、工作时间和正常行为模式，因此会出现假负例。

instance-soft 加入内容相关的跨用户弱正边，预期同时带来：

- 数据 kNN 邻域保持率提高；
- 局部数据距离与表示距离的秩相关提高；
- 正常流形有效秩提高；
- 不会被硬类别压缩成低秩塌缩。

### 6.3 temporal-soft 为什么可能没有独立收益

temporal-soft 的图只由索引差决定。它相当于沿每个用户的时间轴做低通平滑。

潜在问题：

1. CERT 相邻日期可能跨工作日、周末、休假、异常起点；
2. 时间接近不等于行为语义相似；
3. 点异常经常表现为用户内高频偏离，平滑会压低异常方向；
4. 持续异常本身可能时间平滑，temporal prior 无法识别其为异常；
5. T0 已经包含 hard temporal 项，soft temporal 只是重加权；
6. 当 \(\lambda=0\) 时，跨用户 instance 项被完全删除。

因此 temporal-soft 即使有局部平滑作用，也可能无法形成异常检测所需的跨用户判别几何。

---

## 7. 当前机制指标

机制审计固定采样验证期正常日，使用与原始行为特征空间的 Top-10 邻域一致率、局部距离 Spearman、有效秩和表示范数。

| 模型 | kNN overlap | 局部距离 Spearman | 有效秩 | 平均范数 |
| --- | ---: | ---: | ---: | ---: |
| T0 | 0.3902 | 0.4401 | 8.884 | 3.464 |
| T1 | 0.2601 | 0.3974 | 5.110 | 3.271 |
| T2 temporal-only soft | 0.2149 | 0.4498 | 7.669 | 1.972 |
| T3 instance-soft | 0.4977 | 0.5034 | 14.696 | 1.380 |
| T4 full soft | 0.4952 | 0.4933 | 14.375 | 1.357 |

解释：

- T1→T2：temporal-soft 比 hard temporal 略好，但平均范数从 `3.271` 降到 `1.972`，kNN overlap 继续下降；
- T0→T3：instance-soft 显著改善数据邻域保持和有效秩；
- T3→T4：加入 temporal-soft 后 PR、ROC、kNN、有效秩均轻微下降。

对应文件：

```text
E:\MarketRisker\work\cert\experiments\representation_preservation\canonical\softclt_mechanism_geometry.csv
E:\MarketRisker\work\cert\experiments\representation_preservation\canonical\softclt_geometry_T4_minus_T3.csv
```

---

## 8. 当前 canonical 检测结果

### 8.1 T0–T4

三 seed 均值，K=10：

| 模型 | validation PR | test PR | validation ROC | test ROC | test K=10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 | 0.014371 | 0.007447 | 0.809283 | 0.816112 | 12.33/24 |
| T1 | 0.011258 | 0.005496 | 0.755798 | 0.759326 | 6.00/24 |
| T2 tau1.5 | 0.012622 | 0.006252 | 0.771563 | 0.772159 | 10.00/24 |
| T3 instance-soft | 0.016950 | 0.008922 | 0.835534 | 0.842132 | 10.33/24 |
| T4 full soft | 0.016754 | 0.008867 | 0.834066 | 0.841593 | 10.33/24 |

T1→T2 的结果说明 temporal-soft 不是完全无效，但仍远低于含 instance 项的模型；T3→T4 没有独立点级收益。

### 8.2 修复后的 r4.2 联合实验

三 seed 均值，K=10：

| 方法 | PR-AUC | ROC-AUC | K=10 |
| --- | ---: | ---: | ---: |
| T0 + 欧氏库 | 0.011216 | 0.875951 | 10.00/24 |
| T0 + PAI-A1 | 0.010946 | 0.865935 | 11.67/24 |
| T4 + 余弦库 | 0.016704 | 0.905913 | 13.33/24 |
| T4 + 欧氏库 | 0.014908 | 0.893723 | 11.33/24 |
| T4 + PAI-A1 | 0.015148 | 0.880723 | 14.00/24 |

### 8.3 r5.2/r6.2

| 数据/方法 | PR-AUC | ROC-AUC | K=10 |
| --- | ---: | ---: | ---: |
| r5.2 T4+PAI-A1 | 0.007204 | 0.875681 | 32.33/99 |
| r6.2 T4+PAI-A1 | 0.000301 | 0.827063 | 2.00/5 |

r6.2 只有 5 个威胁事件，不能支撑“外部确认成功”。

### 8.4 三 seed 图谱谱实验

新增实验直接计算：

\[
R_W(Z)
=
\frac{\operatorname{tr}(Z^\top L_WZ)}
{\operatorname{tr}(Z^\top D_WZ)},
\]

分别使用 instance 图和 temporal 图；同时评估“到图邻居加权中心的距离”对异常的 ROC/PR。

| 模型 | instance Dirichlet | temporal Dirichlet | instance residual ROC | temporal residual ROC |
| --- | ---: | ---: | ---: | ---: |
| T0 | 0.2937 | 0.2829 | 0.7950 | 0.6071 |
| T1 | 0.0936 | 0.4555 | 0.7756 | 0.6325 |
| T2 | 0.1265 | 0.3457 | 0.7606 | 0.6966 |
| T3 | 0.3456 | 0.2452 | **0.8407** | 0.7651 |
| T4 | 0.3352 | **0.2395** | 0.8398 | **0.7717** |

配对差值：

| 对比 | 结果 |
| --- | --- |
| T2−T1 | temporal Dirichlet `-0.1098`；temporal residual ROC `+0.0642`；3/3 seed 同向 |
| T3−T0 | instance Dirichlet `+0.0519`；instance residual ROC `+0.0458`；temporal residual ROC `+0.1580`；3/3 seed 同向 |
| T4−T3 | temporal Dirichlet `-0.00565`；temporal residual ROC `+0.00660`；但 instance residual ROC 1 正 2 负，最终 PR/ROC 轻微下降 |

因此当前解释需要精确化为：

1. temporal-soft **确实**降低 temporal Dirichlet energy，并改善时间高频残差的单独可分性；
2. 这个局部收益没有传递到最终表征库评分；
3. instance-soft 不是全局低通压缩，因为 instance Dirichlet energy 反而上升；更合理的解释是内容感知的正样本重心重校准；
4. 需要审查 normalized Dirichlet energy 和 residual AUC 是否足以支撑这一解释。

对应文件：

```text
E:\MarketRisker\work\cert\experiments\representation_preservation\s2_graph_spectral\three_seed_20261117
E:\MarketRisker\work\cert\experiments\representation_preservation\canonical\softclt_spectral_contrasts.csv
E:\MarketRisker\work\cert\s2_soft_graph_spectral_analysis.py
```

### 8.5 E11 iForest 外部基线

新增 27 维用户日特征 + IsolationForest，作为经典表格方法基线。只在训练期拟合，固定 `n_estimators=500`、`random_state=20261116`。

| 方法 | test PR-AUC | test ROC-AUC | K=5 | K=10 | K=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| iForest-27D | 0.007440 | 0.815921 | 9.00/24 | 11.00/24 | 14.00/24 |
| T0 mask | 0.007440 | 0.815069 | 6.33/24 | 9.00/24 | 14.00/24 |
| T4+PAI-A1 | **0.015148** | **0.880723** | **10.00/24** | **14.00/24** | **19.00/24** |

解释：

- iForest 的点级 PR 与 T0 基本相同，ROC 略高；
- iForest 在 K=5、K=10 上高于 T0；
- T4+PAI-A1 在 PR、ROC、K=5、K=10、K=20 上均高于 iForest；
- 不能把 iForest 写成很弱基线，否则反而不可信。

---

## 9. PAI-A1/A3 与权重

### 9.1 两个同名问题

- E0/E1 `S5_pai_full_global` 是 A3：用户级五通道 max 后取 top-2 均值；
- E3 `A1_user_channel_mean_pai` 是 A1：用户级五通道 max 后取均值；
- 25.0/46 是 A1；
- 22.67/46 是 A3。

### 9.2 权重

项目权重：

\[
S_{\mathrm{project}}
=
z_{\mathrm{euc}}
+0.6z_G
+0.4z_{T_2}.
\]

官方 PAI 代码权重：

\[
S_{\mathrm{official}}
=
0.6z_{\mathrm{euc}}
+0.4z_G
+0.2z_{T_2}.
\]

官方权重不是“更好权重”，只是固定敏感性对照。完整表见：

```text
E:\MarketRisker\work\cert\experiments\representation_preservation\canonical\pai_weight_sensitivity_summary.csv
```

---

## 10. 已解决的 D1–D8 口径问题

- D1：3.8 旧生产协议与 E0–E10 原始 S2 分表；
- D2：PAI-A1 与 S5-A3 分开；
- D3：E2 汇总从逐种子文件重建；
- D4：magG 点级为负、事件级局部正向，禁止全称判断；
- D5：脚本不再覆盖顶层汇总，新结果写入 run tag；
- D6：只把 8 个产物标为设计名，不伪造空文件；
- D7：时间 `-1` 缺失 bug 已修复并重跑；
- D8：官方 TS2Vec 功能分析已核对。

---

## 11. 请外部 AI 重点审查的数学问题

### A. instance-soft 的严格性

1. 将正权重 InfoNCE 近似成图拉普拉斯平滑，需要什么条件？
2. 在归一化球面表示上，该近似的误差项是什么？
3. soft instance 核是否必然提高 kNN 保持，还是只在数据核与真实语义一致时成立？
4. 稠密弱正边是否可能导致另一种 collapse？为什么本次有效秩没有下降？
5. 训练期画像与验证/测试行为漂移时，错误正边如何影响误报？

### B. temporal-soft 的严格性

1. sigmoid 时间核的拉普拉斯谱是什么？哪些频率被压低？
2. 在随机裁剪和层级池化下，实际频率响应是否与连续时间假设一致？
3. `tau_T * 2^d` 是否会造成粗层级过度尖锐、只在极短邻域形成正样本？
4. temporal-only 的跨用户不可识别性能否形式化为：不同用户表示可在共同旋转/低秩子空间内分离？
5. “点异常被低通平滑抑制”的假设能否通过合成数据验证？
6. normalized Dirichlet energy 下降是否必然意味着异常证据下降？为什么 T4−T3 的时间残差 ROC 反而上升？
7. instance Dirichlet energy 上升但 kNN/残差 AUC 改善，是否说明“图平滑”不是 instance-soft 的正确数学描述？

### C. 比较是否真正隔离

请核对：

- T1/T2 是否只改变 temporal hard/soft；
- T3/T4 是否只改变 temporal hard/soft；
- T0/T3 是否只改变 instance hard/soft；
- 三组比较是否共享相同窗口、种子、epoch、学习率和评价。

如果任何一项不成立，当前理论解释需要降级。

---

## 12. 请外部 AI 重点审查的实现问题

重点代码：

```text
E:\MarketRisker\work\cert\e2_softclt_training.py
E:\MarketRisker\work\cert\e2_softclt_instance.py
E:\MarketRisker\work\cert\e2_mechanism_audit.py
E:\MarketRisker\work\cert\e2_instance_mechanism.py
E:\MarketRisker\work\cert\e3_multichannel_amplitude.py
E:\MarketRisker\work\cert\e4_joint_instance_pai.py
E:\MarketRisker\work\cert\e4_r52_joint_analysis.py
E:\MarketRisker\work\cert\e4_r62_joint_analysis.py
```

官方 SoftCLT：

```text
E:\MarketRisker\references\softclt_repo\softclt_ts2vec\models\soft_losses.py
E:\MarketRisker\references\softclt_repo\softclt_ts2vec\models\timelags.py
E:\MarketRisker\references\softclt_repo\softclt_ts2vec\soft_ts2vec.py
```

请逐项检查：

- pairwise distance 是训练期还是全量数据；
- soft labels 是否只来自无标签正常训练数据；
- `tau_inst` 是否由训练集决定；
- `tau_temp` 是否没有使用测试集；
- 随机裁剪是否导致同一日期对应关系错误；
- batch 中 soft label 的切片顺序是否对应 encoder 的输入顺序；
- mask 位置是否参与 instance/temporal soft positive；
- `lambda=0` 时 instance 分支是否真正消失；
- temporal hierarchy 的层级数是否与 timelag 生成一致；
- 聚合距离的 Top-3 聚类中心是否只用训练期；
- PAI 的 z-score 是否只用训练期；
- 事件命中是否使用了正确时间窗口；
- 重复 seed 是否真正重新初始化而不是只改评价。

---

## 13. 请外部 AI 重点审查的统计问题

1. 只有 3 个训练 seed，Wilcoxon 双侧最小 p 为 0.25，不能声称显著；
2. 三个 seed 共享同一数据切分和同一官方事件集，不是独立数据重复；
3. r6.2 只有 5 个事件，K=10 命中 2/5 仍不稳定；
4. SMD 只有 1 个训练 seed；
5. 多指标测试未统一做多重比较校正；
6. 点级 PR/ROC 与固定 K 事件命中是不同评价目标，不能互相替代；
7. 如果要做置信区间，应对训练 seed、事件和用户做层级 bootstrap；
8. 不要用测试集扫描 `tau_T`、`tau_I` 或融合权重。
9. 谱实验的 residual AUC 是机制诊断，不是最终检测器；不能拿它替代表征库 PR/ROC。

---

## 14. 如果计算资源允许，建议外部 AI 设计的新实验

优先级从高到低：

1. **合成异常实验**
   - 点突变；
   - 上下文异常；
   - 集体异常；
   - 平滑持续漂移；
   - 在相同正样本图上比较 T1/T2/T3/T4。

2. **时间图频率响应实验**
   - 改变 `tau_T`；
   - 固定 instance 项，只检查 temporal-soft；
   - 分析低频/高频异常召回和表示范数。

3. **instance 核消融**
   - hard instance；
   - 47 维画像核；
   - 随机画像核；
   - 错误画像置换核；
   - 检查提升是否来自真实行为相似性。

4. **错误正边敏感性**
   - 随机打乱部分 user pair；
   - 检查 kNN、有效秩和固定 K 事件命中如何变化。

5. **不同异常局部性**
   - 点异常应更支持“temporal low-pass 有害”；
   - 缓慢漂移应更支持“temporal smoothness 有益”。

---

## 15. 当前允许使用的结论

可以说：

- instance-soft 在 CERT r4.2 上改善了正常邻域保持、局部距离一致性和有效秩；
- temporal-soft 确实降低 temporal Dirichlet energy，并改善时间高频残差的局部可分性；
- temporal-soft 在当前任务中没有在 instance-soft 基础上形成稳定端到端收益；
- temporal-only 方案即使使用 soft temporal，也不能替代跨用户 instance 对比；
- T3→T4 的 PR/ROC/kNN/秩小幅下降，但效应量很小；
- T4+PAI-A1 在 r4.2 test 上相对 iForest-27D 的 PR、ROC、K=5、K=10、K=20 均更高；
- iForest 已接近 T0，说明单日表格离群点本身包含信息，不能把它描述成很弱基线。

不能说：

- “temporal-soft 在所有时序异常检测中无效”；
- “instance-soft 的因果关系已经严格证明”；
- “三个 seed 的结果统计显著”；
- “r6.2 外部事件审计成功”；
- “PAI 或 magG 全面有效”；
- “官方权重最优”；
- “资料包解压即可完整复现”。

---

## 16. 关键文件与哈希

canonical 目录：

```text
E:\MarketRisker\work\cert\experiments\representation_preservation\canonical
```

| 文件 | SHA-256 |
| --- | --- |
| `canonical_metric_summary.csv` | `80C8972882DD7E8D9A5F9D0DD0DA7F0D497FEFAF4C13156E83512936D48C7D92` |
| `canonical_metrics_by_seed.csv` | `4C1BDBA763249F9B58E4292360A8AB837DA6A6AE76F221BF7395B24D4B76BBB4` |
| `pai_weight_sensitivity_summary.csv` | `2AA4B44EEDA7ED3F4C32047522E80BCC35ECA9CE37EEB9CE5E40A95A1955ECB8` |
| `joint_paired_comparisons.json` | `DFA19B3A497BA99BD7B59EB77975E1A26DC61C7167F7D6E626A8F1860238FCC6` |
| `softclt_mechanism_geometry.csv` | `61B13AD4D20704BBBB38BD00E5D8169E2788093C1B6EA590CAF38334D0348FCF` |
| `softclt_spectral_contrasts.csv` | `2CFA829DC45A94D21F16D51BC6E1A1453C0F74CF1D68DB80998A0C557F8BB577` |
| `spectral_metrics_by_seed.csv` | `B404D78978B0D047F19DCDC29CE182198F93B1447E8F40D48E79D51F912FCCC9` |
| `spectral_metrics_summary.csv` | `1822135BC94443A2D519F56488D3DB85343E9DA5B67D45B7BAFF451114A71EFD` |
| `e11_iforest/metric_summary.csv` | `FBFEE4D2DF03FF8A899BF7A835862F6DE7EF0597AB1516F043C4FAC6DDFF3A47` |
| `e11_iforest/iforest_vs_reference.csv` | `961FD07916CD923FEE6764CD383E4778ACF577551CF020033DA55CE6DC8CCC8F` |

代码哈希：

| 文件 | SHA-256 |
| --- | --- |
| `e2_softclt_training.py` | `5A47ECAF5B96B1787ABE941295BBF89C62E1BC257F56C3CCED69DFF2A25F16F4` |
| `e2_softclt_instance.py` | `3CC57B02FD8389954E8EAD69FD4615217406A8A98582012D78FEE48A11DFD9F9` |
| `e3_multichannel_amplitude.py` | `183C6326641AE109735C0AF983B914A855A2690047F87E8A991225DD01353930` |
| `soft_losses.py` | `80B0D734C85034DC9205C21C2763CE56E7DFE372A47E764A133856B83D774CF4` |
| `timelags.py` | `76B1379EFCB5FA1F11587E83FA4E333154C2629BDDCEB899D545FAEDF08B0F0C` |
| `s2_soft_graph_spectral_analysis.py` | `1C47D7081DA90EBF637184FA4E99A528382F7F3303EDF6BD0195FC333F92ADF9` |
| `e11_iforest_baseline.py` | `1E52036E8E7B1D87F04741ED86F755F1986059BCC9C46A2733F5C219772DBFAA` |

资料包：

```text
E:\MarketRisker\research\handoff_dual_end_S1-S5_20261117
E:\MarketRisker\research\MarketRisker_D1-D8_修复与审计_20261117.zip
```

背景与叙述材料：

```text
C:\Users\Lenovo\Desktop\AI自监督时序实验.txt
C:\Users\Lenovo\Downloads\异常检测大创申请书初稿.docx
E:\MarketRisker\research\handoff_dual_end_S1-S5_20261117\references\异常检测大创申请书初稿.docx
C:\Users\Lenovo\Downloads\有争议数据清单（已加修正决定）.md
C:\Users\Lenovo\Downloads\SoftCLT_几何机制与成立条件_20261117.md
C:\Users\Lenovo\Downloads\表征信息保全研究完整报告_v2_20261117.md
```

机器环境：

```text
Windows
PowerShell
Python: C:\Users\Lenovo\.conda\envs\ts2vec\python.exe
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
```

---

## 17. 复算命令

查看 canonical：

```powershell
Get-ChildItem `
  'E:\MarketRisker\work\cert\experiments\representation_preservation\canonical'
```

重建 canonical / PAI 权重敏感性：

```powershell
& 'C:\Users\Lenovo\.conda\envs\ts2vec\python.exe' `
  'E:\MarketRisker\work\cert\build_canonical_audit.py'
```

重建 SoftCLT 几何摘要：

```powershell
& 'C:\Users\Lenovo\.conda\envs\ts2vec\python.exe' `
  'E:\MarketRisker\work\cert\build_softclt_geometry_summary.py'
```

重跑三 seed 谱分析：

```powershell
& 'C:\Users\Lenovo\.conda\envs\ts2vec\python.exe' `
  'E:\MarketRisker\work\cert\s2_soft_graph_spectral_analysis.py' `
  --seeds 20261116 20261117 20261118 `
  --output-tag three_seed_20261117
```

注意：不要在原结果目录上覆盖运行，除非明确使用新的 `--output-tag` 或 `--run-tag`。

---

## 18. 最小上传清单

如果外部 AI 无法访问本机路径，至少提供以下内容：

### 18.1 必传文档

1. 本交接说明；
2. `有争议数据清单（已加修正决定）.md`；
3. `SoftCLT_几何机制与成立条件_20261117.md`；
4. `下一课题_数学章节与实验对照表_简版_20261117.md`；
5. `表征信息保全研究完整报告_v2_20261117.md`。

### 18.2 必传结果

1. `MarketRisker_D1-D8_修复与审计_20261117.zip`；
2. 如不能传 zip，至少上传：

```text
canonical_metric_summary.csv
canonical_metrics_by_seed.csv
joint_paired_comparisons.json
pai_weight_sensitivity_summary.csv
softclt_mechanism_geometry.csv
softclt_geometry_T4_minus_T3.csv
softclt_spectral_contrasts.csv
s2_graph_spectral/three_seed_20261117/spectral_metrics_by_seed.csv
s2_graph_spectral/three_seed_20261117/spectral_metrics_summary.csv
e11_iforest/metric_summary.csv
e11_iforest/iforest_vs_reference.csv
```

### 18.3 必传代码

```text
e2_softclt_training.py
e2_softclt_instance.py
e2_mechanism_audit.py
e2_instance_mechanism.py
soft_losses.py
timelags.py
soft_ts2vec.py
s2_soft_graph_spectral_analysis.py
```

如果还要审查 PAI：

```text
e3_multichannel_amplitude.py
e4_joint_instance_pai.py
e4_r52_joint_analysis.py
e4_r62_joint_analysis.py
e11_iforest_baseline.py
```

轻量审计包不包含完整原始日志、47 维画像大文件、`.npz/.pth` 模型和所有逐日中间数组。因此外部 AI 可以复核汇总、公式、代码和已提供的逐 seed 结果，但不能在无原始数据/模型的情况下完整重训。

---

## 19. 可直接复制给外部 AI 的审查提示词

```text
你是机器学习方法论、时序表示学习和统计实验审查专家。

请阅读我提供的 MarketRisker 项目交接文档以及你获得的代码/CSV。
不要替我美化结论，也不要只给摘要。请完成以下工作：

1. 用“通过 / 需修改 / 无法判定”检查 D1–D8 和 T0–T4 的定义是否一致；
2. 逐一检查 SoftCLT 官方实现和当前项目的：
   - instance soft weight；
   - temporal soft weight；
   - lambda；
   - hierarchical timelag；
   - batch label slicing；
   - 训练/验证/测试泄漏；
3. 从数学上审查“正样本图 → 图平滑能量 → 邻域保持/低通滤波”的推导：
   - 哪些步骤严格；
   - 哪些只是近似；
   - 需要什么额外条件；
   - 能否给出反例；
4. 检查 T1/T2、T3/T4 是否真的只改变 temporal soft；检查 T0/T3 是否真的只改变 instance hard/soft；
5. 重新计算所有你能从 CSV 复核的均值、差值和配对统计；
6. 判断“instance-soft 有效、temporal-soft 无独立收益”是否可以这样写；
7. 设计至少三组可证伪实验，区分：
   - 点异常；
   - 上下文异常；
   - 平滑缓慢漂移；
   - 集体异常；
8. 明确指出当前结论中哪些只能写成描述性结果，不能写成因果或普遍定理。

输出格式：

A. 结论审计表：主张 | 当前证据 | 判定 | 需要修改的措辞
B. 数学审查：公式、假设、推导、反例
C. 代码审查：文件、行号、问题、修复建议
D. 统计审查：seed、事件、置信区间、多重比较
E. 必须补做的实验（按优先级）
F. 可直接写入论文的理论段落和证据边界
```

