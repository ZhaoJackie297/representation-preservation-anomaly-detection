# AdaGraph-SoftCLT CCF-B 研究计划审阅包

版本：v1.2.6

冻结日期：2026-11-18

代码分支：`ccf-b`

仓库：<https://github.com/ZhaoJackie297/representation-preservation-anomaly-detection>

## 给审阅 AI 的说明

你是一名熟悉自监督时序表示、软对比学习、内部威胁检测、图关系学习和统计实验设计的审稿人。请只基于本文件指出：

1. 哪些问题会让整套推断失效；
2. 哪些是方法风险，哪些只是实现或写作风险；
3. 每个阻断问题的最小修正方案；
4. 当前计划是否仍足以支撑 CCF-B 目标；
5. 如果不足以支撑，应该补哪一项最关键的证据。

约束：

- 不编造文献；
- 不把新增模块数量当创新；
- 不建议使用真实测试标签调参或早停；
- 不把 r1、r5.2 或小型外部集写成盲测；
- 指出“方法不成立”和“数据门禁失败”的区别。

## 1. 项目背景

论文 v5 已冻结为 R0，定位 CCF-C 上沿。当前目标是冲击 CCF-B。

主方法候选为 `AdaGraph-SoftCLT`：

- 主干仍为 TS2Vec；
- 修改点是 TS2Vec 层次实例对比中的用户软关系权重；
- 47 维行为画像提供用户关系先验；
- EMA 教师表示提供有限的表示反馈；
- 每轮只更新一张用户软图，并用 KL 信任域限制漂移。

当前不是已完成方法，而是候选方法。

## 2. 方法

### 2.1 画像先验

训练期内构造每个用户的 47 维画像，排除部门、角色、团队和 LDAP 派生字段。

```text
P_raw(u,v) = max(exp(-D_profile(u,v)^2 / (2h^2)), 1e-12)
h = median(D_profile)
```

对角置 0，逐行归一化为 `P`。

### 2.2 EMA 表示距离

每阶段开始时 EMA 教师复制学生权重。训练期内：

```text
d_tilde(u,v) = (1 - cosine(z_u,z_v)) / row_median
```

教师停止梯度，行中位数低于 `1e-6` 时截断。

### 2.3 支撑集

每个用户按以下分数取 top-m：

```text
score(u,v) = P(u,v) * exp(-d_tilde(u,v) / tau)
```

在支撑集上重新归一化 `P_m`，KL 都在支撑集上计算。

正式主网格 `m in {5,10}`；`m=20/50` 只在图边覆盖和算力审计通过后作敏感性。

### 2.4 KL 信任域

逐用户求解：

```text
min_W <W,d_tilde> - tau H(W)
s.t.  KL(W || P_m) <= epsilon
```

得到：

```text
W_beta(u,v) ∝ P_m(u,v)^beta *
             exp(-(1-beta) * d_tilde(u,v) / tau)

beta = lambda / (tau + lambda), 0 <= beta < 1
```

KL 随 beta 单调下降，用二分法满足约束。

### 2.5 到 SoftCLT

- 自权重为 1；
- 非自邻接权重为 `rho * W_beta`；
- 主模型 `rho=1.0`；
- 敏感性 `rho={0.25,0.5}`；
- A3-hard 的每个支撑邻居取 `1/m`，邻居总质量仍为 1；
- 邻接总质量固定，不随 m 增长。

### 2.6 两轮更新与 2×2 机制端点

- A1-R0：R0 原始随机 shuffle、稠密画像权重，`B`；
- A1-Cov：固定画像图、共同覆盖批次，`B`；
- A1-sparse：按 P 选 top-m，权重取 P_m；
- A1-sparse-EMA：支撑集与 AdaGraph 相同，权重取 P_m，是真实 `epsilon=0` 端点；
- A1-repSupport-P：表示距离支撑集、权重取归一化 P_m；
- A3：一次图更新，`2B`；
- A4：两次图更新，`3B`；
- A1+：固定图，匹配 A3/A4 的总步数；
- A9：固定图，三段权重续训和逐段优化器重置；
- A3-noAnchor：支撑集与权重只由表示距离决定；
- A3-epsInf：mixed 支撑集、纯表示权重；
- A3-noEMA：当前表示替代 EMA；
- A3-product：替代融合规则，不用 KL 缩放；
- A5a：度保持随机图；
- A5b：画像节点置换图；
- A7：NNCLR；
- A8a/A8b：假负检测/抵消基线。

完整 2×2：

| 支撑集锚 | 权重锚 | 变体 |
| --- | --- | --- |
| 有 | 有 | A1-sparse-EMA |
| 有 | 无 | A3-epsInf |
| 无 | 无 | A3-noAnchor |
| 无 | 有 | A1-repSupport-P |

## 3. 新发现的实现阻断：图边没有可靠落批

官方 TS2Vec 训练实例是“每用户一条序列”，batch size 为 16。SoftCLT 使用：

```python
soft_labels_batch = soft_labels[idx][:, idx]
```

只有同批用户对会进入实例软对比。

随机 batch 下，任一给定边至少出现一次的概率为：

| 数据 | B | 1B | 2B | 3B |
| --- | ---: | ---: | ---: | ---: |
| r4.2，1,000 用户 | 620 | 13.8% | 25.8% | 36.0% |
| r5.2，2,000 用户 | 1,250 | 7.2% | 13.9% | 20.2% |

因此原计划可能写出了全局 W，但多数正关系从不进入梯度。

v1.2.2 的修正是 `CommonCoverageBatchSampler`：

1. 从 A3、A5a、A5b 形成边集 `E_A3 ∪ E_A5a ∪ E_A5b`；
2. batch 16 中固定 8 个覆盖设计槽位和 8 个随机填充槽位；
3. 按固定 hash 排序并贪心选择 8 个设计用户；
4. 同 seed/stage 的所有主对照重放相同批次表；
5. A0-matched 是同批次 + 硬实例对比；
6. `coverage_A3`、`coverage_A5a`、`coverage_A5b` 都必须 `>=80%`；
7. 不允许有用户零边曝光；
8. 记录三类 `active_pair_mass`、每用户边曝光和教师-学生距离；
9. 10 个正式 seed 的 `batch_table_sha256` 必须全部唯一；
10. 报告 seed 间用户对 Jaccard；
11. 失败状态为 `IMPLEMENTATION_BLOCKED` 或 `INVALID_SEED_RANDOMIZATION`，不是方法失败。

合成 smoke（1,000 用户、三张约 6,300 边图、每图 10 度）：

- 80% 三图覆盖可达到；
- 10 个正式 seed 均完成；需要约 1,134–1,146 个 stage batch，约为原 `B=620` 的 1.83–1.85 倍；
- 没有零曝光用户；
- 10 个 seed 的批次表哈希全部不同；
- 相邻 seed 用户对 Jaccard 约 0.173–0.175。

该结果只证明采样器在合成图上可行，不证明真实图或 AdaGraph 有效。

共同批次表仍携带 A3/A5a/A5b 的共现信息，因此它不是无偏采样器。50% 随机填充和 A5a/A5b 联合覆盖用于降低单边倾斜，但不能完全消除选择效应。

预注册分解顺序：

```text
A0-official -> A0-matched -> A1-Cov -> A3/A4
```

A4 vs A0-official 不得写成方法增益；A0-matched 下降只表示批次更难。

边曝光理论：

```text
p = b(b-1) / [N(N-1)]
lambda = n_b p
Pr(exposure >= 1) = 1 - (1-p)^n_b
```

定向批次若保证每条目标边至少出现 c 次，则曝光次数至少为 c。该命题解释覆盖修复的必要性，不证明 AdaGraph 有效。

另有一条预注册停止规则：

```text
positive_PR: Delta_PR >= +0.0005
positive_K10: Delta_K10 >= +max(0.02, 1/N_events)

若 A1-Cov vs A0-matched 在两个 endpoint 上
都未达到上述正向阈值
则立即停止 AdaGraph 主实验。
```

2 seed 时，同一 endpoint 必须两个 seed 都超过阈值。只有一个 endpoint 达标时，标记为 `FRAGILE_SINGLE_ENDPOINT`，先在 r4.2 10-seed 复验，不得直接进入 r5.2。Pilot 不用于调参。

## 4. 数据审计

### CERT r1

本地 `E:\MarketRisker\work\cert\r1_audit.json`：

```json
{
  "answers": {
    "available": true,
    "r1_labels_present": false,
    "r1_label_rows": []
  },
  "label_status": "r1 has no malicious scenario labels in the official answer key"
}
```

r1 有约 1,000 个用户和大量日志，但没有官方恶意场景标签。

v1.2.2 删除其确认集地位，只保留无标签工程诊断。

### CERT r5.2

- 2,000 用户；
- 104 天干净训练期；
- 413 天评价期；
- 99 个事件；
- 1,307 个恶意账户日。

它用于 v1.2.2 冻结后的锁定参数复验，但 R0 已见，因此不是盲测。

### SPEDIA

已下载并核对：

- Zenodo DOI：`10.5281/zenodo.15525713`；
- CC BY 4.0；
- 官方 MD5 与本地下载一致；
- 72,250 行、25 列；
- 17 个唯一 `User` 值；
- 25 个活动日期；
- 41,541 个异常行；
- 第一日已有 1,477 个异常行；
- 官方描述为 30 天演练、真实攻击、模拟正常和 CERT 派生事件混合、平衡恶意/正常分布。

结论：可作会话级压力测试，不能作主用户图确认集。

补充：2026 IJACSA cross-dataset benchmark 删除 CERT 派生行后得到 20,619 条 real-only SPEDIA 事件、82/256 个正用户日，并报告跨数据集 PR-AUC 明显下降。该工作支持“不能只在 CERT 上验证”，但其表示效应依赖源域，没有声称一般性的图方法优势。

### TWOS

- 论文 DOI：`10.1145/3139923.3139929`；
- 数据 DOI：`10.13140/RG.2.2.28309.42729`；
- 五天受控红蓝对抗；
- 规模和许可证仍需审计。

结论：外部压力测试，不作主确认。

### ChimeraLog

- 论文 arXiv：2508.07745，NDSS 2026；
- 官方页：<https://sites.google.com/view/chimerallm>；
- 官方代码：<https://github.com/fish98/Chimera>；
- 三组织，每组织 20 人；
- 一个月、15 类攻击、六类日志；
- 官方页当前仍显示数据正在整理和迁移；
- 官方 GitHub README 进一步说明：作者毕业后学校 OneDrive 账号受影响，旧 ChimeraLog 下载链接已不可访问，正在寻找新托管。

结论：托管恢复后的备选，当前不具备可下载、可审计的确认条件。

### CLUE-LDS

官方资源：

- Zenodo：<https://zenodo.org/records/7119953>；
- DOI：`10.5281/zenodo.7119953`；
- 论文 DOI：`10.1109/BigData55660.2022.10020672`；
- 代码：<https://github.com/ait-aecid/clue-lds>；
- 数据 CC BY 4.0；代码 GPL-3.0；
- `clue.zip` 约 635 MB，解压约 14.5 GB。

官方描述：

- Huemer Group 真实 hBOX/Nextcloud 云存储日志；
- 约 5,000 万事件；
- 超过 5,000 用户；
- 2017-07-07 至 2022-09-29，1,910 天；
- 49 类登录、文件、分享、组、配置事件；
- 官方示例统计 5,389 用户、83,147 用户日。

标签：

- 原始数据只有正常行为；
- 基于官方切换算法实现，由本项目固定 seed 构造半合成账号劫持；
- 标签 seed=`20261101`，训练 seed=`20261116..20261125`；
- 24 pairs，产生 48 个用户级事件；
- 切换前至少 35 个活跃日、切换后至少 30 个活跃日；
- 切换日期只进入 2022-03-01 至 2022-08-30 的评价窗口；
- 不得写成官方异常标签、官方受控注入或真实攻击标签。

优势：

- 可直接下载；
- 真实正常用户行为；
- 应用级语义接近 CERT；
- 有文件、组、分享关系，可构造用户图；
- 有官方切换算法，可由本项目固定 seed 复现半合成标签生成；
- 可用固定预算 Top-K 和冷却期评价。

限制：

- 标签是本项目生成的半合成账号劫持，不是真实恶意内部人员；
- 需要自定义时间切分，避免切换点跨训练期；
- 角色字段不完整，用户图需从组和分享关系构造；
- 全量预处理仍需审计。

定位：`D_app`，首要应用级半合成外部确认候选。

实际执行结果（2026-11-18）：

- `clue.zip` MD5 与官方一致；
- `clue.json` 全量审计 50,522,931 事件、5,389 用户、1,910 数据日、83,147 用户日；
- 本项目固定 seed 生成 24 pairs = 48 个用户级半合成账号劫持事件；
- 处理为 838 用户、1,911 个连续日期槽、52 维特征；
- 48 个事件全部有评价期活动，且没有目标用户训练期全空；
- A3/A5a/A5b 真实 top-10 图 10-seed 覆盖全部通过；
- 覆盖 batch 948–961，零曝光用户 0，10 张批次表全不同；
- 画像先验熵中位数 6.49，接近均匀上界 6.73；
- top-10 `P_m` 触发 `FLAT_PROFILE_PRIOR`，89.98% 用户行熵高于 0.95·log(10)；
- 标签 pair 相似度与画像亲和度的 Spearman 为 −0.199，距离审计通过；
- 组/分享图有 565/838 孤立用户，只作辅助关系参照；
- CPU 真实子集软权重训练 step 已跑通，loss 5.8983。

24 个有效 pair 的 80% 功效召回 MDE 约为 23.0–29.5 个百分点，因此 CLUE-LDS 只能作有限功效外部效应复验。这完成了数据与 Phase 0 覆盖门禁，但还没有完成端到端检测实验。

### LANL

官方资源：

- 页面：<https://csr.lanl.gov/data/cyber1/>；
- DOI：`10.17021/1179829`；
- 许可：CC0；
- 58 个连续日；
- 约 12 GB 压缩；
- 1,648,275,307 条事件；
- 12,425 用户、17,684 台计算机；
- authentication、process、DNS、network flow、red-team events。

LANL 是首要真实网络外部候选，但不能直接替代完整 `D_ext` 门禁：

- 它研究红队入侵/凭证滥用，不是 CERT 式恶意内部人员；
- 没有现成 47 维画像或用户图；
- 官网示例红队时间从 `t=151648` 秒附近开始，约第 1.75 天，不能预设存在 30 天全局干净前缀；
- 下载需要官方 fence token；
- 必须重新冻结用户过滤、事件簇、用户日和关系图协议。

推荐顺序：先用正式团队邮箱获取 `redteam.txt.gz`，审计时间；再取 auth/DNS；process/flows 在 smoke 后决定。

上述 cross-dataset benchmark 还报告：LANL authentication 做固定采样后，chronological split 测试期没有正例，只能使用 user-disjoint 评价。因此本项目的 LANL 时间切分不能预设成功。

### 当前结论

当前采用双外部路线：

```text
D_app = CLUE-LDS
D_net = LANL
```

CLUE-LDS 解决应用级半合成账号劫持的外部效应证据；LANL 计划提供真实企业网络红队事件证据。ChimeraLog 降为托管恢复后的备选。CLUE-LDS 已通过以下数据层门禁：

- 非 CERT 同生成体系；
- 至少 100 用户；
- 至少 20 事件；
- 至少 30 天干净训练期；
- 标签、映射、许可证和切分可审计。

但它只有 24 个有效 pair，且画像先验触发 `FLAT_PROFILE_PRIOR`。完整模型矩阵尚未完成，因此不能把它写成高功效外部确认。

因此计划状态从纯 `pending_data_gate` 升级为：

```text
CLUE_LDS_core_data_and_phase0_passed_full_external_matrix_pending
```

CLUE-LDS 和 LANL 仍需完成完整模型矩阵；r5.2 仍不是盲测。

## 5. 统计设计

主终点：

- PR-AUC；
- K=10 固定预算事件召回，冷却期 7 天。

10 个正式 seed：

```text
20261116 至 20261125
```

主估计对象：

> 条件于当前 10 个已训练模型，在用户/事件簇重采样下，方法 A 与 B 的配对效应。

主推断：

1. 每个 seed 做配对；
2. 10 个 seed 平均；
3. 固定每日 Top-K 名单；
4. 用户/事件簇配对 bootstrap；
5. 多用户事件整体重采样。

双层 bootstrap：

1. 先重采样 seed；
2. 再重采样用户/事件簇；
3. 只作训练随机性敏感性，不自动升级为确认性 p 值。

同时报告逐 seed 配对方向和同号 seed 数，例如 `8/10`。双层 bootstrap 不能证明“任取一次训练都稳定”。

事件不足：

- 事件数 `<20` 时不报告确认性 p 值；
- 只报告效应、区间、逐场景方向；
- r6.2、SPEDIA、TWOS、单个 Chimera 组织默认不进入确认性 p 值。

绝对非劣界：

```text
PR-AUC: -0.0005
K=10 recall: -max(0.02, 2 / N_events)
```

不再使用 `-5%` 相对值。

`-0.0005` 的量级依据：

| 现有 A1 附近口径 | PR-AUC | margin / A1 |
| --- | ---: | ---: |
| r4.2 instance-soft，原 mask | 0.008922 | 5.60% |
| r4.2 instance-soft，S2 欧氏库 | 0.014091 | 3.55% |
| r5.2 S2 欧氏库路径 | 0.007572 | 6.60% |

因此它约等于现有 A1 PR-AUC 的 3.55%–6.60%，不是随极小基线变化的百分比。

业务损失边界：只有至少一个主终点相对 A7/A8b 取得正向收益时，才接受最多 `0.0005` PR-AUC 损失；K=10 召回最多损失两个事件。没有主终点收益时不接受该 trade-off。

`D_ext` 通过数据门禁后，才用其自身事件结构做 80% 功效模拟，冻结最终 MDE 和 Holm 家族。

CLUE-LDS pilot 的独立判据已冻结：

```text
PR0 = A0-matched PR-AUC
positive_PR = Delta_PR >= max(0.001, 0.05 * PR0)
positive_recall = Delta_recall >= 3 / 48 = 0.0625
```

必须运行 2 个独立训练 seed；同一 endpoint 在两个 seed 上都为正且均值过阈值才达标。单 seed 只作实现 smoke，不能决定继续或停止，也不能用于调参。24 个有效 pair 的 MDE 约 23.0–29.5 pp，因此不显著时只能报告有限功效，不能写“方法无效”。

## 6. 主比较

候选比较：

1. A4 vs A1-Cov；
2. A4 vs A1+；
3. A4 vs A7；
4. A4 vs A8b；
5. A4 vs A9；
6. A3 vs A3-noAnchor；
7. A3 vs A3-noEMA。

A0-official vs A0-matched 是批次实现诊断，不属于方法优越性家族。

r5.2 预先冻结三个独立 Holm 家族：

- 主复验：A4 vs A1-Cov、A1+、A7、A8b、A9，共 10 个检验；
- 机制：A3 vs noAnchor、noEMA，共 4 个检验；
- 关系控制：A4 vs A5a、A5b，共 4 个检验。

三个家族分别校正。`D_ext` 的外部 Holm 家族只在其事件数和 MDE 冻结后确定。

机制额外要求：

- A4 相对 A5a/A5b 至少一个主终点有统计支持；
- NCA@10 的 CI 下界 `>=-0.02`；
- ER_ratio 的 CI 下界 `>=0.90`；
- A3 vs noAnchor/noEMA 至少各有一个主终点通过预注册 Holm；
- 仅符号为正不构成机制通过。

## 7. 算力

已测：

- r4.2，1,000 用户，每 epoch 62 批，B=620，单次 149–194 s；
- r5.2，2,000 用户，每 epoch 125 批，B=1,250，单次 168–411 s；
- r6.2，4,000 用户，每 epoch 250 批，B=2,500，单次 145–261 s。

正式变体单 seed 合计约 `40B`。10-seed r4.2+r5.2 的主矩阵在覆盖采样前约 51 GPU·h；按 1.5–2 倍覆盖采样和调参、基线、安全实验，保守总预算约 160–400 GPU·h。

单张 RTX 5060 八周理论算力上限约 1,344 GPU·h。主要风险不是纯训练，而是数据获取、覆盖采样、深度基线适配和统计审计。

## 8. 当前 Gate

### 方法 Gate

1. 图边覆盖门禁通过；
2. A1-Cov vs A0-matched 至少一个主终点达到最小正向效应；
3. r4.2 pilot 只决定继续或停止，不用于调参；
4. A4 在 r5.2 锁定复验中相对 A1+ 和 A9 在两个主终点均优效；
5. A4 相对 A7/A8b 至少一个主终点优效，两个终点均非劣；
6. A4 相对 A1-Cov 有端到端正向收益；
7. A4 相对 A5a/A5b 至少一个主终点有统计支持；
8. A3 对 noAnchor/noEMA 机制家族通过；
9. NCA@10 和 ER_ratio 非劣；
10. 异常用户进入正常核心率不持续恶化。

### 外部 B Gate

额外要求：

- `D_ext` 通过数据门禁；
- `D_ext` 的预注册比较完成；
- r5.2 与 D_ext 不出现相反方向的致命失败；
- 论文明确区分 r4.2 开发、r5.2 已见复验和 D_ext 外部确认。
- `FLAT_PROFILE_PRIOR` 触发时，只报告端到端结果；A3/A5b 的画像机制差异记为“不可检验”，不记为 A3 失败。

### 停止条件

- 图边覆盖率不足或用户零曝光；
- A0-matched 无法解释训练分布变化；
- A1-Cov 相对 A0-matched 的两个 endpoint 均未达到 pilot 最小效应；
- CLUE-LDS 的两个 seed 在两个 endpoint 上都未达到独立阈值时，停止 CLUE 上的 AdaGraph 主实验；
- A4 不能优于 A1+ 或 A9；
- A4 相对 A7/A8b 超过非劣界；
- A3 与 A5b 无差异且 PeerMass 无信号；
- `epsilon=0` 与 A1-sparse-EMA 仍系统性不同；
- 有效秩持续下降；
- 异常用户进入正常核心率持续恶化；
- D_ext 无法在预算内通过数据门禁；
- 深度基线无法公平运行。

第三周末若仍无通过门禁的 `D_ext`，不再维持 B 类外部承诺，转为 C 类上沿或“批次共现与覆盖修正”的方法学/实证定位。

## 9. 需要审阅 AI 重点回答

1. v1.2.2 的 50% 随机填充和 A3/A5a/A5b 联合覆盖，是否足以降低批次表对 A3 的偏向；
2. A1-R0、A1-Cov、A0-matched、A5a、A5b、A3-hard 的分解顺序是否能把“关系信息”“软权重”“批次共现”分开；
3. 把 r5.2 定位为已见锁定复验、把真正 B 证据交给 D_ext，统计上是否诚实且可执行；
4. 事件数 99 的 r5.2、绝对非劣界 `PR=-0.0005` 和 `K10=-max(0.02,2/N)` 是否合理；
5. 条件于 10 个模型的用户/事件簇 bootstrap 加双层敏感性，是否足以回应训练方差；
6. 只有 12 个候选、前 4 名各补 2 seed、最多 20 次调参，是否仍存在明显的赢者诅咒或不公平；
7. `A1-sparse-EMA / A3-epsInf / A3-noAnchor / A1-repSupport-P` 的支撑集×权重 2×2 是否完整；
8. CLUE-LDS 的应用级半合成账号劫持与 LANL 的真实红队事件，能否构成足够互补的外部证据；若不能，还有哪些合法贡献路径；
9. 是否有未列出的单点会让方法整体不成立；
10. 在不改变“TS2Vec 软关系信号创新”主线的前提下，最小但最有价值的理论补充是什么。

## 10. 不在本包中声称的事项

- 不声称 AdaGraph 已有效；
- 不声称 r5.2 是盲测；
- 不声称 r1 是确认集；
- 不声称 SPEDIA/TWOS 已满足外部确认；
- 不声称 ChimeraLog 已可下载；
- 不声称 TGMC、Ripple2Detect 或 ATHITD 的全部细节已核实；
- 不声称共同覆盖批次表已经实现或测试通过。
