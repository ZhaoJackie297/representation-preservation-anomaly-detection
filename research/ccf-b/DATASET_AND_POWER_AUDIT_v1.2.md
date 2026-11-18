# 数据与功效审计 v1.2.7

审计日期：2026-11-18。

本文件回答三个问题：

1. 现有哪些数据能承担开发、锁定复验和独立确认；
2. 哪些数据只是一项辅助压力测试，不能提升证据等级；
3. 在确认集冻结前，哪些统计阈值必须保持未冻结。

## 1. 结论摘要

| 数据 | 当前可审计规模 | 标签与切分 | v1.2.7 地位 |
| --- | --- | --- | --- |
| CERT r4.2 | 1,000 用户，159 天训练，144 天验证，198 天测试 | 有事件标签；已用于 R0 开发 | 开发、pilot、无标签选参、方向性诊断 |
| CERT r5.2 | 2,000 用户，104 天干净训练期，413 天外部测试，99 个事件 | 有标签；R0 已见 | v1.2.2 冻结后的锁定参数复验；不是盲测 |
| CERT r6.2 | 4,000 用户，222 天训练，294 天测试，5 个事件 | 有标签 | 五事件压力审计；不进入显著性 |
| CERT r1 | 约 1,000 用户；logon 849,579、device 65,668、http 3,451,665 行；LDAP 18 个月 | 官方 answer key 无恶意场景标签 | 只能作无标签工程诊断；删除确认集地位 |
| SPEDIA | 72,250 行、25 列、17 个 `User` 值、25 个活动日期、41,541 个异常行 | 事件级 `Anomaly`；首日已有异常；无独立干净训练期 | 会话级辅助压力测试；不作主图确认集 |
| TWOS | 五天受控红蓝对抗 | 有攻击活动，但公开数据与许可证需继续审计 | 外部定性/小样本压力测试 |
| CLUE-LDS | 50,522,931 事件、5,389 用户、1,910 数据日、49 类事件 | 真实正常行为；本项目固定 seed 构造 24 pairs、48 个半合成账号劫持事件 | 数据、标签、图和 10-seed 覆盖已通过；有限功效外部效应复验，完整模型矩阵待跑 |
| LANL | 58 天、12 GB、1.65B 事件、12,425 用户；CC0 | red-team compromise events；无现成用户日/图切分 | 首要真实网络外部候选；构念和时间切分待审计 |
| ChimeraLog | 三组织，每组织 20 人，一个月，15 类攻击；数据量约 25B 条 | 细粒度自动标签；采用用户日切分 | 托管恢复后的备选 |
| OpTC | 规模可能足够 | 需先证明用户日、事件标签、训练期和权限 | 待审计候选 |

截至本日，CLUE-LDS 已通过数据、标签、用户日、图协议和 Phase 0 覆盖门禁，但还没有完成完整模型矩阵，也没有达到高功效确认试验的条件。因此：

- `D_ext` 状态为 `CLUE_LDS_core_data_and_phase0_passed_full_external_matrix_pending`；
- CLUE-LDS 只能作为“有限功效外部效应复验”，24 个有效 pair 的 80% 功效 MDE 约为 23.0–29.5 pp；
- CLUE-LDS、r4.2 和 r5.2 的画像先验均触发 `FLAT_PROFILE_PRIOR`，画像锚机制状态为 `METHOD_NOT_SUPPORTED`；
- 当前不能宣称 B 类外部有效性已经成立；
- r5.2 可以提供事件数足够的锁定复验，但必须保留“已见、非盲测”标记。

## 2. CERT r1：必须撤销确认集地位

本地审计文件：

```text
E:\MarketRisker\work\cert\r1_audit.json
```

关键字段：

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

已核对的数据规模：

- logon：849,579 行、1,000 个用户；
- device：65,668 行、228 个用户；
- http：3,451,665 行、1,000 个用户；
- LDAP：18 个月、最新月 935 个用户；
- 时间范围：2010-01-04 至 2011-05-14。

结论：

- r1 不能计算 PR-AUC、K=10 事件召回或确认性 p 值；
- r1 不进入 Holm 家族；
- r1 最多用于 Schema、账户映射、缺失处理和推理性能诊断；
- v1.1 中所有“r1 独立确认集”表述均作废。

## 3. SPEDIA：公开但不是本项目的主确认集

已核实资源：

- 数据 DOI：`10.5281/zenodo.15525713`；
- 数据记录：<https://zenodo.org/records/15525713>；
- 许可证：CC BY 4.0；
- 文件：`logs_SPEDIA_annotated_en.csv`；
- 文件大小：45,703,322 bytes；
- 官方 MD5：`e01e0baaf3b523a0b493fc9c52af2b1c`；
- 本地下载文件 MD5 与官方值一致。

本地逐列审计结果：

| 项目 | 数值 |
| --- | ---: |
| 行数 | 72,250 |
| 列数 | 25 |
| `Anomaly=1` | 41,541 |
| `Anomaly=0` | 30,709 |
| 唯一 `User` 值 | 17 |
| `User` 缺失行 | 1,372 |
| 活动日期数 | 25 |
| 时间范围 | 2025-03-06 至 2025-03-30 |
| 异常用户字符串数 | 16 |
| 正常用户字符串数 | 7 |
| 同时出现正常/异常的用户字符串数 | 6 |
| `User` 非空且按用户日去重后的异常用户日 | 167 |
| `User` 非空且按用户日去重的正常用户日 | 173 |

异常在第一日活动即出现：

```text
2025-03-06: 1,477 anomalous rows
```

这使“训练期完全早于首个异常事件”的现有 CERT 协议无法直接迁移。

官方描述还明确说明：

- 30 天演练；
- 真实参与者实施攻击；
- 正常行为一部分由角色模型模拟；
- 一部分事件由 CERT 数据派生；
- 恶意与正常事件为平衡分布，面向监督/分类研究。

结论：

- 17 个用户不足以稳定构造 top-m=10 的用户关系图；
- 25 天且首日有攻击，不满足至少 30 天干净训练期；
- 平衡事件分布不是本项目的低流行率固定预算场景；
- SPEDIA 可做“会话/事件级迁移压力测试”或外部描述性对照；
- 不得用它替代 `D_ext` 来宣称 AdaGraph 图级外部确认。

独立的 2026 IJACSA cross-dataset benchmark 对 SPEDIA 做了更严格的 real-only 过滤：删除 `Decoder_name == 'cert'` 行后剩 20,619 条事件、82/256 个正用户日和约 0.32 基率。其 PR-AUC 数值不能和低流行率 CERT 直接混表，但可作为跨数据集基线和方法学参照。

## 4. TWOS：可核实但规模明显不足

已核实论文：

- Harilal 等，`TWOS: A Dataset of Malicious Insider Threat Behavior Based on a Gamified Competition`；
- MIST@CCS 2017；
- DOI：`10.1145/3139923.3139929`。

数据集元记录：

- Homoliak，`TWOS (The Wolf of SUTD) - Dataset of Malicious Insider Threat`；
- DOI：`10.13140/RG.2.2.28309.42729`；
- 类型：Dataset；
- 正式许可证与稳定下载入口尚未核实。

后续独立论文中的可核对描述：

- Chimera 论文称 TWOS 来自受控五天红蓝对抗，并明确将其列为小规模数据集；
- SENTINEL 论文称 TWOS 收集于 2017 年 3 月 SUTD 组织的竞赛；
- 多篇综述指出 TWOS 样本量有限，因此 CERT 仍是主要大规模基准。

结论：

- TWOS 不能提供 1000 级用户图和至少 20 个独立可评价事件；
- 可作定性真实感和规避案例的补充；
- 不进入确认性 Holm 家族。

## 5. ChimeraLog：优先审计，但尚未冻结

已核实论文与资源：

- 论文：`Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation`；
- NDSS 2026 接收；
- arXiv：<https://arxiv.org/abs/2508.07745>；
- 官方项目页：<https://sites.google.com/view/chimerallm>；
- 官方代码页：<https://github.com/fish98/Chimera>。

论文可核实特征：

- 技术公司、金融机构、医疗机构三种组织场景；
- 每个组织 20 个员工代理；
- 一个月的模拟期；
- 15 类内部威胁；
- 六类日志：login、email、web、file、network traffic、system log；
- 自动细粒度标签，并给出用户日特征聚合基准；
- 数据规模约 25B 条，属于大规模生成日志。

但官方页面在本次审计时仍写明：

> We are actively organizing the data and code and will be releasing them as open source shortly.

页面同时显示 `ChimeraLog` 正在从服务器迁移。

进一步核对官方 GitHub `README.md` 后，原因已经从“网页暂未更新”变为可确认的上游状态：

> access to my university OneDrive account has been affected, and the previous ChimeraLog download links are no longer accessible. I am currently looking for a new hosting solution.

因此当前准确状态是：

- Chimera 代码公开；
- 论文和模拟框架可复现到“重新生成数据”的层面；
- 原始 ChimeraLog 下载链接已失联；
- 作者正在寻找新托管；
- 重新生成约 25B 条日志需要 LLM API、Docker、sysdig、tcpdump 和大量 token，不能在八周内假定可完成；
- 不能把“代码可下载”写成“确认数据已到位”。

因此当前不能确认：

- 实际下载地址和许可证；
- 用户日表的字段和标签粒度；
- 每个组织可用于 K=10 的独立事件数；
- 20 个用户/组织是否足够训练 AdaGraph；
- 三个组织能否合并为外部确认分析，还是只能做分层压力测试。

结论：

- CLUE-LDS 已取代 ChimeraLog 成为首要应用级候选，LANL 是首要真实网络候选；
- 不能仅凭论文声明把它写成“已具备确认集”；
- 当前阻塞不是下载速度，而是原始数据托管链接失效；
- 若下载和标签门禁通过，可按组织分层报告，并用固定效应或分层 bootstrap 合并；
- 单组织只有 20 个用户和至多 15 类攻击，默认不满足确认性事件数要求。

## 5.1 CLUE-LDS：有限功效应用级外部候选

官方资源：

- 数据：<https://zenodo.org/records/7119953>；
- DOI：`10.5281/zenodo.7119953`；
- 论文 DOI：`10.1109/BigData55660.2022.10020672`；
- 代码：<https://github.com/ait-aecid/clue-lds>；
- 数据许可：CC BY 4.0；
- 代码许可：GPL-3.0；
- 文件：`clue.zip`；
- 压缩大小：635,105,552 bytes；
- 官方 MD5：`9e318370f96b68077667e9cdc05f26a5`；
- 解压规模：约 14.5 GB。

已核实数据规模：

- Huemer Group 真实 hBOX/Nextcloud 云存储；
- 2017-07-07 至 2022-09-29；
- 1,910 个数据日期；
- 50,522,931 条事件；
- 5,389 个匿名用户；
- 83,147 个用户日；
- 49 类登录、文件、分享、组、配置等事件；
- 部分事件带角色、IP、地理位置。

### 标签机制

原始 CLUE-LDS 只有正常行为。项目基于官方切换算法，由本项目固定 seed 构造半合成账号劫持：

- 标签 seed=`20261101`；
- 训练 seed=`20261116..20261125`；
- 24 pairs，产生 48 个用户级事件；
- 相似度范围 0.1–0.6；
- 切换前至少 35 个活跃日；
- 切换后至少 30 个活跃日；
- 只允许切换日期落入 2022-03-01 至 2022-08-30 的评价窗口；
- 训练截止 2022-02-28，评价期至 2022-09-29。

“官方”只修饰切换算法来源，不修饰标签本身。论文必须写作“基于官方切换算法实现、由本项目固定 seed 构造的半合成账号劫持”，不得写成官方异常标签、官方受控注入或真实攻击标签。

### 用户筛选规则

5,389 到 838 的筛选由以下规则冻结：

1. 只保留 `uidType=name`；
2. 所有 48 个注入目标用户保留；
3. 其他用户至少在训练期出现一次；
4. 非目标用户按训练期事件数降序；
5. 最多保留 1,000 用户；
6. 实际有 838 个 `name` 用户满足训练期活动条件；
7. 任一目标用户训练期全空则整体失败，不进行静默回填。

该规则不使用异常标签、评价期检测结果或 A4 输出。

### 标签距离独立性

pair 选择使用：

```text
0.7 * action Jaccard
+ 0.3 * average-events-per-active-day ratio
```

画像先验使用 52 维用户日特征的 median/q90、活跃比例、标准化欧氏距离和高斯核。两者不是同一个距离。

正式审计：

```text
unique_pairs = 24
Spearman(label pair similarity, profile affinity) = -0.1991304347826087
status = PASS_DISTANCES_DISTINCT
```

审计脚本为 `clue_lds/audit_label_prior_independence.py`，结果见
`clue_lds/label_prior_independence_20261118.json`。

### 先验信息量

画像先验本身接近平坦：

- 画像维度 105；
- 全图行熵中位数 6.4906；
- 均匀上界 `log(837)=6.7298`；
- top-10 支撑集 `P_m` 的 `0.95*log(10)=2.1875`；
- `mean H_m=2.1689`；
- `median H_m=2.3026`；
- 89.98% 用户高于阈值。

因此按预注册规则触发 `FLAT_PROFILE_PRIOR`。触发后：

- A3/A5b 的画像机制贡献记为“不可检验”；
- 不记为 A3 失败；
- 端到端比较仍可运行；
- 必须报告熵、beta 和实际 KL。

组/分享辅助图只包括训练期关系：group edges 1,488、share edges 1,121、非零边 2,140、孤立用户 565/838。它只作辅助参照，不能替代行为画像先验。

### 真实图覆盖

A3/A5a/A5b 使用同一共同覆盖批次表：

- batch size 16；
- 8 个设计槽位 + 8 个随机填充槽位；
- 10 个正式 seed；
- 每张图覆盖 `>=80%`；
- 零曝光用户为 0；
- 10/10 batch table hash 唯一。

最终结果：

- A3 覆盖 85.95%–88.25%；
- A5a 覆盖 80.01%–80.08%；
- A5b 覆盖 85.49%–86.88%；
- 覆盖 batch 948–961；
- 相邻 seed 用户对 Jaccard 0.2184–0.2216。

A5a 总是最后达到 80%，说明随机图是当前覆盖瓶颈。

### 功效

依据注入 pair 的依赖结构，主 bootstrap 按 pair 整簇重采样，不能按 48 个事件独立重采样。24 个 pair 在 80% 功效、单侧 alpha=0.025 下的 MDE 约为 23.0–29.5 个百分点的事件召回差。

因此 CLUE-LDS：

- 对 5–10 pp 的中小效应功效不足；
- 不显著不能直接解释为方法失败；
- 不承担高功效外部确认；
- 只作有限功效外部效应复验，并与效应量、pair-cluster 区间和 MDE 一起报告。

### 数据产物

最终标签文件：

```text
hijack_labels_24pairs_20261101.csv
SHA-256 = 2441744DED393381C9B42D3E1EFB53B11137810D8F94611FCF76BC0C67D248CA
```

最终处理数据和覆盖产物的逐文件 SHA-256 见：

```text
research/ccf-b/clue_lds/artifact_hashes_20261118.json
```

`clue.json` SHA-256 仍为：

```text
2CFE6C172104954DB574DD7648186E7A148FA1D553BD9F80212AE3F8C28956C5
```

### 判定

- CLUE-LDS 可以取代 ChimeraLog，作为首要应用级外部候选；
- 它已在数据、标签、图和 Phase 0 覆盖层通过门禁；
- 正式 A0-matched/A1-R0/A1-Cov 矩阵、A0-A3batch/A1-Cov-A5a/A1-Cov-uniform 控制、训练期内部合成异常验证和 pilot 尚未完成；
- 24 个有效 pair 使其只能承担有限功效外部效应复验，不能单独支撑 B 类外部有效性。

## 5.2 LANL：当前首要真实网络外部候选

官方资源：

- 页面：<https://csr.lanl.gov/data/cyber1/>；
- DOI：`10.17021/1179829`；
- 报告号：LA-UR-15-23810；
- 许可：CC0；
- 引用：A. D. Kent, *Comprehensive, Multi-Source Cyber-Security Events*, LANL, 2015。

已核实规模：

- 58 个连续日；
- 约 12 GB 压缩；
- 1,648,275,307 条事件；
- 12,425 用户；
- 17,684 台计算机；
- 62,974 个进程；
- authentication、process、DNS、network flow、red-team 五类数据。

文件：

| 文件 | 压缩大小 | 内容 |
| --- | ---: | --- |
| `auth.txt.gz` | 7.2G | Windows authentication |
| `proc.txt.gz` | 2.2G | process start/end |
| `flows.txt.gz` | 1.1G | network flows |
| `dns.txt.gz` | 177M | DNS lookups |
| `redteam.txt.gz` | 4.8K | known red-team compromise events |

### 优势

- 真实企业网络，不是 CERT 或 Chimera 生成器；
- 规模、用户数和事件量足够；
- 多源日志可构造用户日行为；
- CC0 许可；
- red-team 标签来自认证事件，可以映射到 user/computer/time。

### 不能直接替代的地方

1. **构念不同**
   - LANL 是红队入侵和凭证滥用；
   - CERT 是恶意内部人员场景；
   - 只能称“真实企业网络红队事件上的外部验证”，不能称同一内部威胁构念的独立复现。

2. **没有现成训练期行为画像**
   - 必须从 authentication、process、flow、DNS 新构造；
   - 用户到计算机是多对多关系；
   - 需要排除 `$` 机器账号、SYSTEM、Local Service 和纯服务账号。

3. **干净训练前缀未知**
   - 官网示例 red-team 时间从 `t=151648` 秒附近开始，约第 1.75 天；
   - 示例不保证是首行，但不能预设存在 30 天全局干净前缀；
   - 必须取得 `redteam.txt.gz` 后计算首末时间和每日分布。

4. **事件不是现成 incident ID**
   - `redteam.txt.gz` 每行是 compromise event；
   - 必须冻结事件簇规则，不能把每行都当成独立 incident；
   - 事后不得为了让事件数超过 20 而改聚类阈值。

5. **用户图不是现成组织结构**
   - 可用 authentication co-access、shared-computer、DNS/flow interaction 构图；
   - 这些图与 CERT 的画像图不同，必须作为单独的外部适配协议。

6. **下载有 fence**
   - 官方页面要求填写邮箱和使用目的后取得 token；
   - 当前没有替用户提交联系信息，也没有下载完整 12 GB。

7. **时间切分已被独立工作证明困难**
   - 2026 cross-dataset benchmark 对 LANL authentication 做固定 0.02 均匀采样后，chronological split 的测试期没有正例；
   - 他们最终只报告 user-disjoint 分布内评价；
   - 这不能替代本项目自己的标签审计，但说明“全局时间外测试”不能预设可行。

### 推荐审计顺序

1. 用团队正式邮箱和真实研究用途获取 fence token；
2. 首取 `redteam.txt.gz`，计算首事件、末事件、每日事件、用户数和事件簇；
3. 没有任何 30 天全局干净前缀时，尝试预先冻结的“每用户事件前局部基线”；
4. 时间切分可行后，再取 `auth.txt.gz` 和 `dns.txt.gz`；
5. 先跑小窗口 smoke，再决定是否取 process 和 flows；
6. 冻结用户过滤、特征、图、事件簇和切分后，才升级为 `D_ext`。

### 当前判定

- LANL 可以取代 SPEDIA/TWOS 成为首要外部候选；
- LANL 不能自动取代 `D_ext` 门禁；
- 若没有有效干净时间或局部基线，只作外部压力测试；
- 若没有至少 20 个可评价事件簇，不报告确认性 p 值。

## 5.3 其他候选为什么不是当前首选

| 候选 | 可核实特点 | 对当前项目的问题 |
| --- | --- | --- |
| WUIL | 真实 Windows GUI 使用轨迹，面向 masquerade detection | 公开稳定访问和许可证不明确；Chimera 对比表称其缺少有标签威胁行为；难以形成企业用户关系 |
| CPTC 2018 | 高校渗透测试竞赛，红蓝队真实操作，部分应用/网络日志 | 数据噪声大、碎片化、良性行为连续性弱；不是稳定用户日企业运营日志 |
| OpTC | DARPA 透明计算，约两周、1000 hosts 规模测试，实际收集 500 hosts，约 1 TB 压缩数据，含红队 ground truth | 规模过大；只有两周且偏 host/network provenance；缺少 CERT 式应用语义和稳定用户画像 |
| OrgForge-IT | 2026 新合成基准，51 天、2,904 条 telemetry、4 个检测场景、MIT | 数据量和攻击场景数太小；尚需独立复现与访问核验；不能支撑 20+ 事件稳定确认 |
| BETH | 主机/kernel-level 序列，训练为 benign/suspicious insider，测试含 botnet OOD | 不是企业应用日志；异常构念是 botnet/OOD，不能验证用户图 |
| Schonlau/RUU 类命令序列 | masquerade 检测经典数据 | 规模小、年代久、缺少现代企业多模态行为 |

因此当前排序为：

```text
CLUE-LDS > LANL > ChimeraLog(若恢复) > SPEDIA/TWOS
```

OpTC、CPTC、WUIL、BETH 和 OrgForge-IT 只有在目标贡献改为网络级/provenance 级或重新定义节点后才值得纳入。

## 6. 锁定复验与功效

CERT r5.2 的训练和评价结构：

- 2,000 个用户；
- 104 天训练期；
- 413 天外部评价期；
- 99 个标注事件；
- 1,307 个恶意账户日。

事件召回的最小离散步长为：

```text
1 / 99 = 0.010101...
```

因此：

- 一个事件约等于 1.01 个百分点；
- 两个事件约等于 2.02 个百分点；
- 使用 `-5%` 相对非劣界在 99 个事件上不足一个稳定的事件步长，已废除；
- v1.2.2 的召回非劣界为 `-max(0.02, 2/N_events)`。

r5.2 不能单独提供的证据：

- 它不是盲测；
- 它不能排除项目团队围绕 R0 已形成的选择偏差；
- 它不能证明跨生成体系或外部组织泛化。

因此 r5.2 的定位是：

> 预注册锁定参数后的一次性版本复验，不是独立外部确认。

### 非劣界的版本变化

v1.2.7 的默认临时界为：

```text
PR: -min(0.0005, 0.05*PR_A1plus)
K10: -max(0.01, 1/N_events)
```

旧 `delta_margin_PR=-0.0005` 只保留为历史量级审计：

| 来源文件 | 方法/口径 | 测试 PR-AUC 均值 |
| --- | --- | ---: |
| `e2_instance/missing_safe_v2_20261117/metrics_by_seed.csv` | `T3_instance_soft_global_z`，原 mask | 0.008922 |
| 同上 | `T3_instance_soft_euclidean`，S2 欧氏库 | 0.014091 |
| `canonical/canonical_metrics_by_seed.csv` | r5.2 `T4_bank_euclidean` | 0.007572 |

计算：

```text
0.0005 / 0.008922 = 5.60%
0.0005 / 0.014091 = 3.55%
0.0005 / 0.007572 = 6.60%
```

因此旧绝对容差约占 A1 的 `3.55%–6.60%`。v1.2.7 改为“绝对上限与 5% A1+ 基线中更严格者”，并只默认允许一个事件步长。

业务上只在一个主终点取得正向收益时接受临时界内损失；没有主终点收益时，任何超过临时界的损失都不接受。

## 7. MDE 与最终冻结顺序

`D_ext` 选定后的冻结顺序：

1. 固定文件哈希、用户映射、事件映射、训练期和评价期；
2. 只运行 A1-Cov/A1-R0 等基线，保存事件簇、用户簇和原始输出；
3. 在 A4/AdaGraph 结果可见前，定义业务最小有意义效应；
4. 以该数据的用户/事件结构做 80% 功效模拟；
5. 冻结最终比较家族、MDE 和非劣界；
6. 一次性运行方法矩阵；
7. 任何后续偏离写入新勘误，并单列探索性结果。

禁止：

- 用 r4.2 或 r5.2 方差直接代替 `D_ext`；
- 看到 A4 结果后下调 MDE 或扩大非劣界；
- 把多个小数据集的显著结果事后挑选合并；
- 把事件数 `<20` 的 p 值写成强证据。

## 8. 当前数据门禁判定

| 条件 | r1 | SPEDIA | TWOS | CLUE-LDS | ChimeraLog | LANL | r5.2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 有事件标签 | 否 | 是 | 待审计 | 本项目半合成账号劫持，48 事件 | 论文声明是 | red-team compromise | 是 |
| 独立生成体系 | 否，CERT | 部分混合 CERT | 是 | 是，真实正常行为 | 是 | 是，真实 LANL 网络 | 否，CERT |
| 至少 100 用户 | 约 1,000 | 否，17 | 待审计 | 是，5,389 | 否/组织 | 是，12,425 | 是，2,000 |
| 至少 20 事件 | 否 | 用户日多、事件语义不足 | 待审计 | 是，48 事件/24 pair 簇 | 待审计 | 原始事件多，incident cluster 待审计 | 是，99 |
| 至少 30 天干净训练期 | 无标签不可定义 | 否 | 否，共五天 | 是，训练截止 2022-02-28 | 待审计 | 未知，可能没有全局前缀 | 是，104 天 |
| 当前可下载 | 是 | 是 | 待审计 | 是，635 MB | 否 | 需 fence token | 是 |
| 主确认资格 | 否 | 否 | 否 | 有限功效外部效应复验；完整矩阵待跑 | 待定，备选 | 待定，当前真实网络首选 | 否，已见 |

## 9. 当前外部数据门禁状态

当前阻塞不是单一的“下载慢”，而是数据语义、统计功效和完整模型矩阵三类问题：

1. **有数据但没有标签**
   - CERT r1 可下载、规模也够，但官方 answer key 不含恶意场景标签；
   - 无标签数据不能计算 PR-AUC、K=10 或事件召回。

2. **有标签但协议不适配**
   - SPEDIA 可下载并已完成文件级审计；
   - 但 17 个 `User` 值、25 天、首日异常，无法承担用户图确认和干净训练期；
   - TWOS 只有五天受控竞赛，规模不足。

3. **最适配的新数据上游失联**
   - ChimeraLog 论文和代码公开；
   - 原始数据曾经通过学校 OneDrive 分享；
   - 作者毕业后账号受影响，旧链接失效，目前正在寻找新托管；
   - 自行重生成约 25B 条日志需要完整多智能体环境、外部 LLM API 和大量 token，不能作为八周计划中的稳定依赖。

4. **CLUE-LDS 已通过应用级数据门禁，但功效有限**
   - 已完成下载、哈希、全量审计、固定 seed 注入、用户日、画像图、组/分享辅助图和 10-seed 覆盖；
   - 24 pairs/48 事件，主统计按 pair 聚类；
   - 24 个有效簇在 80% 功效下的召回 MDE 约为 23.0–29.5 pp；
   - `P_m` 已触发 `FLAT_PROFILE_PRIOR`，A3/A5b 的画像机制结论不可检验；
   - 因此它不是高功效确认试验，只能作有限功效外部效应复验。完整模型矩阵尚未开始。

5. **更好的真实网络候选尚未完成适配**
   - LANL 已核实为 58 天真实企业网络数据，CC0，规模和标签都存在；
   - 但下载需要 fence token；
   - 红队事件时间可能很早，未必有 30 天全局干净前缀；
   - 还需要事件簇、用户过滤、跨模态特征和图适配；
   - 因此它是真实网络首选，但还不能写成已经通过 `D_ext`。

结论：

- `D_ext` 采用 `D_app=CLUE-LDS` 与 `D_net=LANL` 双候选；CLUE-LDS 已通过核心数据与 Phase 0 门禁，LANL 尚未完成适配；
- A0/A1 入口 smoke 已通过；当前可以补齐分解控制、训练期内部合成异常验证和 CLUE-LDS 两 seed pilot；
- CERT 和 CLUE-LDS 的画像先验都是平的，因此端到端结果不能自动解释成画像锚机制成立；
- 不能开始“B 类外部确认已完成”的论文表述；
- 若完整模型结果和 LANL 证据不足，应把 CLUE-LDS 明确写成有限功效外部复验，并保留方法学/实证论文的降级路径。
