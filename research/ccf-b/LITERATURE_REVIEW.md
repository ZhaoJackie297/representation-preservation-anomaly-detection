# AdaGraph-SoftCLT 文献审查与相关工作草案

核查日期：2026-11-18。

状态说明：

- 标为“已核实”的条目已通过 arXiv 元数据、OpenAlex/DOI 或论文官方页面核对标题、作者、年份和发表场所。
- 未取得完整正文或页码的信息，不用于细化方法描述。
- 论文引用以 `references_ccf_b.bib` 中的条目为准。

## 一、直接基础

### TS2Vec

TS2Vec 在增强后的上下文视图上进行层次化实例与时间对比，并学习时间戳级表示。它是本项目的主干编码器。已核实：Yue 等，AAAI 2022，arXiv:2106.10466。

### SoftCLT

SoftCLT 将硬 0/1 对比关系替换为 0 到 1 的软权重。实例项使用数据空间距离，时间项使用时间戳差；在异常检测配置中，原论文默认关闭 instance-soft，主要验证 temporal-soft。已核实：Lee、Park、Lee，ICLR 2024 Spotlight，arXiv:2312.16424。

与本项目的关系：AdaGraph 保留 SoftCLT 的软对比形式，但不固定实例距离。画像距离作为先验，表示反馈只能在信任域内更新关系图。

## 二、近邻、原型与聚类对比

### NNCLR

NNCLR 在表示空间的支持集中检索最近邻，并将最近邻作为额外正样本。已核实：Dwibedi 等，ICCV 2021，DOI:10.1109/ICCV48922.2021.00945。

区别：NNCLR 的正样本完全由当前表示决定；AdaGraph 的边由画像先验约束，EMA 表示只提供有限反馈。NNCLR 式近邻是本项目必须实现的基线。

### PCL

PCL 在 EM 框架下迭代聚类原型和对比表示，通过 ProtoNCE 将样本拉向原型。作者 Li、Zhou、Xiong、Hoi；OpenReview ID `KmykpuSrjcq`，ICLR 2021。

区别：PCL 学习簇原型，AdaGraph 不替换用户原型，而是校正用户对之间的正关系权重。

### SwAV

SwAV 通过交换视图的聚类分配进行自监督学习，并使用多裁剪增强。已核实：Caron 等，NeurIPS 2020，arXiv:2006.09882。

区别：SwAV 的聚类分配是训练目标核心；AdaGraph 的图只进入实例正关系权重，保留 TS2Vec 的层次对比结构。

### MoCo

MoCo 使用队列和动量编码器构造大而一致的负样本字典。已核实：He 等，CVPR 2020，arXiv:1911.05722。

区别：MoCo 关注负样本字典规模与一致性；AdaGraph 关注用户行为语义相似造成的假负样本，不以扩大字典为目标。

## 三、假负样本与去偏

### Incremental False Negative Detection

该方法在训练过程中逐步检测并显式移除以假负样本。已核实：Chen 等，ICLR 2022，arXiv:2106.03719。

区别：该方法主要依赖当前表示空间动态检测；AdaGraph 额外使用稳定画像先验，并限制图漂移。

### False Negative Cancellation

Huynh 等在 WACV 2022 提出 False Negative Cancellation，对潜在同类负样本进行识别和抵消；DOI:10.1109/WACV51458.2022.00106。

注意：文献中的 `False Negative Cancellation` 与 Chen 等的 `Incremental False Negative Detection` 是两项不同工作，引用时不能混写。

### Debiased Contrastive Learning

Chuang 等通过估计同类负样本比例修正 InfoNCE 偏差，并给出下游泛化界。已核实：NeurIPS 2020，arXiv:2007.00224。

区别：Debiased CL 通常估计全局或采样层面的负样本污染；AdaGraph 使用用户级画像先验和表示反馈构造局部软关系。

## 四、表示塌缩与无标签质量控制

### Dimensional Collapse

Jing 等指出，对比学习也可能发生维度塌缩，即表示仅占据低维子空间。已核实：ICLR 2022，arXiv:2110.09348。

AdaGraph 必须逐轮记录有效秩、各向同性和异常用户进入正常核心的比例，不能只观察训练损失。

### RankMe

RankMe 使用表示矩阵的有效秩作为无标签质量指标，并可辅助超参数选择。已核实：Garrido 等，ICML 2023，arXiv:2210.02885。

## 五、内部威胁检测

### CERT 与早期无监督方法

CERT 数据集由 Glasser 和 Lindauer 在 2013 年 IEEE Security and Privacy Workshops 的论文中介绍。Tuor 等在 2017 年 AAAI AICS workshop 使用结构化日志做无监督深度检测。Yuan 与 Wu 在 Computers & Security 2021 综述了内部威胁检测的挑战与机会。

### LAN

LAN 面向活动级实时内部威胁检测，同时学习活动序列内的时间依赖和跨序列关系，并处理审查预算和类别不平衡。已核实：Cai 等，IEEE TIFS 2024，DOI:10.1109/TIFS.2024.3488527。

区别：LAN 构造活动级图并做实时告警；AdaGraph 是日级自监督表示训练中的软正关系校正。

### Ripple2Detect

Ripple2Detect 关注基于语义相似学习的多步内部威胁证据检测。已核实：Liu 等，Computers & Security 2025，DOI:10.1016/j.cose.2025.104387。由于本次未取得摘要，不在正文中扩展其具体算法细节。

### ATHITD

ATHITD 是基于注意力时序异构图神经网络的内部威胁检测方法。已核实：Qi 等，Computers & Security 2025，DOI:10.1016/j.cose.2025.104587。由于本次未取得摘要，只作为近期图关系方法引用。

### Knowledge-Enhanced Prompt Contrastive Learning

Yu 等提出面向多源用户行为图的知识增强自对比框架，使用关系编码的多头注意力、子图采样、提示文本对齐、VAE 过滤和多分类。已核实：Yu 等，IJCNN 2025，DOI:10.1109/IJCNN64981.2025.11229300。

区别：该方法依赖知识提示和行为图编码器；AdaGraph 不生成文本提示，也不改变 TS2Vec 主干。

### TGMC

TGMC 将多源异构图、GAT+GRU、掩码重建和多视图对比学习用于内部用户行为异常检测。OpenAlex 记录：Yuan 等，2026；本次未取得 DOI 和正式卷期信息。

这是当前最接近的“图 + 对比学习”工作，必须在相关工作和方法差异中主动处理：

- TGMC 构造异构图并使用专用于异常检测的图编码器；
- AdaGraph 不新增图编码器，校正的是 TS2Vec 训练中的实例正关系权重；
- TGMC 采用多视图掩码对比；
- AdaGraph 使用画像先验、EMA 反馈和 KL 信任域，把图更新限制在稳定行为关系附近。

如果最终检索确认 TGMC 已正式发表且代码可用，应尽量加入实验对照；至少需要提供概念对照表。

## 六、评价规范

- Kim 等指出 point-adjust 会系统性高估事件级检测能力，建议使用更严格的评价和局部事件指标。已核实：AAAI 2022，DOI:10.1609/AAAI.V36I7.20680。
- Wu 与 Keogh 指出部分时序异常检测基准存在缺陷，可能制造虚假进展。已核实：IEEE TKDE，DOI:10.1109/TKDE.2021.3112126。
- Demšar 系统讨论多数据集、多分类器的统计比较。已核实：JMLR 2006，DOI:10.5555/1248547.1248548。

AdaGraph 实验应同时报告点级指标、固定预算事件命中、延迟和 bootstrap 置信区间，不依赖 point-adjust 作为主要证据。

## 七、数据资源审计

### CERT 与 r1

CERT 仍是内部威胁研究的主要合成基准，但不同版本不能自动视为独立组织。本项目本地 `r1_audit.json` 已确认 CERT r1 的官方 answer key 没有恶意场景标签，因此 r1 不能承担 PR-AUC、K=10 事件召回或确认性统计。

### TWOS

Harilal 等提出 TWOS，用 gamified competition 记录恶意内部人员行为；DOI:10.1145/3139923.3139929。后续论文可交叉核对其为五天受控红蓝对抗，规模明显小于 CERT。数据记录 DOI 为 10.13140/RG.2.2.28309.42729，正式访问和许可证仍需审计。

本项目用途：外部真实感和小样本压力测试，不作主确认集。

### SPEDIA

Álvarez 等公开 SPEDIA 数据集，DOI:10.5281/zenodo.15525713，CC BY 4.0。官方描述称其来自 30 天演练，混合真实参与者攻击、角色模拟正常行为和 CERT 派生事件，并呈平衡的恶意/正常分布。

本地文件审计得到 72,250 行、25 列、17 个 `User` 值、25 个活动日期、41,541 个异常行；首个活动日已有异常。因此它适合会话/事件级迁移压力测试，但不能直接承担本项目 1000 级用户图的主确认。

### ChimeraLog

Yu 等的 Chimera 论文已被 NDSS 2026 接收，arXiv:2508.07745。论文构建三个组织场景、每组织 20 个员工代理、一个月的模拟和 15 类威胁，提供六类日志和用户日基准特征。官方项目页为 <https://sites.google.com/view/chimerallm>。

本次审计时，官方页面仍说明数据正在整理和迁移，因此：

- 不能写成“已获得确认数据”；
- 官方 GitHub README 说明旧 ChimeraLog 链接位于作者学校 OneDrive，作者毕业后账号受影响，旧链接已不可访问；
- 必须先审计实际下载、许可证、用户日字段、事件窗口和每组织事件数；
- 每组织只有 20 个用户，默认不满足主图确认规模。

### LANL Comprehensive Multi-Source Cyber-Security Events

Kent 公开的 LANL 数据集 DOI:10.17021/1179829，许可为 CC0。官方页面描述 58 个连续日、约 12 GB、1,648,275,307 条事件、12,425 用户和 17,684 台计算机，包含 authentication、process、DNS、network flow 和 known red-team events。

它比 SPEDIA/TWOS 更接近真实企业环境，但研究构念是红队入侵/凭证滥用，不是 CERT 式恶意内部人员。官网示例红队事件从 `t=151648` 秒附近开始，不能预设存在 30 天全局干净前缀。用户日特征、事件簇和用户关系图都需要单独冻结的外部适配协议。

## 八、相关工作可直接采用的段落骨架

时序自监督表示学习通过时间邻域、层次对比、软对比或聚类原型学习无标签表示。TS2Vec 通过层次化实例与时间对比获得时间戳级表示；SoftCLT 进一步用数据空间距离缓解假负样本，但其实例关系在训练前固定。NNCLR、PCL 和 SwAV 分别从近邻、原型与聚类分配获得语义正样本，说明固定实例判别目标并不充分。假负样本检测与去偏方法进一步指出，随机负样本可能来自同一语义类别，需要动态识别或概率校正。上述方法主要让正样本来源更丰富，但没有解决企业用户关系中的稳定先验与表示反馈之间的冲突。

内部威胁检测研究从规则、统计和行为画像发展到序列模型、图模型与实时活动级检测。LAN 通过活动级图学习同时建模时间依赖和跨序列关系；近期工作进一步结合异构图、知识提示或多视图对比。与这些方法不同，AdaGraph 不构造新的内部威胁检测器，而是在 TS2Vec 的训练目标中用画像先验锚定用户软正关系，并通过有限表示反馈校正假负样本。该方法的关键是信任域约束下的软图更新，以及异常用户不能被无约束地吸收到正常表示核心。

## 九、待办

1. 获取 TGMC 正式出处、卷期与代码状态。
2. 获取 Ripple2Detect、ATHITD 全文并核对具体机制。
3. 把 NNCLR 和 False Negative Cancellation 写成可执行基线。
4. 检索 2026 年以后“用户行为图 + 对比学习 + 内部威胁”的新工作，提交前再做最后一轮更新。
5. 完成 ChimeraLog、LANL 和 OpTC 的访问、许可证、用户日、事件和训练期审计。

