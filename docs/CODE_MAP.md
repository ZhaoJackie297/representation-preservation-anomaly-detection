# 代码地图

仓库保留两套视图：

- `code/by_layer/` 面向论文阅读，按 S1–S5 分层。
- `runtime/cert/` 保留原实验的平铺导入关系，供实际运行。

两套视图的代码来自同一实验快照。运行时应使用 `runtime/cert/`，不要直接把 `code/by_layer/` 中的单个文件复制出来执行。

## S1 数据处理

主要代码：

- `audit_r42.py`、`audit_r52.py`、`audit_r62.py`：原始文件、字段、时间和标签审计。
- `build_r42_user_day.py`、`build_r52_user_day.py`、`build_r62_user_day.py`：27 维用户日特征和活跃掩码。
- `build_r52_incidents.py`、`build_r62_incidents.py`：威胁实例与恶意账户日映射。
- `prepare_r42_sequences.py`、`prepare_r52_sequences.py`、`prepare_r62_sequences.py`：固定日期网格和模型输入张量。

核心不变量：

- 计数特征缺失填 0。
- 时间特征缺失使用 `-1` 哨兵，在画像和距离计算前转为缺失。
- 标签不进入 S1–S4 的特征、训练、软权重、聚类和评分校准。
- 全局切分边界参考首个恶意事件日期，因此属于 label-informed split。

## S2 自监督检测与双端优化

### 官方 TS2Vec 接入

- `train_r42_ts2vec.py`
- `train_r52_ts2vec.py`
- `train_r62_ts2vec.py`
- `e0_e1_representation_preservation.py`
- `e1_saved_score_analysis.py`

### instance-soft SoftCLT

- `e2_softclt_training.py`：T0/T1/T2 时间项对照。
- `e2_softclt_instance.py`：T3/T4 instance-soft 与 full SoftCLT。
- `e2_full_soft_r52_train.py`、`e2_full_soft_r62_train.py`：外部数据完整 SoftCLT。
- `e2_instance_distance_validation.py`：47 维画像的时间稳定性。
- `e2_instance_mechanism.py`、`e2_mechanism_audit.py`：邻域、有效秩和谱机制审计。
- `s2_soft_graph_spectral_analysis.py`：图 Dirichlet 能量和高频残差分析。

### 打分端与消融

- `e3_multichannel_amplitude.py`：mask L1、余弦库、欧氏库、PAI 和通道聚合。
- `paI` 相关常量位于 `e3_multichannel_amplitude.py` 与 `e7_smd_pai.py`。
- `e9_pai_component_ablation.py`：magG/T2 单组件和完整 PAI 消融。

### 联合与外部验证

- `e4_joint_instance_pai.py`：CERT r4.2 双端联合。
- `e4_r52_joint_analysis.py`：CERT r5.2 外部组织验证。
- `e4_r62_matched_t0.py`、`e4_r62_joint_analysis.py`：CERT r6.2 匹配 T0 压力审计。
- `e7_smd_pai.py`：SMD PAI。
- `e8_smd_softclt.py`、`e8_smd_matched_t0.py`：SMD SoftCLT 与匹配 T0。
- `e10_joint_control_comparison.py`、`e10_cert_joint_comparison.py`：匹配对照和用户规模控制。
- `e11_iforest_baseline.py`：27 维用户日 iForest 基线。

## S3 风险处理

论文将 S3 定位为历史系统原型，不作为本轮主创新。相关代码：

- `production_protocol_r42.py`：生产协议和 Top-K 告警。
- `risk_topk_r42.py`：风险聚合和冷却。
- `s3_v2m_risk_queue.py`：固定预算风险队列。
- `s3_severity_only_control.py`：无权严重度控制。
- `s3_dfm_v12.py`、`s3_e_cusum_risk.py`、`s3_gaussian_hmm_risk.py` 等：历史动态风险尝试。

## S4 证据报告

S4 尚未与当前 47 维 T4 表示重新连接。相关代码：

- `s4_risk_reports.py`
- `s4_ssl_evidence_reports.py`
- `s4_user_timelines.py`
- `frozen_model_evidence_cards.py`
- `channel_evidence_experiment.py`

## S5 评价验证

- `quota_relationship_sim.py`：固定 Top-K、7 天冷却、PR/ROC 与事件命中。
- `s5_evidence_audit.py`、`s5_ssl_evidence_audit.py`：证据审计。
- `build_canonical_audit.py`：D1–D8 口径修正和 canonical 汇总。
- `build_e11_iforest_comparison.py`：iForest、T0 和 T4+PAI-A1 的统一对照。
- `build_softclt_geometry_summary.py`：SoftCLT 几何机制汇总。

## 重要共享文件

- `quota_relationship_sim.py`：数据加载、评分、Top-K 和评价函数。
- `application_utility_analysis.py`：应用效用指标。
- `risk_topk_r42.py`、`topk_trend_r42.py`：风险队列和历史趋势。
- `v2_pipeline.py`、`v2_m_pipeline.py`：S3 历史原型依赖。

## 运行顺序

建议按以下顺序执行：

1. S1：审计、用户日特征、威胁实例、序列数据。
2. E0/E1：Native TS2Vec 和冻结评分分析。
3. E2：T0–T4 训练端机制实验。
4. E3：多通道幅度与 PAI 组件。
5. E4/E5/E6：CERT 联合与外部验证。
6. E7/E8/E9：SMD 和组件消融。
7. E10/E11：匹配对照、用户规模和 iForest。
8. `build_canonical_audit.py`：生成最终 canonical 口径。

