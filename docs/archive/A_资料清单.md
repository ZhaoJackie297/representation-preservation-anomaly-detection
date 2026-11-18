# A. 资料清单

## 一、代码

### S1 数据处理

目录：`code/S1_data`

- `build_r42_user_day.py`
- `build_r52_user_day.py`
- `build_r62_user_day.py`
- `build_r52_incidents.py`
- `build_r62_incidents.py`
- `prepare_r42_sequences.py`
- `prepare_r52_sequences.py`
- `prepare_r62_sequences.py`
- `audit_r42.py`
- `audit_r52.py`
- `audit_r62.py`

作用：原始 CERT 日志解析、用户日特征、LDAP、标签、27 维张量和时间切分。

### S2 双端优化

目录：`code/S2_dual_end`

用户点名脚本：

- `e2_softclt_training.py`
- `e2_instance_mechanism.py`
- `e2_softclt_instance.py`
- `e4_joint_instance_pai.py`
- `e2_full_soft_r52_train.py`
- `e4_r52_joint_analysis.py`

E8/E9/E10 主脚本：

- `e8_smd_softclt.py`
- `e8_smd_matched_t0.py`
- `e9_pai_component_ablation.py`
- `e10_cert_joint_comparison.py`
- `e10_joint_control_comparison.py`
- `e11_iforest_baseline.py`
- `build_e11_iforest_comparison.py`

### S3 风险处理

目录：`code/S3_risk`

包含 `s3*.py`、Top-K、冷却、队列、配额和动态风险主脚本，例如：

- `s3_dfm_v12.py`
- `s3_v2m_risk_queue.py`
- `risk_topk_r42.py`
- `quota_relationship_sim.py`
- `adaptive_count_quota.py`
- `production_protocol_r42.py`
- `multiscale_risk_r42.py`
- `multiscale_risk_r52.py`
- `multiscale_risk_r62.py`

### S4 证据输出

目录：`code/S4_evidence`

- `s4_risk_reports.py`
- `s4_ssl_evidence_reports.py`
- `s4_user_timelines.py`
- `frozen_model_evidence_cards.py`

### S5 评价验证

目录：`code/S5_evaluation`

- `s5_evidence_audit.py`
- `s5_ssl_evidence_audit.py`
- `e5_r52_pai_analysis.py`
- `e6_r62_audit.py`
- `e7_smd_pai.py`
- `e10_cert_joint_comparison.py`
- `e10_joint_control_comparison.py`
- `smd_pipeline.py`
- `analyze_r52_external.py`
- `tune_r52_sensitivity.py`
- 多种子基线脚本

### 完整参考快照

- `code/full_reference_root`：`E:\MarketRisker\work\cert` 根目录全部 156 个 Python 脚本；
- `code/research_reference`：research/ssl_user_repro 下全部 112 个 Python 脚本；
- 主目录代码是分层后的阅读入口，完整快照用于避免遗漏依赖。

## 二、实验设计文档

- `docs/E0-E10_实验设计卡.md`
- `docs/S1-S5_分层设计说明.md`
- `docs/current/有争议数据清单_修正版_20261117.md`
- `docs/current/有争议数据清单（已加修正决定）.md`
- `docs/current/SoftCLT_几何机制与成立条件_20261117.md`
- `docs/current/MarketRisker_外部AI审查与复算交接说明_20261117.md`
- `docs/current/下一课题_软正样本图谱偏置与异常可分性_20261117.md`
- `docs/current/下一课题_数学章节与实验对照表_简版_20261117.md`
- `docs/current/E11_iForest基线实验报告_20261117.md`
- `docs/current/基于自监督时序表征的用户行为异常检测_文献综述_20261117.md`

说明：

> 原项目没有形成 E0–E10 的独立预注册文件。上述文档根据代码、已有结果和实际运行记录事后补写，并明确区分“当时假设”和“实际结论”。

## 三、工作日志

- `docs/工作日志_双端优化_20261116-17.md`

包含：

- 实验运行时间近似记录；
- 运行内容；
- 主要结果；
- 报错与处理；
- 关键决策点。

## 四、原始结果

目录：`results/e0_e1` 至 `results/e11_iforest`

- 已包含 E0–E10 下全部 CSV 和 JSON；
- 共 131 个结果文件；
- 不包含 `.npy/.npz/.pth` 大文件；
- 点位级、固定 K、匹配对照、组件消融和事件审计明细均在对应目录。

目录：`canonical_audit_20261117`

- D1–D8 修正后的统一口径表；
- `baseline_protocol_map.csv`、`pai_variant_mapping.csv`；
- PAI 项目权重与官方权重固定敏感性；
- E2 逐种子重建汇总与覆盖审计；
- r4.2/r5.2/r6.2 修复后联合结果和配对差值；
- `data_provenance.json` 记录输入文件与哈希。

目录：`results/processed_metadata`

- 包含 processed_r42/r52/r62 下全部 JSON 元数据；
- 包括数据规模、切分、训练摘要、审计、模型配置等。

## 五、数据说明

- `cert_docs/r42_readme.txt`
- `cert_docs/r52_readme.txt`
- `cert_docs/r62_readme.txt`
- `cert_docs/r4.2_audit.json`
- `cert_docs/r5.2_audit.json`
- `cert_docs/r6.2_audit.json`
- `docs/CERT_数据说明.md`
- `docs/AI设计产物对照表.md`
- `docs/RAW_DATA_NOT_INCLUDED.md`

## 六、旧版基线材料

- `docs/legacy/AI自监督时序实验.txt`
- `docs/legacy/AI_3.8_Native基线原文摘录.md`

这是此前提供的 AI 自监督时序实验文档，包含 3.8 节基线实验的详细记录。

## 七、当前正式文档

目录：`docs/current`

- 表征信息保全完整报告 v2
- 申报书修正版
- 实例距离构造与效度
- 预研未核实项核验台账

目录 `docs/drafts` 还单独放有：

- `异常检测大创申请书初稿.docx`
- `CRET and SMD.docx`

## 八、其他相关材料

- `docs/research_reference`：research 目录中的报告、文本、JSON、CSV、HTML、PDF 等参考材料；
- `references`：E 盘 references 目录快照；
- `data_samples`：各版本 malicious account days、incidents、LDAP 缺失和用户日特征样本；
- `logs`：CERT 端与 research 端日志。
- `historical_outputs/outputs`：E 盘 outputs 目录全量快照，包含旧报告、实验 CSV、图表、申报书素材等；
- `historical_outputs/attachments`：历史粘贴附件；
- `historical_outputs/artifacts`：历史导出 PDF。

## 九、未包含的大型内容

未直接复制：

- 约 145 GB 的 CERT 原始与抽取数据；
- 146.7 GB 的全项目 CSV 总量；
- `.npz` 表征数组；
- `.pth` 模型权重；
- 大型缓存。
- `appdata`：浏览器缓存、Cookies、Local Storage 等运行态数据，可能含本机敏感状态，未打包。

处理方式：保留路径索引、数据审计、摘要元数据和结果明细；见 `docs/RAW_DATA_NOT_INCLUDED.md`。


