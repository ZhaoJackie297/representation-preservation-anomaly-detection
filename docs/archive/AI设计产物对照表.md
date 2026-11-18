# `AI自监督时序实验.txt` 设计产物对照表

## 一、原文是否包含

已包含：

- 完整原文：`docs/legacy/AI自监督时序实验.txt`
- 3.8 节单独摘录：`docs/legacy/AI_3.8_Native基线原文摘录.md`
- 原始大创初稿（主包）：`references/异常检测大创申请书初稿.docx`
- 原始大创初稿（整理副本）：`docs/drafts/异常检测大创申请书初稿.docx`
- 原始大创初稿（补充包）：`research/AI_design_missing_supplement_20261117/drafts/异常检测大创申请书初稿.docx`
- CERT/SMD 说明草稿：`docs/drafts/CRET and SMD.docx`
- 当前申报书修正版：`docs/current/基于表征信息保全的自监督时序异常识别_申报书修正版_20261117.md`

结论：

> 原文、3.8 节详细基线、大创初稿和当前修正版均已放入资料包。初稿存在主包 `references/`、`docs/drafts/` 和补充包 `drafts/` 三个同内容位置；引用时以 `references/` 为主，另外两处为整理副本。

## 二、原文中出现的 31 个产物名

原文中的部分文件名是设计阶段提出的“目标产物”，不一定曾在 E 盘以完全相同的文件名生成。下表逐项对照。

| 原文名称 | 包内状态 | 实际对应/位置 | 说明 |
| --- | --- | --- | --- |
| `datautils.py` | 已包含 | `references/ts2vec_official/datautils.py`、`references/softclt_repo/.../datautils.py` | 官方与 SoftCLT 版本都在 |
| `logon.csv` | 未包含原文件 | `cert_docs/r42_readme.txt` 等 | 原始日志体积大，代码和字段说明在；路径见 RAW_DATA_NOT_INCLUDED |
| `device.csv` | 未包含原文件 | 同上 | 同上 |
| `file.csv` | 未包含原文件 | 同上 | 同上 |
| `email.csv` | 未包含原文件 | 同上 | 同上 |
| `http.csv` | 未包含原文件 | 同上 | 同上 |
| `user_day_features.parquet` | 未包含完整大文件 | `data_samples/*user_day_features_sample.csv`；`results/processed_metadata/*user_day_summary.json` | 完整 parquet 在 E 盘原 processed 目录 |
| `sequence_dataset.npz` | 未包含大文件 | `results/processed_metadata/*sequence_metadata.json`；S1 代码 | 可由用户日特征重建 |
| `sequence_metadata.json` | 已包含版本化文件 | `results/processed_metadata/processed_r42/r4.2_sequence_metadata.json`，r52/r62 同 | 类名对应实际版本文件 |
| `malicious_account_days.parquet` | 已包含等价 CSV | `data_samples/r4.2_malicious_account_days.csv` 等 | 实际项目使用 CSV，不是 parquet |
| `incident_windows.parquet` | 已包含等价 CSV | `data_samples/r4.2_incidents.csv` 等 | 实际项目使用 incidents CSV |
| `data_audit.json` | 已包含版本化文件 | `cert_docs/r4.2_audit.json`、r5.2、r6.2 | 三版本分别保存 |
| `source_manifest.json` | 未按同名生成 | 数据审计 JSON、S1 代码、原始路径说明 | 审计 JSON 记录文件大小/字段/数量；未复制 145GB 原始文件 |
| `split_manifest.parquet` | 已包含等价 JSON | `processed_r42/r4.2_sequence_metadata.json` 的 `splits` 字段 | 切分信息存在 |
| `ts2vec_model.pth` | 未包含当前主模型 | 原路径 `processed_r42/r52/r62` | 模型二进制大；旧历史包中若出现只作历史材料 |
| `training.json` | 已包含版本化文件 | `results/processed_metadata/processed_r42/r4.2_ts2vec_training.json` 等 | 训练损失和配置 |
| `ts2vec_scores.npz` | 未包含大数组 | E0–E10 详细 CSV/JSON | 分数明细和指标已提供 |
| `daily_anomaly_scores.parquet` | 未包含大文件 | `r*.2_ts2vec_daily_errors.csv` 原路径 | 大型逐日分数未重复打包 |
| `causal_check.json` | 已包含等价字段 | `r4.2_ts2vec_training.json` 的 `causal_future_perturbation_max_diff` | 未单独生成同名文件 |
| `daily_risk_scores.parquet` | 未按同名生成 | S3 代码与历史 outputs 中对应风险结果 | 设计名为概念产物 |
| `daily_topk_queue.parquet` | 未按同名生成 | `r4.2_daily_topk_alerts.csv`、`r4.2_production_alerts.csv` 等历史输出 | 现有实现多为 CSV |
| `risk_manifest.json` | 未按同名生成 | S3 汇总 JSON 与历史 outputs | 原设计名未成为统一当前产物 |
| `review_state.json` | 未按同名生成 | S4 复核/证据相关旧输出 | 当前 S4 尚未完整重接到 47 维 T4 |
| `user_timelines.parquet` | 未按同名生成 | `s4_user_timelines.py` 及其历史输出 | 当前为代码+历史原型 |
| `evidence_audit.json` | 未按同名生成 | `s5_evidence_audit.py`、`s5_ssl_evidence_audit.py` 及 outputs | 当前未统一命名 |
| `point_metrics.json` | 已包含等价 CSV/JSON | E0–E10 的 PR/ROC 明细与汇总 | 例如 `metric_summary.csv` |
| `topk_metrics.csv` | 已包含等价明细 | E2–E5、E10 的 K=5/10/20 明细 | 文件名按实验拆分 |
| `external_metrics.json` | 已包含等价文件 | `e5/r52`、`e7/smd`、`e10` | external test 结果 |
| `baseline_comparison.csv` | 已包含 | `historical_outputs/outputs/.../baseline_comparison.csv` | 历史基线对照 |
| `statistical_tests.csv` | 已包含等价 | `e9/smd_component_tests.csv`、`e10/*.json` | 统计表按实验拆分 |
| `evaluation_manifest.json` | 未按同名生成 | E0–E10 各目录 summary/audit JSON | 评价元数据分散于各实验目录 |

## 三、实际完整性判断

### 完整包含

- `AI自监督时序实验.txt`；
- 3.8 节详细基线原文；
- 初稿 DOCX；
- 当前修正版申报书；
- 各版本官方 readme、license、数据审计；
- 用户日摘要和序列元数据；
- 小体积标签、事件、特征样本；
- S1–S5 主代码和全量代码快照；
- E0–E10 全部 CSV/JSON；
- outputs/research/references/日志。

### 未包含原文件

- 原始五类日志 CSV；
- 完整用户日 parquet；
- 完整时序 NPZ；
- 大型逐日分数 CSV；
- NPY/NPZ/PTH 数组和模型。

### 未按同名生成

原文中有一部分文件名属于设计目标，如：

- `risk_manifest.json`
- `review_state.json`
- `user_timelines.parquet`
- `evidence_audit.json`
- `evaluation_manifest.json`

其中部分只有代码原型或历史 CSV/JSON 等价产物，不是当前统一产物。资料包没有伪造同名空文件。

## 四、准确回答

如果你的问题是“这份 AI 文本和相关设计资料在不在包里”：

> 在。

如果你的问题是“AI 文本里提到的每一个概念文件名，是否都有完全同名的当前文件”：

> 不是。31 个名称中，一部分已按版本化名称存在，一部分是大文件未复制，还有一部分是设计阶段概念名，当前只有代码或等价 CSV/JSON。

建议后续若要把该文本直接作为“大创初稿的数据与产物附录”，应使用本对照表，而不是声称所有同名文件均已生成。
