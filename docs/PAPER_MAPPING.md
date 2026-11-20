# 论文与代码映射

| 论文内容 | 主要代码 | 结果 |
| --- | --- | --- |
| 27 维用户日特征 | `build_r42_user_day.py`、`build_r52_user_day.py`、`build_r62_user_day.py` | `results/canonical/`、`data/samples/` |
| Native TS2Vec | `e0_e1_representation_preservation.py`、`train_r42_ts2vec.py` | `results/representation_preservation/e0_e1/` |
| 五种训练配置 T0–T4 | `e2_softclt_training.py`、`e2_softclt_instance.py` | `results/representation_preservation/e2/`、`e2_instance/` |
| 47 维画像和 instance-soft | `e2_softclt_instance.py`、`e2_instance_distance_validation.py` | `results/canonical/e2/` |
| 邻域/有效秩/谱机制 | `e2_instance_mechanism.py`、`s2_soft_graph_spectral_analysis.py` | `results/canonical/softclt_*.csv` |
| 余弦、欧氏库和 PAI | `e3_multichannel_amplitude.py` | `results/representation_preservation/e3/` |
| r4.2 双端联合表 7 | `e4_joint_instance_pai.py` | `results/canonical/canonical_metric_summary.csv` |
| r5.2 外部验证表 8 | `e4_r52_joint_analysis.py`、`e5_r52_pai_analysis.py` | `results/canonical/` |
| r6.2 压力审计表 9 | `e4_r62_matched_t0.py`、`e4_r62_joint_analysis.py` | `results/canonical/` |
| SMD 匹配对照表 10 | `e8_smd_matched_t0.py` | `results/representation_preservation/e8/` |
| iForest 基线表 11 | `e11_iforest_baseline.py` | `results/representation_preservation/e11_iforest/` |
| CERT 10-seed 表 12 | `research/ccf-b/r4.2_10seed_paired_tests.csv`、`r5.2_10seed_paired_tests.csv` | `research/ccf-b/` |
| Coverage batch 与 A3/A5a/A5b | `research/ccf-b/phase0/common_coverage_sampler.py`、`research/ccf-b/clue_lds/build_prior_graphs.py`、`build_profile_prior.py` | `research/ccf-b/r4.2_10seed_coverage_summary.csv`、`COVERAGE_BATCH_CONSTRUCTION.md` |
| CLUE-LDS 表 13 | `research/ccf-b/clue_lds_10seed_paired_tests.csv` | `research/ccf-b/phase1/results/clue_coverage_v1/` |
| 异常形态表 14–15 | `research/ccf-b/phase1/controlled_signal_study.py` | `research/ccf-b/phase1/results/r42_temporal_v1/` |
| 附录 C 表 17–18 | `research/ccf-b/phase1/controlled_signal_study.py` | `research/ccf-b/phase1/results/` |
| v6 Word 构建 | `scripts/build_v5_extended.py` | `paper/manuscript_v6.docx` |
| S3 Top-K 风险队列 | `production_protocol_r42.py`、`risk_topk_r42.py` | 论文作为历史原型 |
| S4 证据报告 | `s4_risk_reports.py`、`s4_ssl_evidence_reports.py` | 论文明确为历史原型 |
| S5 评价协议 | `quota_relationship_sim.py`、`s5_evidence_audit.py` | canonical 与各 E 目录 |

## 论文修正

`paper/manuscript.docx` 与 `paper/manuscript_v6.docx` 是当前 v6 扩展稿，
新增 §7.9 和附录 C；`paper/manuscript_v5.docx` 是归档的 v5 R0。
`paper/fact_check_20261118.md`
记录 v5 修正前事实性核查。`results/canonical/` 优先于任何历史截图、
草稿或旧汇总表。

