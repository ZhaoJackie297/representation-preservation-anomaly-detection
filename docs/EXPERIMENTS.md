# 实验矩阵

论文实验编号为 E0–E11。表中脚本均位于 `runtime/cert/`。

| 编号 | 目标 | 主要脚本 | 主要数据 |
| --- | --- | --- | --- |
| E0 | Native TS2Vec 表征与 mask L1 基线 | `e0_e1_representation_preservation.py` | CERT r4.2 |
| E1 | 冻结分数校准、表征库和 PAI 诊断 | `e1_saved_score_analysis.py`、`e0_e1_representation_preservation.py` | CERT r4.2 |
| E2 | SoftCLT T0–T4、47 维画像和机制审计 | `e2_softclt_training.py`、`e2_softclt_instance.py`、`e2_instance_mechanism.py` | CERT r4.2 |
| E3 | 余弦/欧氏库、magG/T2、通道聚合 | `e3_multichannel_amplitude.py` | CERT r4.2 |
| E4 | instance-soft 与欧氏/PAI 双端联合 | `e4_joint_instance_pai.py` | CERT r4.2 |
| E5 | r5.2 外部组织验证 | `e4_r52_joint_analysis.py`、`e5_r52_pai_analysis.py` | CERT r5.2 |
| E6 | r6.2 五事件压力审计 | `e4_r62_matched_t0.py`、`e6_r62_audit.py` | CERT r6.2 |
| E7 | SMD 跨数据类型 PAI | `e7_smd_pai.py` | SMD 28 台机器 |
| E8 | SMD SoftCLT 与匹配 T0 | `e8_smd_softclt.py`、`e8_smd_matched_t0.py` | SMD 28 台机器 |
| E9 | magG/T2 单组件消融 | `e9_pai_component_ablation.py` | CERT r4.2、SMD |
| E10 | 匹配对照与用户规模比较 | `e10_joint_control_comparison.py`、`e10_cert_joint_comparison.py` | CERT、SMD |
| E11 | iForest 外部基线 | `e11_iforest_baseline.py` | CERT r4.2 |

## 固定协议

- 随机种子：`20261116`、`20261117`、`20261118`。
- CERT 主实验：10 epochs、batch size 16、输出维数 64、隐藏维数 64、深度 5。
- K 网格：`5`、`10`、`20`。
- 告警冷却期：7 天。
- 表征库：MiniBatchKMeans，中心数 `min(500, max(1, round(0.1M)))`。
- 查询：最近 3 个中心均值。
- PAI 项目权重：`1.0:0.6:0.4`。
- PAI 官方敏感性权重：`0.6:0.4:0.2`。
- SMD：512 点窗口、10 epochs、训练 seed `20261116`。

## 结果位置

- 论文统一口径：`results/canonical/`。
- E0–E11 原始轻量结果：`results/representation_preservation/`。
- 历史结果与覆盖情况：对应目录中的 `WARNING_SUPERSEDED.md`、`audit.json` 和 `README.md`。

