# 实验矩阵

论文实验编号为 E0–E14。E0–E11 的脚本位于 `runtime/cert/`，
E12–E14 的扩展结果和运行入口位于 `research/ccf-b/`。2026-11-20
新增 E15–E16 作为升级实验批次。

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
| E12 | CERT 固定 10-seed 扩展验证 | `research/ccf-b/r4.2_10seed_paired_tests.csv`、`r5.2_10seed_paired_tests.csv` | CERT r4.2、r5.2 |
| E13 | CLUE-LDS 外部应用级复验 | `research/ccf-b/clue_lds_10seed_paired_tests.csv` | CLUE-LDS 半合成账号劫持 |
| E14 | 受控异常形态压力测试 | `research/ccf-b/phase1/controlled_signal_study.py` | CERT r4.2 正常序列 |
| E15 | SMD 5-seed 扩展与深度基线 | `research/ccf-b/smd_multiseed/`、`research/ccf-b/deep_baselines/` | SMD 28 台机器 |
| E16 | CERT 深度基线 pilot 与官方 PAI 权重 | `research/ccf-b/deep_baselines/` | CERT r4.2 |

## 固定协议

- 历史随机种子：`20261116`、`20261117`、`20261118`。
- CERT 主实验：10 epochs、batch size 16、输出维数 64、隐藏维数 64、深度 5。
- K 网格：`5`、`10`、`20`。
- 告警冷却期：7 天。
- 表征库：MiniBatchKMeans，中心数 `min(500, max(1, round(0.1M)))`。
- 查询：最近 3 个中心均值。
- PAI 项目权重：`1.0:0.6:0.4`。
- PAI 官方敏感性权重：`0.6:0.4:0.2`。
- SMD 历史结果：512 点窗口、10 epochs、训练 seed `20261116`。
- SMD 新扩展：seed `20261201`–`20261205`；T0/T4 与
  OmniAnomaly-adapted 使用 512 点窗口，Anomaly Transformer 使用官方
  100 点窗口。
- E12：seed `20261116`–`20261125`，固定矩阵不调参。
- E14：模型 seed `20261116`–`20261118`，标签 seed `20261119`；
  受控实验不作为外部验证。
- E16：CERT 深度基线 pilot 使用 `20261201`–`20261203`，只作探索性适配
  结果，不进入确认性 20-seed 家族。T0/T4 也已补跑相同 seed，并输出
  逐 seed 配对差值；仍未形成确认性 T4 端到端增量。

## 结果位置

- 论文统一口径：`results/canonical/`。
- E0–E11 原始轻量结果：`results/representation_preservation/`。
- E12–E16 汇总：`research/ccf-b/`、`research/ccf-b/phase1/results/`、
  `research/ccf-b/smd_multiseed/results/`、
  `research/ccf-b/deep_baselines/results/` 与
  `research/ccf-b/official_pai/`。
- 历史结果与覆盖情况：对应目录中的 `WARNING_SUPERSEDED.md`、`audit.json` 和 `README.md`。

