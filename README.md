# Representation Preservation for Self-Supervised User Behavior Anomaly Detection

本仓库整理论文《基于表征信息保全的自监督时序异常识别与风险预警方法研究》对应的代码、实验协议、canonical 结果和修正版论文。

研究对象是以 TS2Vec 为编码器，在 CERT 用户日行为和 SMD 服务器指标上验证训练端 instance-soft SoftCLT 与打分端欧氏表征库的双端优化。仓库不包含原始 CERT/SMD 数据、模型权重和大型中间数组。

## 仓库内容

| 目录 | 内容 |
| --- | --- |
| `code/by_layer/` | 按 S1 数据处理、S2 表征检测、S3 风险处理、S4 证据报告、S5 评价验证整理的论文代码 |
| `runtime/cert/` | 保留原实验导入关系的可运行代码快照 |
| `results/canonical/` | 论文引用的统一口径结果、逐种子指标和配对比较 |
| `results/representation_preservation/` | E0–E11 的 CSV、JSON 和轻量审计记录 |
| `data/` | CERT 官方说明、许可证和脱敏字段样本，不含原始数据 |
| `docs/` | 代码地图、实验矩阵、数据协议、复现步骤和已知边界 |
| `paper/` | 修正版论文与事实性核查报告 |
| `references/ts2vec_official/` | TS2Vec 官方实现快照，保留 MIT 许可证 |

## 快速开始

需要 Python 3.10。GPU 实验建议使用 CUDA 版 PyTorch。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt

# 下载 SoftCLT，并将官方 TS2Vec 固定到记录版本
pwsh .\scripts\setup_third_party.ps1

# 将本地 processed 数据链接到运行时目录
pwsh .\scripts\link_data.ps1 `
  -ProcessedR42 "D:\datasets\cert\processed_r42" `
  -ProcessedR52 "D:\datasets\cert\processed_r52" `
  -ProcessedR62 "D:\datasets\cert\processed_r62" `
  -OmniAnomaly "D:\datasets\omnianomaly"

pwsh .\scripts\validate_release.ps1
```

实验入口统一位于 `runtime/cert/`：

```powershell
Set-Location .\runtime\cert
python e0_e1_representation_preservation.py --dataset r42
python e2_softclt_training.py --variants T0 T1 T2 --tau-temps 1.5
python e2_softclt_instance.py --seeds 20261116 20261117 20261118
```

完整的实验顺序、数据目录和结果映射见：

- `docs/REPRODUCIBILITY.md`
- `docs/EXPERIMENTS.md`
- `docs/CODE_MAP.md`
- `docs/DATA_AND_SPLITS.md`
- `docs/PAPER_MAPPING.md`
- `docs/KNOWN_LIMITATIONS.md`

## 结果口径

论文最终结果以 `results/canonical/` 为准。历史结果目录用于追溯，不应把不同评价协议的数字直接混表。关键口径包括：

- E2 中的 T0 是 SoftTS2Vec 内部参照，E0/E1 的 T0 是 Native TS2Vec，两者不能交叉混用。
- 项目 PAI-A1 权重为 `1.0:0.6:0.4`，官方 PAI 敏感性权重为 `0.6:0.4:0.2`。
- SMD 的 28/28 只适用于 PR-AUC；ROC-AUC 为 27/28，PA-F1@Top5 为 26/28。
- 时间特征缺失哨兵 `-1` 必须按缺失处理，不能转换为零点。
- r6.2 只有 5 个威胁事件，只用于压力审计，不作为强外部确认。

## 数据与模型

原始数据、预处理张量、表征数组和模型权重均未提交。请从官方渠道获取数据，并按 `docs/DATA_AND_SPLITS.md` 准备目录。仓库中的样例数据只用于检查字段格式。

## 第三方代码

- TS2Vec: <https://github.com/yuezhihan/ts2vec>
- SoftCLT: <https://github.com/seunghan96/softclt>

SoftCLT 通过 `scripts/setup_third_party.ps1` 下载到固定 commit，不直接复制到主仓库。TS2Vec 官方代码以 MIT 许可证随仓库保留。

## 许可证

本项目代码采用 MIT License。CERT/SMD 数据和第三方代码继续适用各自许可证。

