# 复现说明

## 环境

- Python 3.10
- PyTorch 2.x，GPU 实验使用 CUDA
- NumPy、Pandas、PyArrow、SciPy、scikit-learn、DuckDB、joblib

论文实验使用过 `PyTorch 2.11.0+cu128`、RTX 5060 笔记本和 `ts2vec` Conda 环境。其他兼容版本可通过调整 `requirements.txt` 使用。

## 初始化

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pwsh .\scripts\setup_third_party.ps1
```

`setup_third_party.ps1` 会：

- 检查 `references/ts2vec_official`。
- 下载 SoftCLT 到 `references/softclt_repo`。
- 将 SoftCLT 固定到 `14c638979b129075d7a1111e9f529b9a275ea394`。

如需手动配置 TS2Vec，可设置：

```powershell
$env:TS2VEC_REPO = (Resolve-Path .\references\ts2vec_official).Path
```

## 数据目录

原始数据和模型不进入 Git。将已有 processed 目录链接到运行时根目录。
下面的占位符需要替换为当前机器上的实际目录：

```powershell
$ProcessedRoot = "<path-to-cert-processed>"
pwsh .\scripts\link_data.ps1 `
  -ProcessedR42 "$ProcessedRoot\processed_r42" `
  -ProcessedR52 "$ProcessedRoot\processed_r52" `
  -ProcessedR62 "$ProcessedRoot\processed_r62" `
  -OmniAnomaly "<path-to-omnianomaly-server-machine-dataset>"
```

脚本会拒绝覆盖已存在的链接或目录。

正式发布前运行：

```powershell
python .\scripts\validate_release.py
```

该命令会检查路径、语法、大文件和完整单元测试。CPU 契约测试覆盖时间缺失
`-1 -> NaN`、训练期画像隔离、每日 Top-K 与七天冷却。

## 运行

从 `runtime/cert/` 运行。查看参数：

```powershell
Set-Location .\runtime\cert
python e0_e1_representation_preservation.py --help
python e2_softclt_training.py --help
python e4_joint_instance_pai.py --help
```

主实验的推荐顺序：

```powershell
# E0/E1
python e0_e1_representation_preservation.py --dataset r42 `
  --seeds 20261116 20261117 20261118 --epochs 10

# E2 temporal-only
python e2_softclt_training.py --variants T0 T1 T2 --tau-temps 1.5 `
  --seeds 20261116 20261117 20261118 --epochs 10 --batch-size 16

# E2 instance-soft
python e2_softclt_instance.py --seeds 20261116 20261117 20261118 `
  --epochs 10 --batch-size 16 --output-tag missing_safe_v2_20261117

# E3
python e3_multichannel_amplitude.py

# E4 CERT r4.2
python e4_joint_instance_pai.py --variants T3_instance_soft T4_full_soft

# E7/E8 SMD
python e7_smd_pai.py --epochs 10
python e8_smd_matched_t0.py

# E11 iForest
python e11_iforest_baseline.py

# Canonical 汇总
python build_canonical_audit.py
```

具体参数以各脚本 `--help` 为准。部分历史分析脚本没有完整 CLI，路径和实验选择写在脚本顶部。

## 已知运行边界

- `runtime/cert/` 依赖原实验的数据目录布局。
- 分层目录 `code/by_layer/` 用于阅读，不是自包含包。
- S3/S4 的若干历史脚本还依赖中间 `.npy` 或主工作目录的外部模块。
- SoftCLT 与官方 TS2Vec 都包含名为 `models` 的包，不能在同一 Python 进程直接混用。
- SoftCLT `encode()` 参数名是 `casual`，官方 TS2Vec 使用 `causal`。

