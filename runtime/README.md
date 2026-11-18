# 可运行代码快照

`cert/` 保留原 CERT 实验目录中的顶层 Python 模块和导入关系。为了让现有脚本保持兼容，本目录刻意没有改造成 Python package。

运行时以 `runtime/cert/` 为工作目录：

```powershell
Set-Location .\runtime\cert
python e0_e1_representation_preservation.py --help
```

`processed_r42`、`processed_r52`、`processed_r62`、`OmniAnomaly/` 和运行时 `experiments/` 不进入 Git，由 `scripts/link_data.ps1` 或手动建立链接。

