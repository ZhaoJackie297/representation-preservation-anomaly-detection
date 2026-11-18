# 结果目录说明

## E0–E10

以下目录由 `E:\MarketRisker\work\cert\experiments\representation_preservation` 原样复制，仅包含 `.csv` 和 `.json`：

| 目录 | 内容 |
| --- | --- |
| `e0_e1` | Native TS2Vec 表征、cosine/Euclidean 表征库、冻结分数校准 |
| `e2` | temporal-only SoftCLT、机制结果 |
| `e2_instance` | instance-soft、47 维画像、机制审计、距离效度 |
| `e3` | magG/T2、多通道幅度和组件变体 |
| `e4_joint` | r4.2/r5.2/r6.2 双端联合、T0m、用户规模敏感性 |
| `e5` | r5.2 外部 PAI |
| `e6` | r6.2 Native 审计 |
| `e7` | SMD PAI |
| `e8` | SMD T4 与匹配 T0 |
| `e9` | magG/T2 单组件消融 |
| `e10` | CERT/SMD 匹配比较与总体比较 |

文件命名中的：

- `metrics_by_seed.csv` 是逐随机种子明细；
- `metric_summary.csv` 是均值汇总；
- `audit.json`、`summary.json` 是协议/审计元数据；
- 以 seed 或机器命名的文件是具体运行结果。

## processed_metadata

包含 processed_r42/r52/r62 的 JSON 元数据和摘要，例如：

- 序列形状；
- 日期切分；
- active cells；
- 用户日事件总量；
- 标签数；
- 训练摘要；
- 模型配置。

## 未包含

- 大型 `.npz` 分数数组；
- `.npy` 表征；
- `.pth` 模型；
- 原始事件 CSV。

这些文件在 E 盘原实验目录中，路径见 `docs/A_资料清单.md` 与 `code/DEPENDENCIES.md`。
