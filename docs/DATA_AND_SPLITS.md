# 数据与切分

## CERT

论文使用 CERT Insider Threat Dataset v4.2、v5.2 和 v6.2。原始数据由卡内基梅隆大学 CERT 发布，仓库不包含原始日志。数据许可证和官方说明位于 `data/cert_docs/`。

| 数据集 | 用户数 | 日期数 | 训练 | 验证 | 测试 | 角色 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CERT r4.2 | 1,000 | 501 | 159 天 | 144 天 | 198 天 | 主开发与内部时间外测试 |
| CERT r5.2 | 2,000 | 517 | 104 天 | - | 413 天 | 外部组织验证 |
| CERT r6.2 | 4,000 | 516 | 222 天 | - | 294 天 | 五事件压力审计 |

r6.2 联合实验还使用与 T4 匹配的 1,000 用户训练子集。

## SMD

SMD 包含 28 台服务器。原有结果使用 512 点窗口、10 epochs、seed
`20261116`。2026-11-20 扩展使用新 seed `20261201-20261205`，每个
seed 都对 28 台机器逐一训练和测试。T0/T4 与 OmniAnomaly-adapted 使用
512 点窗口；Anomaly Transformer 按官方配置使用 100 点窗口，因此后者
明确标为适配基线。

SMD 只用于跨数据类型外部验证，不与 CERT 的日级样本数直接比较。

## 期望目录

将处理后数据链接到 `runtime/cert/`：

```text
runtime/cert/
├─ processed_r42/
│  ├─ r4.2_sequence_dataset.npz
│  ├─ r4.2_user_day_features.parquet
│  ├─ r4.2_malicious_account_days.csv
│  └─ r4.2_incidents.csv
├─ processed_r52/
├─ processed_r62/
└─ OmniAnomaly/
   └─ ServerMachineDataset/
      ├─ train/
      ├─ test/
      └─ test_label/
```

实际文件名以 `docs/archive/A_资料清单.md`、`docs/archive/CERT_数据说明.md` 和各构建脚本为准。大文件由 `scripts/link_data.ps1` 建立 Junction，不复制进仓库。

## 标签与缺失

- 用户日标签用于点级 PR-AUC、ROC-AUC。
- 官方威胁实例窗口用于固定 K 事件命中。
- 标签不进入特征构造、模型训练、软权重、表征库聚类或评分校准。
- 时间特征缺失使用 `-1` 哨兵。画像计算时先转 NaN，再按用户计算 `nanmedian` 和 `nanquantile`。
- 计数特征缺失填 0。
- 全局切分边界参考首个恶意事件日期，因此论文明确将其写成 `label-informed split`，而不是完全 label-blind。

## Canonical 口径

优先读取：

- `results/canonical/canonical_metric_summary.csv`
- `results/canonical/canonical_metrics_by_seed.csv`
- `results/canonical/joint_paired_comparisons.json`
- `results/canonical/data_provenance.json`

`results/representation_preservation/` 是原始轻量结果的追溯目录。引用具体数字前先查看 `baseline_protocol_map.csv` 与 `pai_variant_mapping.csv`。

