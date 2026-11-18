# CERT 画像先验信息量审计

日期：2026-11-18

目的：在查看新的 A3/A4 结果前，检查 CERT 47 维训练期行为画像是否也接近平坦，
并检验“主高斯核带宽过宽”这一实施解释。

## 结果

主带宽为中位 pairwise 距离；替代带宽为用户第 10 近邻距离的中位数。两者都在
训练期画像上计算，不使用验证/测试标签或模型分数。

| 数据集 | m | 主带宽 mean H_m | 主带宽超过阈值比例 | kNN mean H_m | kNN 超过阈值比例 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CERT r4.2 | 10 | 2.30154 | 1.000 | 2.22480 | 0.889 | 平坦 |
| CERT r4.2 | 5 | 1.60888 | 0.998 | 1.56898 | 0.911 | 平坦 |
| CERT r5.2 | 10 | 2.29736 | 0.9975 | 2.24142 | 0.8905 | 平坦 |
| CERT r5.2 | 5 | 1.60592 | 0.9975 | 1.57791 | 0.915 | 平坦 |

阈值为 `0.95 * log(m)`。因此 CERT 的主先验和 kNN 替代先验都没有提供可检验的
top-m 排序结构。

已按 R0 的 `log1p(count)`、时间哨兵 `>=0`、训练期 active mask、median/q90 和
mean/std 标准化复现。r4.2 当前的归一化距离中位数约为 `0.20788`，归档 R0
`distance_metadata.json` 为 `0.20847`，差异约 0.28%；这可能是历史运行版本/浮点
路径差异。平坦结论在两套数值上都很强，但正式论文应继续保留“near-reproduction”
字样，直到用归档训练脚本做一次 bit-level 复算。

## 解释边界

- 这是 `METHOD_NOT_SUPPORTED`，表示当前画像锚机制在 CERT 上也不可识别；
- 不是 CERT 数据门禁失败；
- 不能用 A5b 无差异证明画像无用；
- A3/A4 的端到端结果可以报告，但机制归因必须同时报告“批次共现”和“稀疏支撑/表示反馈”。

复现脚本与 JSON：

```text
research/ccf-b/audit_cert_profile_information.py
research/ccf-b/cert_profile_information_r42.json
research/ccf-b/cert_profile_information_r42_m5.json
research/ccf-b/cert_profile_information_r52.json
research/ccf-b/cert_profile_information_r52_m5.json
```
