# Canonical 数据口径

生成日期：2026-11-17  
生成脚本：`runtime/cert/build_canonical_audit.py`

## 使用顺序

1. `baseline_protocol_map.csv`：先确认数字属于哪个评价协议；
2. `pai_variant_mapping.csv`：区分 PAI-A1、S5-A3 和官方权重；
3. `d2_a3_reconciliation.csv`：A3 与 E0/E1 S5 的逐 seed 重算校验；
4. `canonical_metric_summary.csv`：按 seed 重算后的统一汇总；
5. `canonical_metrics_by_seed.csv`：逐 seed 原始规范表；
6. `pai_weight_sensitivity_summary.csv`：项目权重与官方固定权重对照；
7. `joint_paired_comparisons.json`：匹配 T0 的逐 seed 配对差值；
8. `softclt_mechanism_geometry.csv`：T0–T4 的正样本图配置、邻域几何和性能；
9. `softclt_geometry_T4_minus_T3.csv`：temporal-soft 在 instance-soft 之上的增量；
10. `softclt_spectral_contrasts.csv`：三 seed 图 Dirichlet 与高频残差配对差值；
11. `../s2_graph_spectral/three_seed_20261117/`：逐 seed 谱实验原始汇总；
12. `e11_iforest_vs_reference.csv`：iForest、T0 与 T4+PAI-A1 的 test 对照；
13. `data_provenance.json`：输入文件路径与哈希。

逐 seed CSV 和 provenance JSON 中保留的旧 Windows 绝对路径属于实验当时
的溯源元数据，不是当前仓库的默认路径。当前运行入口使用显式参数或
`runtime/cert/` 下的数据链接。

## 关键口径

- 3.8 旧生产协议不能与 E0–E10 原始 S2 混表；
- E2 T0 是 SoftTS2Vec 内部参照，不是 E0/E1 Native T0 的逐位复刻；
- E0/E1 `S5_pai_full_global` 是 A3；
- E4 联合实验的 `pai_a1` 是 A1；
- 项目权重为 `1.0:0.6:0.4`；
- 官方 PAI 标量权重为 `0.6:0.4:0.2`；
- 时间缺失 `-1` 已按缺失处理，不再映射成零点；
- 测试集未参与权重、阈值、K 或距离选择。
- iForest 只使用训练期活跃用户日拟合，评价复用同一 Top-K/冷却协议。

完整的逐项修正见：

`docs/archive/current/有争议数据清单_修正版_20261117.md`
