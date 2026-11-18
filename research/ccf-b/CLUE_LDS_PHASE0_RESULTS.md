# CLUE-LDS 数据与 Phase 0 结果

日期：2026-11-18

状态：数据访问、哈希、全量 Schema、24-pair 半合成账号劫持、用户日、画像/关系图、10-seed 覆盖和训练单步 smoke 已通过。完整 A0/A1 训练入口、训练期内部合成异常验证、CLUE 两 seed pilot 和外部模型矩阵尚未开始。

## 1. 下载与校验

- 数据 DOI：`10.5281/zenodo.7119953`
- 官方文件：`clue.zip`
- 文件大小：635,105,552 bytes
- 官方 MD5：`9e318370f96b68077667e9cdc05f26a5`
- 本地 MD5：`9E318370F96B68077667E9CDC05F26A5`，一致
- `clue.zip` SHA-256：`0C9EADB104ACF1DA6DE738BA9BABE957C83CD8602A01FA6D846A6EA4A6611D96`
- 解压文件：`clue.json`
- 解压大小：14,862,210,499 bytes
- `clue.json` SHA-256：`2CFE6C172104954DB574DD7648186E7A148FA1D553BD9F80212AE3F8C28956C5`

## 2. 全量审计

使用 DuckDB 对 50,522,931 条 JSONL 事件完成全量扫描：

| 项目 | 结果 |
| --- | ---: |
| 事件数 | 50,522,931 |
| 唯一用户 | 5,389 |
| 唯一日期 | 1,910 |
| 用户日 | 83,147 |
| `uidType=name` | 50,377,040 |
| `uidType=ipaddress` | 145,891 |
| 事件类型 | 49 |
| 时间范围 | 2017-07-07 至 2022-09-29 |

主要事件：

| 事件类型 | 数量 |
| --- | ---: |
| file_accessed | 18,447,579 |
| login_attempt | 13,227,628 |
| login_successful | 12,920,800 |
| file_written | 1,796,064 |
| admin_settings_changed | 1,091,611 |
| file_updated | 920,986 |
| file_created | 876,081 |

Schema smoke 和全量扫描均无解析错误。

## 3. 半合成账号劫持协议

原始 CLUE-LDS 只有正常行为。标签由本项目基于官方切换算法实现、用固定 seed 构造，不是官方恶意标签，也不是真实攻击记录。

冻结协议：

```text
label_seed=20261101
training_seeds=20261116..20261125
pairs=24
events=48
min_similarity=0.1
max_similarity=0.6
switch_window=2022-03-01..2022-08-30
min_days_before=35
min_days_after=30
profile_cutoff=2022-02-28
min_profile_days=20
```

结果：

- 24 个用户对；
- 48 个用户级事件；
- 切换日期 2022-03-03 至 2022-07-25；
- 标签相似度 0.2109–0.5961；
- 48 个目标用户都在训练期有活动，避免“无画像的新用户异常”。

标签文件：

```text
hijack_labels_24pairs_20261101.csv
bytes=5,526
SHA-256=2441744DED393381C9B42D3E1EFB53B11137810D8F94611FCF76BC0C67D248CA
```

20-pair/40-event 文件只保留为开发历史，不进入最终确认统计。

## 4. 5,389 到 838 用户筛选

规则在任何模型结果可见前冻结：

1. 只保留 `uidType=name`；
2. 所有 48 个注入目标用户必须保留；
3. 其他用户至少在训练期出现一次；
4. 非目标用户按训练期事件数降序；
5. 最多保留 1,000 用户；
6. 实际只有 838 个 `name` 用户满足训练期活动条件；
7. 任一注入目标用户训练期全空则整体失败；
8. 不回填全空目标用户，不使用评价期检测结果或异常标签。

## 5. 标签距离与画像距离独立性

pair 选择距离：

```text
0.7 * action Jaccard
+ 0.3 * average-events-per-active-day ratio
```

画像关系使用：

```text
52 维用户日特征的 median/q90 + active fraction
训练用户中位数/IQR 标准化
标准化欧氏距离 + Gaussian kernel
```

两者不是同一个距离。正式审计对 24 个 pair 计算：

```text
Spearman(label similarity, profile affinity) = -0.1991304347826087
abs(rho) = 0.1991304347826087 < 0.5
status = PASS_DISTANCES_DISTINCT
```

因此当前 pair 选择没有触发预注册的“高相关”失败条件，不需要重新生成标签。该审计不使用模型结果。

复现文件：

- `research/ccf-b/clue_lds/audit_label_prior_independence.py`
- `research/ccf-b/clue_lds/label_prior_independence_20261118.json`

## 6. 用户日数据

处理后：

```text
users=838
dates=1,911
features=52
active_user_days=71,877
events=48
train=2017-07-07..2022-02-28
eval=2022-03-01..2022-09-29
```

特征为 49 类事件计数、总事件数、事件类型数和外部 IP 事件数。

最终数据集：

```text
clue_user_day_dataset.npz
bytes=1,559,876
SHA-256=E503014FDD5E9BE9D57908D1DC267C74425BC8185E26D9259AF6B0F52615E571
```

元数据：

```text
clue_user_day_metadata.json
bytes=57,077
SHA-256=E272E7A51DF9603D441F4F05CF6A38E84F75EE847D0C3AC0C779939A1FA63D3E
```

## 7. 画像先验与平坦性

- 画像维度：105（52 个 median、52 个 q90、1 个活跃比例）；
- 每用户只使用训练期数据；
- 训练用户中位数/IQR 标准化；
- 高斯核带宽为全体 pairwise 距离中位数；
- 概率下限 `1e-12`；
- pairwise 距离中位数约 9.71。

全图行熵：

```text
mean   = 5.9308
median = 6.4906
max    = log(837) = 6.7298
```

top-10 支撑集：

```text
flat_threshold = 0.95 * log(10) = 2.1874558383
mean H_10      = 2.1689294186
median H_10    = 2.3025850924
fraction H_10 > threshold = 0.8997613365
```

按预注册规则触发：

```text
FLAT_PROFILE_PRIOR
```

后果：

- A3/A5b 的画像机制贡献记为“不可检验”；
- 不记为 A3 失败；
- A3/A4 的端到端比较仍可运行；
- 必须报告 `P_m` 熵、beta 和实际 KL。

画像先验：

```text
clue_profile_prior.npy
bytes=2,809,104
SHA-256=9BBDFC575479AC452AEA3A74CAD5260C5134DFC77F81D7A06E7B3EE0AB768526
```

可复现判定：

- `research/ccf-b/clue_lds/audit_profile_prior_information.py`
- `research/ccf-b/clue_lds/profile_prior_information_20261118.json`

## 8. 组与分享关系图

使用训练期：

- 用户加入/移除组；
- shared/unshared user；
- shared/unshared group；
- permission changed。

结果：

- group edges：1,488；
- share edges：1,121；
- 非零关系边：2,140；
- 孤立用户：565 / 838；
- 权重：`0.6_group_cooccurrence + 0.4_log1p_share_interactions`；
- 行归一化。

该图只作辅助组织结构或关系质量评价，不能作为全用户主先验，也不能用其覆盖不足解释 A3/A5b 结果。

## 9. 真实图覆盖门禁

从画像先验构造：

- A3：画像 top-10；
- A5a：同度序列随机图；
- A5b：画像节点置换图；
- batch size=16，8 个设计槽位 + 8 个随机填充槽位；
- 目标覆盖 80%；
- 10 个正式 seed。

| seed | 覆盖 batch | A3 | A5a | A5b | 零曝光用户 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20261116 | 948 | 87.30% | 80.04% | 86.06% | 0 |
| 20261117 | 954 | 86.46% | 80.06% | 86.27% | 0 |
| 20261118 | 950 | 85.95% | 80.06% | 85.65% | 0 |
| 20261119 | 959 | 86.78% | 80.02% | 85.59% | 0 |
| 20261120 | 957 | 87.85% | 80.01% | 85.49% | 0 |
| 20261121 | 961 | 87.41% | 80.08% | 86.77% | 0 |
| 20261122 | 957 | 87.43% | 80.04% | 86.06% | 0 |
| 20261123 | 959 | 88.25% | 80.05% | 86.88% | 0 |
| 20261124 | 948 | 86.24% | 80.06% | 86.02% | 0 |
| 20261125 | 958 | 87.63% | 80.06% | 85.95% | 0 |

汇总：

- 10 个 seed 全部通过；
- 无零曝光用户；
- 10 张 batch table SHA-256 全部唯一；
- 相邻 seed 用户对 Jaccard：0.2184–0.2216；
- 覆盖 batch 数：948–961；
- A5a 总是最后一个达到 80%，说明随机图是当前覆盖瓶颈。

覆盖文件：

```text
coverage_10seed.json
bytes=11,149
SHA-256=CB651D1A8F4D4793D77F869F3383360655E9F1773BD8E6B194EC3E515D55CD3B
```

先验图：

```text
clue_prior_graphs.npz
bytes=1,108,326
SHA-256=EFCA9CEFF1A5683AC62C90EC3F39AEE5E0B75FD10243D535D606225A3018F608
```

## 10. 最终产物哈希

`processed_final/` 的 12 个文件和 1 个标签文件共 13 个最终产物，已冻结在：

```text
research/ccf-b/clue_lds/artifact_hashes_20261118.json
```

哈希清单包含每个产物的绝对路径、字节数和 SHA-256；关键字段不得在其他文档中手工替换，引用时以该 JSON 为准。

## 11. 真实训练接线 smoke

`CoverageSoftTS2Vec` 已实现为显式 batch table 训练器，替代原随机 shuffle：

- 同一 batch table 驱动所有匹配变体；
- 只使用同批用户对；
- 记录每 batch `active_pair_mass` 和 `realized_pairs`；
- 不依赖 SoftCLT 的任务评估模块。

CPU smoke：

```text
users=16
timesteps=256
features=52
realized_pairs=240
active_pair_mass=16.0
loss=5.898293
```

该结果只证明软关系权重进入训练 loss，不证明性能。

真实 `processed_final` 的 2-step `A0-official` 入口 smoke 也已通过：

```text
users=838
timesteps=1698 (training prefix)
features=52
optimizer_steps=2
loss=3.573866 -> 3.015208
batch_table_sha256=24add3f7e3e551c17fa8bdcb2c23f2313f88580ef23d782293be603c117cbed5
```

这证明用户日、画像用户顺序、图和随机批次表能正确对齐，但不代表 A1-Cov 已训练或 pilot 已完成。

## 12. 统计功效

账号劫持每次切换一对用户，同 pair 的两个目标事件共享切换，因此有效独立单元是 pair，不是 48 个独立事件。

在 24 个 pair、80% 功效、单侧 alpha=0.025 下：

| baseline recall | cluster SD=0.5 | cluster SD=1.0 | cluster SD=1.5 |
| ---: | ---: | ---: | ---: |
| 10% | 23.37 pp | 25.95 pp | 28.61 pp |
| 25% | 27.21 pp | 28.25 pp | 29.53 pp |
| 50% | 26.31 pp | 24.46 pp | 22.97 pp |

因此：

- 80% 功效 MDE 约为 23.0–29.5 个百分点；
- 对 5–10 pp 的中小效应功效不足；
- 不显著不能解释为方法一定无效；
- CLUE-LDS 只能定位为“有限功效外部效应复验”。

CLUE-LDS pilot 使用独立门槛：

```text
PR0 = A0-matched 的 PR-AUC
positive_PR = Delta_PR >= max(0.001, 0.05 * PR0)
positive_recall = Delta_recall >= 3 / 48 = 0.0625
```

必须使用 2 个独立训练 seed；同一 endpoint 在两个 seed 上均为正且均值过阈值才算 pilot 达标。单 seed 只作实现 smoke，不能决定继续或停止，也不能用于调参。

## 13. 当前判定

已通过：

- 数据下载、MD5 和 SHA-256；
- 全量 Schema 与事件审计；
- 固定标签 seed 的 24 pairs/48 个半合成账号劫持事件；
- 标签距离与画像距离独立性审计；
- 用户筛选和用户日特征；
- 画像先验与 `FLAT_PROFILE_PRIOR` 判定；
- A3/A5a/A5b 真实 10-seed 覆盖门禁；
- 一个真实用户子集的软权重训练 step。

尚未通过：

- 完整 A0-matched/A1-R0/A1-Cov 训练入口；
- 训练期内部合成异常选参；
- CLUE-LDS 2-seed pilot；
- A3/A4 外部模型矩阵；
- 最终外部确认性统计。

当前状态：

```text
CLUE_LDS_core_data_and_phase0_passed_full_external_matrix_pending
```

这不是“外部有效性已成立”，也不是“B 类结论已完成”。
