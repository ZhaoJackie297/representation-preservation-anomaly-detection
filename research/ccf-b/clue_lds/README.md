# CLUE-LDS 数据门禁

数据：

- DOI：`10.5281/zenodo.7119953`
- Zenodo 文件：`clue.zip`
- 大小：635,105,552 bytes
- 官方 MD5：`9e318370f96b68077667e9cdc05f26a5`
- 解压后约 14.5 GB

## 当前步骤

1. 下载 `clue.zip`；
2. 验证 MD5；
3. 解压得到 `clue.json`；
4. 流式审计 JSONL schema、用户、日期、事件类型；
5. 冻结账号劫持切换协议；
6. 构建用户日特征和组/分享用户图。

## 流式审计

```powershell
python audit_clue.py `
  --input E:\MarketRisker\work\cert\external\clue_lds\clue.json `
  --output E:\MarketRisker\work\cert\external\clue_lds\audit.json `
  --user-stats-output E:\MarketRisker\work\cert\external\clue_lds\user_stats.json
```

先使用 `--max-events 10000` 做 schema smoke，再跑全量。

## 单元测试

```powershell
python -m unittest -v `
  test_audit_clue.py `
  test_generate_hijack_labels.py `
  test_build_user_days.py `
  test_power_pair_clusters.py `
  test_audit_profile_prior_information.py `
  test_audit_label_prior_independence.py
```

## 固定 seed 半合成账号劫持标签

```powershell
python generate_hijack_labels.py `
  --user-stats E:\MarketRisker\work\cert\external\clue_lds\user_stats.json `
  --output E:\MarketRisker\work\cert\external\clue_lds\hijack_labels_24pairs_20261101.csv `
  --seed 20261101 `
  --pairs 24 `
  --days-before 35 `
  --days-after 30 `
  --profile-cutoff 2022-02-28 `
  --min-profile-days 20 `
  --start-date 2022-03-01 `
  --end-date 2022-08-30
```

“固定 seed”指本项目对官方切换算法实现的采样 seed，不是官方异常标签 seed。

## 用户日特征

```powershell
python build_user_days.py `
  --input E:\MarketRisker\work\cert\external\clue_lds\clue.json `
  --audit E:\MarketRisker\work\cert\external\clue_lds\audit.json `
  --user-stats E:\MarketRisker\work\cert\external\clue_lds\user_stats.json `
  --labels E:\MarketRisker\work\cert\external\clue_lds\hijack_labels_24pairs_20261101.csv `
  --output-dir E:\MarketRisker\work\cert\external\clue_lds\processed_final `
  --train-end 2022-02-28 `
  --eval-end 2022-09-29 `
  --max-users 1000
```

## 状态

- 数据已下载并通过官方 MD5；
- 50,522,931 条事件全量审计完成；
- 固定标签 seed=`20261101` 的 24 pairs / 48 事件完成；
- 838 用户、1,911 日期槽、52 维用户日数据完成；
- 画像先验和组/分享辅助图完成；
- 标签/画像距离独立性审计通过，Spearman=-0.1991；
- top-10 画像先验触发 `FLAT_PROFILE_PRIOR`；
- 真实 A3/A5a/A5b 10-seed 覆盖通过；
- SoftTS2Vec 单步训练接线通过；
- 完整 stage/EMA 训练和检测评价尚未完成。

详细结果见 `../CLUE_LDS_PHASE0_RESULTS.md`。
