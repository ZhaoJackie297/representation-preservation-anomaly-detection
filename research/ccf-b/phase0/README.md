# Phase 0：CommonCoverageBatchSampler

当前实现：

- 联合 A3/A5a/A5b 目标边；
- 每 batch 固定 8 个覆盖设计槽位和 8 个随机填充槽位；
- 三个图分别记录覆盖率；
- 支持 seed/stage 依赖的批次表；
- 记录用户曝光、边曝光、批次哈希和训练表哈希；
- 对零曝光用户、覆盖率不足和预算耗尽设置硬门禁。

## 单元测试

```powershell
python -m unittest -v test_common_coverage_sampler.py test_coverage_soft_ts2vec.py
```

`CoverageSoftTS2Vec` 使用显式共同批次表、真实 `active_pair_mass` 日志和逐阶段优化器重置；已通过 CPU 单步训练测试。

## CLUE 双端训练入口

`dual_end_entry.py` 提供首批入口：

```text
A0-official
A0-matched
A1-R0
A1-Cov
```

它从 `clue_lds/processed_final` 读取用户日数据、画像特征、A3/A5a/A5b 图和用户顺序，生成与 seed 绑定的共同覆盖批次表，并保存每个变体的模型、batch table、摘要和可选分数。

正式运行前必须先完成 `--encode` 的分数 smoke 和训练期内部合成异常验证。当前入口只支持首个训练阶段；A3/A4 的两阶段更新、EMA 教师和 trust-region 更新尚未接入。

实现 smoke 可用较小的 `--base-steps` 只验证显式批次表接线；不能用该 smoke 解释覆盖率或检测性能。

示例：

```powershell
python dual_end_entry.py `
  --data-dir E:\MarketRisker\work\cert\external\clue_lds\processed_final `
  --output-dir E:\MarketRisker\work\cert\external\clue_lds\dual_end_pilot `
  --seeds 20261116 20261117 `
  --variants A0-matched A1-R0 A1-Cov `
  --base-steps 530 `
  --encode
```

每个 seed 的 A0-matched/A1-Cov 使用该 seed 实际生成并通过 80% 门禁的覆盖批次表；A1-R0 使用相同 optimizer steps 的随机 shuffle 表；A0-official 使用未覆盖的随机表。

## 合成 smoke

```powershell
python smoke_synthetic.py `
  --users 1000 `
  --degree 10 `
  --seeds 20261116 20261117 20261118 `
  --target-coverage 0.8 `
  --min-batches 620 `
  --max-batches 5000
```

已记录结果见 `../PHASE0_SMOKE_RESULTS.md`。

正式 smoke 已使用 10 个正式 seed：

```text
20261116–20261125
```

10 张批次表哈希全部不同。

## 限制

- 合成图不是实际画像图或表示图；
- 当前没有接入 TS2Vec `custom_dataset`；
- active pair mass 需要 trainer 根据真实 `W` 另行记录；
- 真实图通过 80% 门禁前，不能启动 A3/A4 主矩阵。
