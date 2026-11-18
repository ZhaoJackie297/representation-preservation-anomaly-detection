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
