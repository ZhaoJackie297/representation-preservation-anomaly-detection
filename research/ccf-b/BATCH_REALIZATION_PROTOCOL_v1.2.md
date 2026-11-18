# 图边落批协议 v1.2.1

日期：2026-11-18。

状态：正式 A3/A4 主实验的前置门禁。未通过本协议的实现检查，不启动确认性矩阵。

## 1. 问题

现有 TS2Vec/SoftCLT 训练单元为：

```text
每用户一条时间序列
```

`DataLoader(batch_size=16, shuffle=True)` 每次只取 16 个用户。SoftCLT 代码通过：

```python
soft_labels_batch = soft_labels[idx][:, idx]
```

只使用当前 16 个用户之间的软权重。全局图中的一条边 `(u,v)` 如果不与两个端点同批出现，就不会进入该步梯度。

## 2. 现有随机批次的实际覆盖率

任一给定用户对同批出现的概率近似为：

```text
p_pair = (batch_size / N_users) *
         ((batch_size - 1) / (N_users - 1))
```

在 `batch_size=16` 下：

| 数据 | B 步数 | 单边至少出现一次 | 2B | 3B |
| --- | ---: | ---: | ---: | ---: |
| r4.2，1,000 用户 | 620 | 13.8% | 25.8% | 36.0% |
| r5.2，2,000 用户 | 1,250 | 7.2% | 13.9% | 20.2% |
| r6.2，4,000 用户 | 2,500 | 3.7% | 7.2% | 10.6% |

因此原计划中“每行 top-m、邻居总质量 1”的图结构在随机批训练下大部分不会实现。若不修正，A3/A4 的阴性结果首先说明的是图没有进入梯度，而不是假设被否证。

## 3. 解决方案：共同覆盖批次表

正式主比较采用 `CommonCoverageBatchSampler`。它不是新的模型组件，而是把已经定义的图边可靠地送入 TS2Vec 同一实例对比损失。

### 3.1 目标边集合

对每个 seed、训练阶段 `s` 和主方法 A3/A4：

1. 计算该阶段开始时的支撑集 `S_s(u)`；
2. 形成所有非零软边：

```text
E_A3 = {(u,v): v in S_A3(u), W_A3(u,v) > 0}
E_A5b = {(u,v): v in S_A5b(u), W_A5b(u,v) > 0}
E_cov = E_A3 union E_A5b
```

3. 按 `sha256(seed, stage, min(u,v), max(u,v))` 排序，消除集合和哈希随机化差异；
4. 正式主网格先限制 `m in {5,10}`；`m=20/50` 只有在覆盖与算力审计通过后才作为敏感性分析。

参考 A3 边和 A5b 置换边必须进入同一张共同批次表。只覆盖 A3 会让 A5b 的“无差异”无法解释。

### 3.2 批次打包

每个 stage 的固定批次表按以下确定性规则生成：

1. 从尚未覆盖、软质量最大的边开始；
2. 将两个端点放入当前 batch；
3. 依次加入能覆盖最多未覆盖目标边的相邻用户；
4. batch 达到 16 个用户或没有可增加目标边的用户时停止；
5. 不足 16 个用户时，用固定 seed 的随机非重复用户补齐；
6. 标记该 batch 已覆盖的目标边；
7. 重复，直到 `E_A3` 和 `E_A5b` 中每条目标边都至少进入一个 batch；
8. 若生成批数少于最低训练批数，按软质量加权的已覆盖批表重复；
9. 最终顺序由固定 seed 决定，不再 shuffle。

### 3.3 对照共享

- A1、A1-sparse、A1-sparse-EMA、A1+、A2、A3、A4、A3-noAnchor、A3-noEMA、A3-product、A3-hard、A5a、A5b、A7、A8a、A8b、A9 在同一 seed/stage 下重放完全相同的用户批次表；
- 各变体只改变自己的权重、支撑集或训练目标；
- 新增 `A0-matched`：相同覆盖批次表 + 原始硬实例对比，用于区分“批次覆盖”与“软图权重”；
- 保留 `A0-official`：原版随机 shuffle，作为外部参照，不参与主因果对照；
- 每个 batch 保存用户索引、已实现目标边集合、真实非零权重和 `active_pair_mass`。

每个正式 seed 必须生成自己的批次表，不允许 10 个 seed 共用一个全局表。即使图边相同，`sha256(seed, ...)` 的排序、覆盖起点和补齐顺序也必须给出生效的 seed 差异。

### 3.4 动态训练预算

覆盖表优先于固定 epoch 数：

```text
B_stage = max(B_base, number_of_required_coverage_batches)
```

其中：

- `B_base` 是原 10 epochs 的优化步数；
- A3 使用两阶段共同覆盖表，A4 使用三阶段；
- A1+ 和 A9 重放相同阶段批次并按总步数匹配；
- 若覆盖导致预算显著增加，先缩小 `m` 或减少重复轮数，不通过偷删目标边降低覆盖率。

实际 `B_stage` 写入 `COMPUTE_BUDGET.md` 和 run registry 后才能正式开跑。

## 4. 必须记录的指标

每个 stage：

```text
intended_support_edges
realized_support_edges
coverage_A3
coverage_A5b
active_pair_mass_A3
active_pair_mass_A5b
median_user_edge_exposures
min_user_edge_exposures
fraction_users_with_zero_realized_edge
teacher_student_cosine_distance
batch_table_sha256
seed_pair_jaccard
```

正式门禁：

- `coverage_A3 >= 0.80`；
- `coverage_A5b >= 0.80`；
- `fraction_users_with_zero_realized_edge = 0`；
- 每个用户的中位边曝光次数 `>=1`；
- 10 个正式 seed 的 `batch_table_sha256` 全部唯一；
- seed 间用户对 Jaccard 不得全部为 1；
- 所有权重总质量与 A1/A1+ 对照可核对；
- 未达到时，状态为 `IMPLEMENTATION_BLOCKED`，不得解释为方法失败。

## 5. 为什么需要 `A0-matched`

共同覆盖批次会改变用户共现分布，即使不使用软图权重，也可能改变硬对比学习。

因此：

- `A0-official` 回答原版 TS2Vec 在其原始训练分布下的表现；
- `A0-matched` 回答共同批次本身带来的变化；
- A3/A4 相对 A0-matched 的差异才更接近软图权重和信任域更新的作用；
- A4 相对 A1+、A9 仍为主算力/机制比较。

## 6. 与因果归因的关系

共同批次表来自参考方法 A3/A4 的目标边集合，不读取任何真实异常标签。它可能使不同变体看到同一组用户对，因此实验估计的是：

> 在相同用户对暴露机会下，不同关系权重与图更新规则是否产生差异。

这是一种匹配刺激设计。它减少“边根本没有出现”的实现混杂，但不能消除所有选择效应，所以仍需：

- A5b 错误画像锚；
- A5a 随机图；
- A3-hard 总质量受控；
- 有效秩、邻域一致率和 PairMass 监控；
- 失败实验完整报告。

## 7. 当前状态

本协议已写入 v1.2.1，但尚未实现和跑通。

必须先交付：

1. `coverage_sampler.py`；
2. 单元测试：A3∪A5b 边覆盖、用户不重复、同 seed 可复现、不同 seed 表不同、共同批次表一致；
3. r4.2 单 seed 的 smoke run；
4. pair coverage 报告；
5. 实际 `B_stage` 和 GPU 小时。

只有以上五项通过，才能把 A3/A4 从“候选实现”升级为“可进入主实验”。
