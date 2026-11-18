# CCF-B 算力预算

版本：v1.2.4

日期：2026-11-18。

本预算只用于判断八周是否可执行，不替代 run registry 的真实墙钟和 GPU 小时。

## 1. 已有实测

R0 单次 `10 epochs` 的实测耗时：

| 数据 | 用户数 | 每 epoch 批数 | B 步数 | 单次耗时 |
| --- | ---: | ---: | ---: | ---: |
| CERT r4.2 | 1,000 | 62 | 620 | 149–194 s |
| CERT r5.2 | 2,000 | 125 | 1,250 | 168–411 s |
| CERT r6.2 | 4,000 | 250 | 2,500 | 145–261 s |

批数来自官方 TS2Vec：

```text
floor(n_users / batch_size), batch_size=16, drop_last=True
```

耗时包含当前训练环境和 R0 配置，不包含图边覆盖采样、评估、调参或基线适配的额外开销。

## 2. 正式变体的预算倍数

以一次 R0 `B=10 epochs` 为 `1B`：

| 变体 | 预算 |
| --- | ---: |
| A0-official | 1B |
| A0-matched | 1B |
| A1-R0 | 1B |
| A1-Cov | 1B |
| A1-sparse | 1B |
| A1-sparse-EMA | 1B |
| A1-repSupport-P | 2B |
| A1+（对 A3） | 2B |
| A1+（对 A4） | 3B |
| A2 | 2B |
| A3 | 2B |
| A4 | 3B |
| A3-noAnchor | 2B |
| A3-noEMA | 2B |
| A3-epsInf | 2B |
| A3-product | 2B |
| A3-hard | 2B |
| A5a | 2B |
| A5b | 2B |
| A7 | 1B |
| A8a | 1B |
| A8b | 1B |
| A9 | 3B |

单 seed 全部正式变体合计约：

```text
40 B
```

### Phase 0 实测覆盖倍率

在 1,000 用户、三张约 6,300 边合成图、8+8 槽位和 80% 覆盖率下，stage 需要：

```text
1,134–1,146 batches
```

相对 r4.2 原 B=620：

```text
1.83–1.85x
```

因此当前预算表只能作为 `B_base` 名义倍数。覆盖后应先写成：

```text
A1-Cov  ~= 1.83–1.85 B_base
A3      ~= 3.66–3.70 B_base
A4      ~= 5.49–5.55 B_base
```

真实 A3/A5a/A5b 图、m 和 stage 数不同，最终必须用真实覆盖表重算。合成倍率不是正式预算常量。

## 3. 10-seed 主矩阵粗估

使用实测单次耗时的中值：

- r4.2：约 171.5 s / B；
- r5.2：约 289.5 s / B。

| 数据 | 10-seed 训练粗估 | 覆盖采样前 |
| --- | ---: | --- |
| r4.2 | `40 × 10 × 171.5 s ≈ 19.1 GPU·h` | 不加调参 |
| r5.2 | `40 × 10 × 289.5 s ≈ 32.2 GPU·h` | 不加调参 |
| 合计 | 约 51.2 GPU·h | 未乘覆盖倍率的名义下限 |

Phase 0 合成倍率约 1.83–1.85 倍，且只作用于覆盖版变体。按总变体约 1.8 倍粗估：

```text
10-seed 主矩阵约 92–95 GPU·h
```

若真实图需要 2 倍以上 stage 预算，必须先缩小 m 或改进覆盖算法。

## 4. 调参预算

每方法族最多 20 次训练：

- 12 个候选各 1 个 seed；
- 前 4 名各补 2 个 seed；
- 总计 20 次。

按 6 个主要方法族估算：

```text
6 × 20 = 120 B-equivalent
```

在 r4.2 上：

```text
120 × 171.5 s ≈ 5.7 GPU·h
```

考虑 AdaGraph 候选的 2B/3B 和覆盖采样，按 2–3 倍计：

```text
约 12–18 GPU·h
```

放宽到 40 次的调参敏感性分析只对排名靠前的 3–4 个方法执行，预计另需：

```text
约 10–20 GPU·h
```

## 5. 深度基线和安全实验

当前未取得 OmniAnomaly、DCdetector、Anomaly Transformer、TimesNet、TS-TCC 等在该机器上的实测时间，不能精确记账。

保守预留：

| 项目 | 预留 |
| --- | ---: |
| 深度基线适配与 10-seed 主表 | 30–60 GPU·h |
| S×A 交叉和评分端重复 | 10–20 GPU·h |
| 安全扰动、low-and-slow、案例复算 | 10–20 GPU·h |
| 失败重跑、环境差异和缓冲 | 20–30 GPU·h |

## 6. 总量判断

在覆盖采样 1.5–2 倍、6 个调参族和全部预留下：

```text
约 240–450 GPU·h
```

若深度基线和外部数据预处理超出预期，可能达到：

```text
约 400–600 GPU·h
```

单张 RTX 5060 八周 24 小时理论上限：

```text
8 × 7 × 24 = 1,344 GPU·h
```

纯训练算力有充足余量，真正风险是：

- 外部数据访问和预处理；
- 覆盖采样实现和调试；
- 深度基线官方代码适配；
- 数据泄漏与统计协议检查；
- 失败重跑和人工分析。

## 7. 优先级与削减顺序

必须保留：

- A0-official、A0-matched；
- A1-R0、A1-Cov、A1-sparse-EMA、A1-repSupport-P；
- A1+ 和 A9；
- A3、A4；
- A3-noAnchor、A3-noEMA；
- A5a、A5b；
- A7、A8b；
- 主终点和统计协议；
- 至少一个独立外部数据门禁。

优先削减：

1. `m=20/50` 全矩阵；
2. A6 画像中位数消融；
3. A8a 的扩展 seed；
4. 重复的 S3–S5 组合；
5. 额外深度基线；
6. 非主线的运营成本敏感性。

不得通过削减以下内容来“省算力”：

- 等算力 A1+/A9；
- 共同批次对照；
- 置信区间；
- 失败实验；
- r5.2 的已见标记；
- 外部数据门禁。

## 8. 开跑前必须补齐

每个正式 run 的 registry 字段：

```text
run_id
data_hash
split_hash
seed
variant
stage
B_stage
support_edge_coverage
active_pair_mass
teacher_student_cosine_distance
gpu_name
gpu_hours
wall_seconds
peak_memory_mb
status
failure_reason
```

若真实累计 GPU 小时超过 300，先停止探索性矩阵并检查数据门禁，不削减主对照。
