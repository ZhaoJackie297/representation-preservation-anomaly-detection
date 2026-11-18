# SoftCLT 实例软权重审计

日期：2026-11-18

## 结论

官方 `softclt_ts2vec/models/soft_losses.py` 的 `inst_CL_soft` 使用
`soft_labels_L/R` 对全部非自身 logits 做加权求和，但没有对每个 batch 的每行
软权重做归一化。`hier_CL_soft` 外层只除以层级数 `d`，也没有按行总权重归一化。

项目 v5 使用的 `densify` 产生：

```text
对角权重 = 1.0
非对角权重 = 2*alpha / (1 + exp(-tau*x)) <= 2*alpha
```

在 `alpha=0.5` 时，非对角权重通常不超过 0.5，因此 batch=16 的一行通常有

```text
diagonal mass = 1
off-diagonal mass <= 7.5
row sum <= 8.5
```

所以原始软权重不是概率，也不是单位质量的目标分布。将整个矩阵乘以常数会按同样
比例改变实例损失和梯度；对不同行做归一化则会改变不同样本的相对贡献。论文中
不能把“软权重”直接解释为归一化目标概率。

## 数值审计

运行：

```powershell
python audit_softclt_instance_weights.py
```

脚本检查：

- 行总权重的范围、均值和 off-diagonal fraction；
- off-diagonal 目标熵；
- off-diagonal 相对均匀分布的逐行 KL；
- 原始权重、逐行归一化权重、identity 权重的官方实例损失；
- 权重乘 2 是否线性地使损失乘 2。

## 已跑出的 toy 数值

以 `B=16`、对角 `1.0`、非对角 `0.5` 的矩阵：

```text
row sum mean       = 8.5
off-diagonal mass  = 0.88235
off-diagonal entropy mean = 2.70805
KL(off-diagonal || uniform) = 0.0

official loss(raw)       = 86.86944
official loss(row-norm)  = 10.21993
official loss(identity)  = 5.38365
official loss(2x weights)= 173.73889
```

这直接说明权重尺度不是可忽略的实现细节：原始权重、逐行归一化和 hard identity
在本玩具输入上分别给出不同的优化目标。

在 CLUE canonical 838 用户上取现有 2 个 A0-official 随机 batch 的 16 用户子矩阵，
得到更接近真实数据的描述统计：

```text
batch row_sum mean range = 12.1597–12.5949
diagonal mass            = 16 per batch
off-diagonal fraction    = 0.9178–0.9206
off-diagonal entropy     = 2.6874–2.6919
off-diagonal KL(uniform)= 0.0162–0.0207
```

这不是 A3 覆盖表的最终统计；它只证明 v5/CLUE 的软权重在真实 batch 上确实
不是单位质量目标，且关系质量占主导。

在 r4.2 的 32 用户、5 个覆盖 batch 的接线 smoke 中，`A1-Cov-rownorm`
的损失从 `62.60 -> 2.90` 量级变化明显，而 `uniform/permuq` 仍保持原始尺度。
这只验证尺度控制生效，不是性能结果。

脚本和测试：

```text
research/ccf-b/audit_softclt_instance_weights.py
research/ccf-b/test_audit_softclt_instance_weights.py
```

## 对论文的影响

当前不能声称：

- “软标签已经是归一化概率目标”；
- “对角/非对角质量天然平衡”；
- “软实例损失只改变负样本关系而不改变损失尺度”。

后续必须把两个通道显式分开报告：

```text
quality/self channel: 对角权重或正样本质量
relation channel:     非对角权重或关系邻居质量
```

建议的实验对照是同一批次表下：

1. identity/hard 权重；
2. 原始 densify 权重；
3. 原始权重逐行归一化；
4. 固定 self/relation 质量比的权重；
5. Uniform-soft 权重。

## 适用条件

这不是“官方实现错误”的断言；官方实现可能有意使用不同尺度的软目标。它确实意味着
v5 的解释必须把权重尺度、正样本质量和关系质量分开。只要论文把这一点写成可检验的
设计选择，而不是隐藏的归一化假设，升级路线仍然成立。

进一步的三通道公式、`K=2B-2` 的精确候选数和 2000 次数值验证见
`THREE_CHANNEL_THEORY.md`。
