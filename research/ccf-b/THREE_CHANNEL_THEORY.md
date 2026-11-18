# SoftCLT 三通道理论审计

日期：2026-11-18

## 0. 结论

三通道分解在代数上成立，而且比原来的“质量/关系”二分更适合解释 v5：

```text
尺度通道 W_a
质量通道 π_a
关系通道 q_a
```

但它是分析工具，不是新的基础算法定理。soft-target InfoNCE 已有类似梯度分解的
风险，论文中应把它写成“我们采用并实现了可检验的三通道分解”，不能单独当作
算法创新。

## 1. 官方代码的实际映射

`models/timelags.py::dup_matrix` 对每个锚点生成两个长度 `2B-1` 的权重向量：

```text
K = 2B-2 个非对角候选
1 个对角正样本
候选总数 = 2B-1
```

因此 B=16 时：

```text
K = 30 个非对角候选
```

不是 `B-1`，也不是把同一个非对角权重只计一次。

## 2. 精确恒等式

对锚点 `a`，记正样本权重 `w_{a,+}=1`，非对角权重 `m_a`：

```text
M_a = sum_j m_{a,j}
W_a = 1 + M_a
pi_a = M_a / W_a
q_a = m_a / M_a
t_a = (1 - pi_a) e_+ + pi_a q_a
```

官方实例损失满足：

```text
l_a = -sum_j w_{a,j} log p_{a,j}
    = W_a H(t_a, p_a)
```

梯度：

```text
partial l_a / partial s_{a,j} = W_a p_{a,j} - w_{a,j}
```

与 hard identity 权重的差：

```text
g_soft - g_hard
  = M_a (p_a - q_a)
  = M_a (p_a - u) + M_a (u - q_a)
```

其中 `u` 是 `K=2B-2` 个非对角候选上的均匀分布。

## 3. 三通道解释

| 通道 | 量 | 解释 |
| --- | --- | --- |
| 尺度 | `W_a` | 锚点级梯度倍率 |
| 质量 | `pi_a` | 非对角目标总质量，类似 label smoothing |
| 关系 | `q_a` | 非对角质量如何分配 |

关系通道的 Pinsker 上界：

```text
|| M_a (u - q_a) ||_1
  <= M_a sqrt(2 KL(q_a || u))

KL(q_a || u) = log K - H(q_a)
```

这正是已有 `P_m` 行熵审计的对应量，不需要新增训练数据。

## 4. 数值验证

运行：

```powershell
python three_channel_check.py --output three_channel_check.json
```

2000 次随机试验结果：

```text
identity max error       = 4.26e-14
gradient max error       = 3.10e-08
soft-hard difference max = 8.88e-16
Pinsker violations       = 0
free-logit max error     = 1.68e-14
```

这些结果支持：

- 恒等式和梯度分解成立；
- Pinsker 关系通道上界成立；
- 自由 logit 训练可以恢复 `log w` 的相对顺序，但在真实有限样本上只能看
  斜率和 `R^2`，不能要求精确相等。

在 CLUE-LDS 现有的两个 A0-official 16-user batch 上：

```text
W_a mean range       = 23.32–24.19
pi_a mean range      = 0.9565–0.9578
q entropy mean range = 3.3805–3.3850
KL(q||u) mean range  = 0.0162–0.0207
```

这进一步说明：当前 v5 的 batch 内目标远未归一化；归一化后约 95.7% 的目标质量
分配给非对角候选，只有约 4.3% 留给正样本。虽然关系通道拥有较大的总质量，但
`KL(q||u)` 仅约 0.018，说明 q 的分布形状仍接近均匀，方向性关系信息较弱。后续
若不先固定 `W_a` 和 `pi_a`，不能把结果直接归因于关系对齐。

## 5. 对平坦先验的预测

若批内权重行接近均匀：

```text
KL(q_a || u) -> 0
关系通道 -> 0
```

此时 SoftCLT 主要退化为：

```text
带锚点级 W_a 重加权的均匀质量目标
```

因此 CERT/CLUE 画像先验平坦时，T3/T4 的收益不能自动解释为关系通道有效；
它可能主要来自质量通道和尺度通道。这是待实验检验的预测，不是已确认结论。

## 6. 由三通道推出的控制矩阵

| 对照 | 权重设定 | 隔离的通道 |
| --- | --- | --- |
| A1-Cov-rownorm | 保持 `pi_a,q_a`，令 `W_a=1` | 尺度 |
| A1-Cov-uniform | 保持 `W_a,pi_a`，令 `q_a=u` | 关系 |
| A1-Cov-permq | 保持 `W_a,pi_a`，随机置换批内 `q_a` | 关系对齐 |
| A0-A3batch | 同批次，hard identity | 批次组成 |

`A1-Cov-uniform` 必须逐锚点匹配，而不是只匹配全局平均。

## 7. 预注册预测

- P1：平坦先验下，A1-Cov 应接近逐锚点匹配的 Uniform-soft；
- P2：raw soft 与 rownorm 的差异可归因于尺度通道；
- P3：若关系信息被编码，已训练模型中非对角 logit 差与 `log w` 回归的斜率
  应明显大于 0；若接近 0，则关系通道没有成为主要表示驱动力。

这些预测不能替代 A1/A3/A4 性能实验。
