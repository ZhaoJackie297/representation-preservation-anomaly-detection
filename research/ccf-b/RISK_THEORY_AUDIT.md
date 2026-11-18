# EWMA/CUSUM 风险积累审计

日期：2026-11-18

## 定义

对训练期 median/MAD 标准化后的分数 `s_t`：

```text
EWMA: Z_t = alpha*s_t + (1-alpha)*Z_{t-1}
CUSUM: C_t = max(0, C_{t-1} + s_t - k)
```

## 均值漂移近似

在标准正态、独立同分布、变化后均值漂移为 `delta` 的假设下：

- EWMA 均值从 0 开始，变化后的期望为

  ```text
  E[Z_t] = delta * (1 - (1-alpha)^t)
  ```

  若阈值 `h < delta`，均值穿越时间的近似为

  ```text
  t ≈ log(1 - h/delta) / log(1-alpha)
  ```

- CUSUM 的平均漂移近似为 `delta-k`，因此

  ```text
  delay ≈ h / (delta-k)
  ```

这些是均值漂移近似，不是真实分数上的精确定理。真实异常分数重尾且自相关，
必须用训练期分数和回放事件重新估计延迟和误报。

## 模拟协议

`risk_control_check.py` 使用标准正态正常期、变化后均值漂移 `delta=1.0`、
变化点第 20 天、序列长度 100、2000 次重复，并以“每条正常序列至少一次告警”
概率 1% 校准阈值。

## 结果

| 方法 | 参数 | 实测序列 FPR | 平均延迟 |
| --- | ---: | ---: | ---: |
| EWMA | alpha=0.05 | 0.0075 | 15.62 天 |
| EWMA | alpha=0.10 | 0.0045 | 15.43 天 |
| EWMA | alpha=0.20 | 0.0080 | 17.69 天 |
| EWMA | alpha=0.50 | 0.0125 | 46.48 天 |
| CUSUM | k=0.25 | 0.0040 | 18.69 天 |
| CUSUM | k=0.50 | 0.0010 | 15.67 天 |
| CUSUM | k=1.00 | 0.0060 | 26.79 天 |

在该标准正态模拟下，EWMA `alpha=0.05–0.10` 和 CUSUM `k=0.5` 的
延迟/误报折中较合理。这个结论不能直接搬到 CERT/CLUE；真实分数需要重新校准。

## 项目使用边界

- 风险参数只能在训练期正常分数上选择；
- CERT 用 `first_observed_date`，CLUE 用 pair `change_date` 定义提前量；
- r6.2 只作五事件描述性审计；
- 图/邻居传播只用训练期图；
- 风险层若只提高单一数据集指标，必须标为探索性。

脚本和测试：

```text
research/ccf-b/risk_control_check.py
research/ccf-b/risk_control_check.json
research/ccf-b/test_risk_control_check.py
```
