# 申报书技术口径说明（数据与实验）

日期：2026-11-20  
用途：锁定主线版本、模块状态、摘要与创新点边界

## 主线版本

申报书以 `paper/manuscript_v6.docx` 为统一文本，方法定义沿用 v5：

```text
27 维用户日序列
→ TS2Vec 自监督编码
→ instance-soft / Coverage-aware soft instance
→ 47 维训练期画像关系
→ mask residual + 欧氏表征库
→ 每日 Top-K、7 天冷却
→ 证据报告
```

v6 新增的是证据，不是另起方法：CERT 固定 10-seed、CLUE-LDS 外部半合成复验、
受控异常形态分析。`paper/manuscript.docx` 与 `paper/manuscript_v6.docx` 是
同一版 v6 内容；v5/R0 保留为 `paper/manuscript_v5.docx`。LCTC、AdaGraph
和双分数均值都不写成已完成的主创新。

## 三项必须确认的事实

| 问题 | 结论 | 证据 |
| --- | --- | --- |
| 间隔编码/LCTC 有独立结果吗 | **没有** | 仅有数学定义、实验设计和 `lctc_math_sanity.js`；没有训练或基准输出 |
| 双分数有独立结果吗 | **有旧版事件级预研，但不是主线** | `ts2vec_dual_score_*` 覆盖 r4.2 original/retrained、r5.2；保存固定 K、utility、McNemar，无 PR/ROC 主表 |
| iForest 是基线还是队列组件 | **v6 主方案中是外部基线** | E11 只用 27 维用户日特征、训练期拟合；旧 `quota_relationship_sim.py` 才把 iForest 当队列补位组件 |

双分数的关键数字：r4.2 original test K=10，双分数组合 `14/24`，iForest 和
mask-only 均为 `15/24`；r5.2 external K=10，双分数 `43/99`、iForest `35/99`，
但 McNemar 原始 p≈`0.096`，Holm 后不显著。因此只能算待验证候选。

## 23 维与 27 维

| 口径 | 组成 | 对应实验 |
| --- | --- | --- |
| 23 维 | 20 个计数特征 + 3 个时间特征 | 47 维画像：23 维中位数 + 23 维 q90 + 活跃比例；对应 E2 T3/T4 软实例权重和画像稳定性 |
| 27 维 | 23 维 + `has_any_event` + `is_weekend` + `weekday_sin/cos` | 当前 TS2Vec、mask 分数、E11 iForest、CERT 固定 10-seed |

所以旧图的 23 维只能解释画像或早期行为特征实验；v5/v6 的 27 维才是当前模型
输入。两组图不能混成一组消融，也不能直接横向比数值。

## 模块状态

| 模块 | 状态 |
| --- | --- |
| S1 数据处理 | 已跑通 |
| TS2Vec / T0–T4 | 已跑通 |
| instance-soft / 47 维画像 | 已跑通 |
| Coverage-aware training | 已跑通；CERT 10-seed 正向，CLUE 外部未复现 |
| 欧氏表征库 / PAI-A1 | 已跑通 |
| 固定 K 与冷却期评价 | 已跑通 |
| CLUE-LDS 外部复验 | 已跑通；半合成账号劫持，有限功效 |
| iForest | 已跑通；仅基线 |
| S3 风险积累 | 部分验证；EWMA/CUSUM 未达到运营门槛 |
| S4 证据卡 | 有旧版输出；与当前 T4 全链路联调待完成 |
| 双分数 | 旧版事件级预研，未进入主线 |
| LCTC / 间隔编码 | 未实现、未验证 |
| 前端复核界面 | 待实现 |

## 摘要和创新点写法

可以写：

1. instance-soft 与训练期画像缓解硬实例对比的正常类碰撞；
2. 欧氏表征库保留行为幅度，PAI 作为辅助评分；
3. CERT r4.2/r5.2 完成固定 10-seed 同协议验证；
4. CLUE-LDS 完成有限外部分支验证，PAI 正向、Coverage 未复现；
5. 固定预算 Top-K、冷却期和证据卡构成应用层原型。

不能写：

- 间隔编码/LCTC 已经有效；
- 双分数已经独立优于 iForest 或 mask-only；
- 完整双端方法已经外部确认；
- iForest 是 v6 主队列组件；
- 23 维与 27 维结果可以直接横向比较。

结论：申报书主线锁定 v6，但方法核心仍是 v5 的 instance-soft + 欧氏/PAI +
固定预算。当前可支撑主线的证据是 CERT 10-seed、CLUE-LDS 的 PAI 外部支持和
iForest 外部基线；LCTC 和双分数不能充当已完成的核心创新。
