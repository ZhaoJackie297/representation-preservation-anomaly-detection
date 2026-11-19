# Phase 1 受控异常与风险实验结果

日期：2026-11-19

## 1. 证据定位

本阶段使用 CERT r4.2 与 CLUE-LDS 的真实正常行为作为母体，
构造固定 seed 的点尖峰、持续偏移、缓慢漂移和账号切换事件。
数据 seed 与模型 seed 分离，模型没有在注入事件上训练或选参。

**这是受控机制实验，不是新的真实数据集，也不是外部验证。**

## 2. Temporal-soft 桥接

主比较为 `T4_full_soft - T3_instance_soft`，同一事件、同一强度、
同一评分方式。表中使用 point residual 分数。

| signal | strength | delta_pr | delta_k10 |
| --- | --- | --- | --- |
| drift | 0.500000 | 0.000083 | 0.013889 |
| drift | 1.000000 | 0.000589 | 0.000000 |
| drift | 2.000000 | 0.022430 | 0.000000 |
| point | 0.500000 | 0.000061 | -0.027778 |
| point | 1.000000 | 0.000727 | -0.013889 |
| point | 2.000000 | 0.067190 | -0.013889 |
| sustained | 0.500000 | 0.000237 | 0.027778 |
| sustained | 1.000000 | 0.003186 | 0.000000 |
| sustained | 2.000000 | 0.076616 | 0.000000 |
| switch | 0.500000 | 0.000145 | -0.000000 |
| switch | 1.000000 | 0.001088 | 0.000000 |
| switch | 2.000000 | 0.007740 | 0.041667 |

合计：12/12 个信号-强度设置的 PR-AUC 差值为正，10 个设置为 3/3 seeds 同向，平均差值 0.015008。K=10 只有 3 个设置为正，9 个为非正。

解释：T4 在持续偏移和缓慢漂移上的 PR-AUC 差值方向稳定为正，
但 K=10 事件召回没有同步提高。点尖峰上 PR-AUC 也可能提高，
固定预算事件召回却出现负向。因此只能支持“temporal-soft 改变
了持续信号的局部几何”，不能支持“temporal-soft 已有稳定
端到端固定预算收益”。

## 3. r4.2 Coverage 对照

| signal | strength | delta_pr | delta_k10 |
| --- | --- | --- | --- |
| drift | 0.500000 | 0.003041 | 0.083333 |
| drift | 1.000000 | 0.018679 | 0.000000 |
| drift | 2.000000 | 0.229757 | 0.000000 |
| point | 0.500000 | 0.001322 | 0.111111 |
| point | 1.000000 | 0.017116 | -0.027778 |
| point | 2.000000 | 0.567386 | -0.027778 |
| sustained | 0.500000 | 0.007674 | 0.097222 |
| sustained | 1.000000 | 0.077879 | 0.000000 |
| sustained | 2.000000 | 0.452947 | 0.000000 |
| switch | 0.500000 | 0.002718 | 0.041667 |
| switch | 1.000000 | 0.018767 | 0.097222 |
| switch | 2.000000 | 0.079642 | 0.083333 |

r4.2 受控点级 PR-AUC：12/12 个信号-强度设置为正，12 个设置为 3/3 seeds 同向，平均差值 0.123077。

这种近乎全正的受控结果，加上后文 iForest 的强表现，说明注入器更
偏向高幅度、单行可分的表格信号。它可用于识别边界，不能作为
A1-Cov 已证明优于官方训练的确认性证据。

## 4. CLUE 受控 Coverage 对照

| signal | strength | delta_pr | delta_k10 |
| --- | --- | --- | --- |
| drift | 0.500000 | 0.003417 | 0.000000 |
| drift | 1.000000 | 0.009011 | 0.000000 |
| drift | 2.000000 | 0.013761 | 0.000000 |
| point | 0.500000 | 0.003019 | 0.097222 |
| point | 1.000000 | 0.004631 | 0.055556 |
| point | 2.000000 | 0.007897 | 0.055556 |
| sustained | 0.500000 | 0.012007 | 0.027778 |
| sustained | 1.000000 | 0.016233 | 0.013889 |
| sustained | 2.000000 | 0.022795 | 0.013889 |
| switch | 0.500000 | 0.000460 | 0.013889 |
| switch | 1.000000 | 0.000303 | 0.013889 |
| switch | 2.000000 | 0.000136 | 0.013889 |

CLUE 受控点级 PR-AUC：12/12 个设置为正，9 个设置为 3/3 seeds 同向，平均差值 0.007806。

CLUE 受控数据上 A1-Cov 的点级指标多数提高。该方向与真实
CLUE-LDS 10-seed 的 Coverage 负结果相反，说明当前注入器主要
制造了高幅度、低歧义、tabular 可分的信号，不能据此推翻真实负结果。

## 5. 点级、上下文和组合信号

### 5.1 T4 temporal 模型

| signal_type | aggregation | metric | mean_delta | positive_seed_configs | n_seed_configs |
| --- | --- | --- | --- | --- | --- |
| drift | combined | event_recall_k10 | 0.000000 | 0 | 9 |
| drift | combined | pr_auc | 0.018592 | 9 | 9 |
| drift | context | event_recall_k10 | 0.000000 | 0 | 9 |
| drift | context | pr_auc | -0.004571 | 5 | 9 |
| point | combined | event_recall_k10 | -0.050926 | 2 | 9 |
| point | combined | pr_auc | -0.098224 | 0 | 9 |
| point | context | event_recall_k10 | -0.134259 | 0 | 9 |
| point | context | pr_auc | -0.115355 | 0 | 9 |
| sustained | combined | event_recall_k10 | 0.009259 | 2 | 9 |
| sustained | combined | pr_auc | 0.039881 | 9 | 9 |
| sustained | context | event_recall_k10 | 0.009259 | 2 | 9 |
| sustained | context | pr_auc | -0.005502 | 7 | 9 |
| switch | combined | event_recall_k10 | 0.023148 | 4 | 9 |
| switch | combined | pr_auc | 0.015719 | 9 | 9 |
| switch | context | event_recall_k10 | 0.046296 | 7 | 9 |
| switch | context | pr_auc | 0.017914 | 9 | 9 |

### 5.2 CLUE A1-Cov

| signal_type | aggregation | metric | mean_delta | positive_seed_configs | n_seed_configs |
| --- | --- | --- | --- | --- | --- |
| drift | combined | event_recall_k10 | 0.000000 | 0 | 9 |
| drift | combined | pr_auc | -0.003260 | 0 | 9 |
| drift | context | event_recall_k10 | 0.000000 | 0 | 9 |
| drift | context | pr_auc | -0.007107 | 0 | 9 |
| point | combined | event_recall_k10 | -0.027778 | 0 | 9 |
| point | combined | pr_auc | -0.005216 | 0 | 9 |
| point | context | event_recall_k10 | -0.041667 | 0 | 9 |
| point | context | pr_auc | -0.010008 | 0 | 9 |
| sustained | combined | event_recall_k10 | -0.027778 | 0 | 9 |
| sustained | combined | pr_auc | -0.011434 | 0 | 9 |
| sustained | context | event_recall_k10 | -0.027778 | 0 | 9 |
| sustained | context | pr_auc | -0.021141 | 0 | 9 |
| switch | combined | event_recall_k10 | -0.027778 | 0 | 9 |
| switch | combined | pr_auc | -0.000189 | 0 | 9 |
| switch | context | event_recall_k10 | -0.037037 | 0 | 9 |
| switch | context | pr_auc | 0.000058 | 3 | 9 |

结论：不存在一个对全部信号都最优的分数。点信号应由 point
残差主导；持续偏移和漂移的 PR-AUC 可由 context/combined 提高，
但对孤立点使用上下文平滑会降低排序和召回。

## 6. 统一 iForest 基线

| dataset | signal | iForest_pr | best_neural_pr | best_neural_method | iForest_k10 | best_neural_k10 |
| --- | --- | --- | --- | --- | --- | --- |
| CERT-CAS | drift | 0.229885 | 0.170992 | T4_full_soft | 1.000000 | 1.000000 |
| CERT-CAS | point | 0.320780 | 0.342022 | T4_full_soft | 0.333333 | 0.430556 |
| CERT-CAS | sustained | 0.576670 | 0.695699 | T4_full_soft | 1.000000 | 1.000000 |
| CERT-CAS | switch | 0.012207 | 0.041138 | T4_full_soft | 0.875000 | 0.930556 |
| CLUE-CAS | drift | 0.456720 | 0.063136 | A1-Cov | 0.875000 | 0.875000 |
| CLUE-CAS | point | 0.376222 | 0.034454 | A1-Cov | 0.416667 | 0.388889 |
| CLUE-CAS | sustained | 0.686718 | 0.128838 | A1-Cov | 0.847222 | 0.861111 |
| CLUE-CAS | switch | 0.034422 | 0.018019 | A1-Cov | 0.847222 | 0.861111 |

iForest 在受控的高幅度点尖峰和漂移上经常强于神经表征。
这直接说明注入器不能作为“深度方法优越”的证据；它是基线
敏感性审计，而不是新主结果。

## 7. S3 风险控制

| method | pr_auc_mean | roc_auc_mean | event_recall_3d_mean | median_contact_time_days | false_alerts_per_day_mean |
| --- | --- | --- | --- | --- | --- |
| cusum | 0.007974 | 0.851842 | 0.154167 | -2.650000 | 9.919697 |
| ewma | 0.011047 | 0.872125 | 0.345833 | -1.250000 | 9.836869 |
| ewma_propagation | 0.011135 | 0.868153 | 0.312500 | -2.400000 | 9.834848 |
| instant | 0.012839 | 0.876870 | 0.579167 | -3.000000 | 9.838384 |
| propagation | 0.012430 | 0.880699 | 0.579167 | -3.150000 | 9.829798 |

相对 instant 的配对结果：

| method | metric | mean_delta | positive_seeds | negative_seeds | n |
| --- | --- | --- | --- | --- | --- |
| cusum | roc_auc | -0.025028 | 0 | 10 | 10 |
| cusum | event_recall_3d | -0.425000 | 0 | 10 | 10 |
| cusum | median_contact_time_days | 0.350000 | 5 | 2 | 10 |
| cusum | false_alerts_per_day | 0.081313 | 10 | 0 | 10 |
| ewma | roc_auc | -0.004745 | 0 | 10 | 10 |
| ewma | event_recall_3d | -0.233333 | 0 | 10 | 10 |
| ewma | median_contact_time_days | 1.750000 | 9 | 0 | 10 |
| ewma | false_alerts_per_day | -0.001515 | 5 | 3 | 10 |
| ewma_propagation | roc_auc | -0.008717 | 0 | 10 | 10 |
| ewma_propagation | event_recall_3d | -0.266667 | 0 | 10 | 10 |
| ewma_propagation | median_contact_time_days | 0.600000 | 7 | 1 | 10 |
| ewma_propagation | false_alerts_per_day | -0.003535 | 5 | 5 | 10 |
| propagation | roc_auc | 0.003829 | 10 | 0 | 10 |
| propagation | event_recall_3d | 0.000000 | 3 | 3 | 10 |
| propagation | median_contact_time_days | -0.150000 | 0 | 3 | 10 |
| propagation | false_alerts_per_day | -0.008586 | 2 | 8 | 10 |

EWMA 和 CUSUM 明显损伤 3 天事件召回。单步传播仅带来很小的
ROC/Precision 变化，事件召回不提高，提前量仅约 0.15 天，
误报下降远低于 10%。S3 的运营收益门槛未通过。

## 8. S4 案例

固定规则生成的成功/失败案例位于：

`research/ccf-b/phase1/results/risk_cases_v1/S4_CASES_generated.md`

案例只作排序证据展示，不作攻击归因。

## 9. 冻结结论

1. temporal-soft 的局部几何作用可以保留，端到端固定预算收益未确认；
2. point/context 必须按异常形态分别报告，统一融合不成立；
3. 受控 A1-Cov 正向不能覆盖真实 CLUE 负结果；
4. iForest 在受控幅度信号上很强，必须在论文中如实报告；
5. EWMA/CUSUM 当前没有运营收益，风险章节只能保留理论和小幅
   propagation 排序结果。

## 10. 运行参数

```json
{
  "temporal": {
    "profile": "r42-temporal",
    "data": "E:\\MarketRisker\\work\\cert\\processed_r42\\r4.2_sequence_dataset.npz",
    "model_root": "E:\\MarketRisker\\work\\cert\\experiments\\representation_preservation\\e2_instance\\missing_safe_v2_20261117",
    "seeds": [
      20261116,
      20261117,
      20261118
    ],
    "label_seed": 20261119,
    "signal_types": [
      "point",
      "sustained",
      "drift",
      "switch"
    ],
    "strengths": [
      0.5,
      1.0,
      2.0
    ],
    "k_grid": [
      5,
      10
    ],
    "cooldown_days": 7,
    "context_window": 5,
    "background_users": 96,
    "event_users": 24,
    "channels": [
      "9",
      "2",
      "10",
      "1",
      "6",
      "24",
      "4",
      "7",
      "5",
      "14",
      "18",
      "0"
    ],
    "crop_start": 239,
    "eval_start": 303,
    "eval_end": 500,
    "train_end": 158,
    "claims": {
      "external_validation": false,
      "dataset_role": "controlled anomaly-signal benchmark built on real normal behavior; not an external dataset"
    },
    "selected_user_count": 96,
    "target_user_count": 24
  },
  "r42_coverage": {
    "profile": "r42-coverage",
    "data": "E:\\MarketRisker\\work\\cert\\processed_r42\\r4.2_sequence_dataset.npz",
    "model_root": "E:\\MarketRisker\\work\\cert\\external\\pilot_r42_10seed_matrix_v1",
    "seeds": [
      20261116,
      20261117,
      20261118
    ],
    "label_seed": 20261119,
    "signal_types": [
      "point",
      "sustained",
      "drift",
      "switch"
    ],
    "strengths": [
      0.5,
      1.0,
      2.0
    ],
    "k_grid": [
      5,
      10
    ],
    "cooldown_days": 7,
    "context_window": 5,
    "background_users": 96,
    "event_users": 24,
    "channels": [
      "9",
      "2",
      "10",
      "1",
      "6",
      "24",
      "4",
      "7",
      "5",
      "14",
      "18",
      "0"
    ],
    "crop_start": 239,
    "eval_start": 303,
    "eval_end": 500,
    "train_end": 158,
    "claims": {
      "external_validation": false,
      "dataset_role": "controlled anomaly-signal benchmark built on real normal behavior; not an external dataset"
    },
    "selected_user_count": 96,
    "target_user_count": 24
  },
  "clue_coverage": {
    "profile": "clue-coverage",
    "data": "E:\\MarketRisker\\work\\cert\\external\\clue_lds\\processed_final\\clue_user_day_dataset.npz",
    "model_root": "E:\\MarketRisker\\work\\cert\\external\\clue_dual_end_cap256_10seed_v1",
    "seeds": [
      20261116,
      20261117,
      20261118
    ],
    "label_seed": 20261119,
    "signal_types": [
      "point",
      "sustained",
      "drift",
      "switch"
    ],
    "strengths": [
      0.5,
      1.0,
      2.0
    ],
    "k_grid": [
      5,
      10
    ],
    "cooldown_days": 7,
    "context_window": 5,
    "background_users": 54,
    "event_users": 24,
    "channels": [
      "34",
      "49",
      "24",
      "33",
      "26",
      "12",
      "43",
      "38",
      "0",
      "32",
      "37",
      "19"
    ],
    "crop_start": 1634,
    "eval_start": 1698,
    "eval_end": 1910,
    "train_end": 1697,
    "claims": {
      "external_validation": false,
      "dataset_role": "controlled anomaly-signal benchmark built on real normal behavior; not an external dataset"
    },
    "selected_user_count": 54,
    "target_user_count": 24
  }
}
```
