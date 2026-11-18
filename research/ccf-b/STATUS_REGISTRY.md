# 研究状态登记表

日期：2026-11-18

本表只区分状态，不把不同种类的失败混写。

| 状态码 | 当前对象 | 当前判断 |
| --- | --- | --- |
| `IMPLEMENTATION_BLOCKED` | 未通过覆盖/形状/训练接线的 run | 目前没有新的 blocked run；A0/A1 入口 smoke 通过 |
| `DATA_GATE_FAILED` | 外部数据许可、标签、时间或用户映射 | 目前 CLUE-LDS canonical 数据门禁通过 |
| `METHOD_NOT_SUPPORTED` | CERT/CLUE 画像锚机制 | 主带宽、kNN 带宽和 m=5/10 均平坦，机制不可识别 |
| `UNDERPOWERED` | CLUE-LDS 外部效应、r5.2 严格 Gate | 事件簇/MDE 不足，不能作高功效确认 |
| `PENDING` | A3/A4 两阶段、完整 pilot | 尚未执行或尚未通过 |
| `NOT_PURSUED` | LANL、OpTC、ChimeraLog | 数据包已冻结，不再等待外部数据 |

## 必须遵守的解释

- 平坦先验不是数据门禁失败；
- 低功效不是方法无效；
- 实现失败不是方法失败；
- 只有通过对应门禁的结果才能进入确认表。
