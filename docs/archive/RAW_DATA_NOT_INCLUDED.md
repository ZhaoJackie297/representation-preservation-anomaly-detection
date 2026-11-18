# 原始大文件未随包复制

## 原因

E 盘原始 CERT 与中间数据规模过大，直接复制会影响传输和存储。

当前目录规模：

| 目录 | 约大小 |
| --- | ---: |
| `r4.2_extracted` | 15.07 GB |
| `r5.2_extracted` | 37.42 GB |
| `r6.2_extracted` | 93.17 GB |

合计约 145.7 GB。

全项目 CSV 总量约 146.7 GB，其中主要来自原始日志与抽取表。

## 已包含的替代材料

1. 各版本官方 readme 和 license；
2. 各版本数据审计 JSON；
3. 用户日摘要 JSON；
4. 序列元数据 JSON；
5. E0–E10 全部 CSV/JSON 结果；
6. 数据构建和审计代码；
7. `AI自监督时序实验.txt`；
8. research 目录的报告、文本和小型参考结果；
9. references 目录；
10. 日志。

## 未包含

- CERT 原始 CSV；
- tar/7z 压缩包；
- `.npz/.npy` 大数组；
- `.pth` 模型；
- 大型 cache；
- 训练中间二进制。
- `E:\MarketRisker\appdata` 浏览器运行态（可能含 Cookies、Local Storage、Trust Tokens，未打包）。

## 原始路径

```text
E:\MarketRisker\work\cert\r4.2_extracted\r4.2
E:\MarketRisker\work\cert\r5.2_extracted\r5.2
E:\MarketRisker\work\cert\r6.2_extracted\r6.2
E:\MarketRisker\work\cert\processed_r42
E:\MarketRisker\work\cert\processed_r52
E:\MarketRisker\work\cert\processed_r62
```

## 如需要全量数据

建议：

1. 在两台机器间直接复制 E 盘目录；
2. 或按版本建立压缩包并附 SHA-256；
3. 不要把 145 GB 原始数据塞入本项目轻量代码/文档包；
4. 复现实验时优先使用本包中的构建脚本和审计 JSON。
