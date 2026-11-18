# 归档路径说明

`docs/archive/` 保留原研究交付包中的设计和审计文档。部分文档包含实验当时的 Windows 绝对路径，例如 `E:\MarketRisker\...`。这些路径用于追溯原始实验，不是当前仓库的运行要求。

当前仓库使用：

- 代码根：`runtime/cert/`
- TS2Vec：`references/ts2vec_official/`
- SoftCLT：`references/softclt_repo/`，由安装脚本下载
- 处理后数据：通过 `scripts/link_data.ps1` 链接
- 最终结果：`results/canonical/`

