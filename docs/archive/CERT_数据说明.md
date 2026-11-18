# CERT 数据说明

## 一、使用的数据版本

| 版本 | 用户数 | 日期数 | 用户日行数 | 已处理事件 | 恶意账户日 | 威胁实例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CERT r4.2 | 1,000 | 501 | 464,687 | 32,770,222 | 986 | 70 |
| CERT r5.2 | 2,000 | 517 | 973,252 | 67,566,656 | 1,307 | 99 |
| CERT r6.2 | 4,000 | 516 | 1,963,756 | 126,190,921 | 78 | 5 |

说明：

- “威胁实例”来自 `incidents.csv` 聚合后的评价事件数；
- r4.2 的 70 个事件按时间切分为验证和测试；
- r5.2、r6.2 作为外部时间测试；
- r6.2 虽然恶意账户日更多，但仅 5 个威胁实例，不适合作为稳定事件估计。

## 二、原始行为表

### logon.csv

字段：

- id；
- date；
- user；
- pc；
- activity（Logon/Logoff）。

用途：

- 登录/登出次数；
- 首次登录、最后登出、活跃跨度；
- 非工作时间登录；
- 其他机器登录；
- primary PC。

### device.csv

字段包括：

- date；
- user；
- pc；
- file_tree；
- activity（connect/disconnect）。

用途：USB 连接和断开次数。

### file.csv

CERT 原版中间事件包含 open/copy/write/delete 等类型。本项目为了和 r4.2 的可移动介质语义一致，主要使用写入可移动介质的文件事件：

- file copy count；
- archive copy；
- executable copy；
- unique file extensions。

### email.csv

使用发送事件：

- email count；
- recipients；
- external recipients；
- attachments；
- message size；
- after-hours email。

r6.2 的 email 包含 View/Send；项目只使用 Send，以匹配早期版本。

### http.csv

使用：

- HTTP count；
- after-hours HTTP；
- external HTTP；
- unique domains。

### LDAP

逐月组织快照：

- role；
- functional_unit；
- department；
- team；
- supervisor。

处理：

- 同月匹配；
- 缺失时回退到上月；
- 组织字段不直接进入 27 维模型特征，但保留用于群体/情境分析。

缺失情况保存在：

- `cert_docs/r42_readme.txt`；
- `cert_docs/r52_readme.txt`；
- `cert_docs/r62_readme.txt`；
- `cert_docs/r4.2_audit.json`；
- `cert_docs/r5.2_audit.json`；
- `cert_docs/r6.2_audit.json`；
- processed 元数据。

## 三、标签定义

标签来自官方 answers 中的：

- insider user/account；
- scenario；
- event details；
- 时间信息。

项目构造的标签为：

> 官方恶意账户在相应日期上的“恶意账户日”。

处理原则：

- 标签只用于评价；
- 不进入特征构造；
- 不进入 SoftCLT；
- 不进入画像距离；
- 不进入表征库聚类；
- 不进入训练期标准化；
- 不进入测试期阈值/权重选择。

### 集体异常标签

CERT 没有统一的跨用户集体异常标签。官方信息主要描述内部威胁用户、账户和场景。因此：

- 不能把多用户同时活跃直接当集体异常；
- 不能在没有标签时量化集体异常 AUC；
- 只能做探索性案例分析或另行模拟。

## 四、27 维用户日特征

1. `has_any_event`
2. `logon_count`
3. `logoff_count`
4. `first_logon_minute`
5. `last_logoff_minute`
6. `active_span_minutes`
7. `after_hours_logon_count`
8. `other_pc_logon_count`
9. `usb_connect_count`
10. `usb_disconnect_count`
11. `file_copy_count`
12. `archive_copy_count`
13. `executable_copy_count`
14. `unique_file_extensions`
15. `email_count`
16. `email_recipient_count`
17. `external_recipient_count`
18. `email_attachment_count`
19. `email_size_bytes`
20. `after_hours_email_count`
21. `http_count`
22. `after_hours_http_count`
23. `external_http_count`
24. `unique_http_domains`
25. `is_weekend`
26. `weekday_sin`
27. `weekday_cos`

## 五、SoftCLT 47 维画像使用的特征

从 27 维中排除：

- `has_any_event`；
- 3 个日历特征。

保留 23 个：

- 20 个行为计数；
- 首次登录分钟；
- 最后登出分钟；
- 活跃跨度。

画像：

```text
23 median + 23 q90 + 1 active_fraction = 47
```

## 六、时间切分

### r4.2

- train：2010-01-02 至 2010-06-09；
- validation：2010-06-10 至 2010-10-31；
- test：2010-11-01 至 2011-05-17。

### r5.2

- train：2010-01-02 至 2010-04-15；
- external test：2010-04-16 至 2011-06-02。

### r6.2

- train：2010-01-02 至 2010-08-11；
- external test：2010-08-12 至 2011-06-01。

## 七、官方数据说明位置

- `cert_docs/r42_readme.txt`
- `cert_docs/r52_readme.txt`
- `cert_docs/r62_readme.txt`
- `cert_docs/r4.2_audit.json`
- `cert_docs/r5.2_audit.json`
- `cert_docs/r6.2_audit.json`


