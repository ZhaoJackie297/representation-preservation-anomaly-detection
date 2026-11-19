# 论文文件

- `manuscript.docx`：当前申报书主稿，等同 v6。
- `manuscript_v6.docx`：v6 版本化副本，加入 CERT 固定 10-seed、
  CLUE-LDS 外部应用级复验和受控异常形态分析。
- `manuscript_v5.docx`：v5/R0 冻结稿，保留用于版本追溯。
- `fact_check_20261118.md`：修正前事实核查报告。

论文最终数字以 `results/canonical/` 为准。历史草稿中的 E0–E10 编号、旧生产协议数字和“完整 S1–S5 系统实现”等表述已经被修正版替代。

v5 保留为 R0 基线；当前 `manuscript.docx` 是 v6，在原有章节后增加
§7.9 和附录 C。

重新生成 v6：

```powershell
pip install -r requirements-docs.txt
python scripts/build_v5_extended.py
```

