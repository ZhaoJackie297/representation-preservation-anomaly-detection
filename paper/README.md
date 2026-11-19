# 论文文件

- `manuscript.docx`：2026-11-18 修正版。
- `manuscript_v6.docx`：2026-11-19 扩展稿，加入 CERT 固定 10-seed、
  CLUE-LDS 外部应用级复验和受控异常形态分析。
- `fact_check_20261118.md`：修正前事实核查报告。

论文最终数字以 `results/canonical/` 为准。历史草稿中的 E0–E10 编号、旧生产协议数字和“完整 S1–S5 系统实现”等表述已经被修正版替代。

v6 不替代 v5：v5 保留为 R0 基线，v6 在原有章节后增加 §7.9 和附录 C。

重新生成 v6：

```powershell
pip install -r requirements-docs.txt
python scripts/build_v5_extended.py
```

