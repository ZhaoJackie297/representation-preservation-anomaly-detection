from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from shutil import copyfile

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "manuscript_v5.docx"
OUTPUT = ROOT / "paper" / "manuscript.docx"
VERSIONED_OUTPUT = ROOT / "paper" / "manuscript_v6.docx"


def set_run_font(
    run,
    *,
    size: float,
    bold: bool | None = None,
) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:ascii"), "Times New Roman")
    fonts.set(qn("w:hAnsi"), "Times New Roman")
    fonts.set(qn("w:eastAsia"), "SimSun")


def clear_content(paragraph: Paragraph) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)


def replace_text(paragraph: Paragraph, text: str) -> None:
    first_run = next(
        (run for run in paragraph.runs if run.text),
        paragraph.runs[0] if paragraph.runs else None,
    )
    size = (
        first_run.font.size.pt
        if first_run is not None and first_run.font.size
        else 12.0
    )
    clear_content(paragraph)
    run = paragraph.add_run(text)
    set_run_font(run, size=size)


def iter_all_paragraphs(container):
    yield from container.paragraphs
    for table in container.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from iter_all_paragraphs(cell)


def add_text_paragraph_before(
    anchor: Paragraph,
    text: str,
    *,
    size: float = 12.0,
    first_line: bool = True,
    bold_lead: str | None = None,
    keep_with_next: bool = False,
) -> Paragraph:
    paragraph = anchor.insert_paragraph_before(style="Normal")
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.keep_with_next = keep_with_next
    if first_line:
        paragraph.paragraph_format.first_line_indent = Inches(0.3333)
    if bold_lead:
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, size=size, bold=True)
        body = paragraph.add_run(text)
        set_run_font(body, size=size, bold=False)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, size=size)
    return paragraph


def add_heading_before(
    anchor: Paragraph,
    text: str,
    level: int,
    *,
    page_break_before: bool = False,
) -> Paragraph:
    paragraph = anchor.insert_paragraph_before(
        style=f"Heading {level}"
    )
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.page_break_before = page_break_before
    run = paragraph.add_run(text)
    set_run_font(
        run,
        size=14.0 if level <= 2 else 12.5,
        bold=True,
    )
    return paragraph


def add_caption_before(anchor: Paragraph, text: str) -> Paragraph:
    paragraph = anchor.insert_paragraph_before(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(8)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, size=10.5, bold=True)
    return paragraph


def set_cell_text(
    cell,
    text: str,
    *,
    size: float,
    bold: bool,
    align: WD_ALIGN_PARAGRAPH,
) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.0
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold)


def add_table_before(
    document: Document,
    anchor: Paragraph,
    rows: list[list[str]],
    widths: list[float],
    *,
    size: float = 9.5,
) -> Table:
    table = document.add_table(
        rows=len(rows),
        cols=len(rows[0]),
    )
    template = document.tables[9]
    table.style = template.style
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    template_borders = template._tbl.tblPr.find(
        qn("w:tblBorders")
    )
    existing_borders = table._tbl.tblPr.find(qn("w:tblBorders"))
    if existing_borders is not None:
        table._tbl.tblPr.remove(existing_borders)
    if template_borders is not None:
        table._tbl.tblPr.append(deepcopy(template_borders))

    for row_index, row_values in enumerate(rows):
        row = table.rows[row_index]
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
        if row_index == 0:
            header = OxmlElement("w:tblHeader")
            tr_pr.append(header)
        for column_index, value in enumerate(row_values):
            cell = row.cells[column_index]
            cell.width = Inches(widths[column_index])
            alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if column_index == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            set_cell_text(
                cell,
                value,
                size=size,
                bold=row_index == 0,
                align=alignment,
            )
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    anchor._p.addprevious(table._tbl)
    return table


def append_cloned_row(table: Table, values: list[str]) -> None:
    old_row = table.rows[-1]
    new_tr = deepcopy(old_row._tr)
    table._tbl.append(new_tr)
    new_row = table.rows[-1]
    tr_pr = new_row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))
    for index, value in enumerate(values):
        source_cell = old_row.cells[index]
        source_run = (
            source_cell.paragraphs[0].runs[0]
            if source_cell.paragraphs[0].runs
            else None
        )
        size = (
            source_run.font.size.pt
            if source_run is not None and source_run.font.size
            else 10.5
        )
        bold = source_run.bold if source_run is not None else False
        alignment = source_cell.paragraphs[0].alignment
        set_cell_text(
            new_row.cells[index],
            value,
            size=size,
            bold=bool(bold),
            align=alignment or WD_ALIGN_PARAGRAPH.LEFT,
        )


def clone_paragraph_after(
    reference: Paragraph,
    text: str,
) -> Paragraph:
    clone = deepcopy(reference._p)
    reference._p.addnext(clone)
    paragraph = Paragraph(clone, reference._parent)
    clear_content(paragraph)
    run = paragraph.add_run(text)
    set_run_font(run, size=12.0)
    return paragraph


def find_exact(document: Document, text: str) -> Paragraph:
    matches = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.strip() == text
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one paragraph {text!r}, got {len(matches)}"
        )
    return matches[0]


def main() -> None:
    document = Document(SOURCE)
    directory_table = document.tables[20]

    replace_text(
        find_exact(document, "以 CERT 用户行为数据为例"),
        "以 CERT 与 CLUE-LDS 用户行为数据为例",
    )

    abstract = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith(
            "内部威胁检测面临恶意行为高度隐蔽"
        )
    )
    addition = abstract.add_run(
        "本文进一步在 r4.2 与 r5.2 上固定 10 个随机种子复核联合方案，"
        "并在独立云存储正常行为上构造半合成账号劫持进行外部复验。"
        "结果显示联合方案在 CERT 两个版本上保持方向一致，欧氏与 PAI "
        "打分端获得有限的跨应用场景支持；受控异常实验表明，点级残差与"
        "时间上下文适用于不同异常形态，但 temporal-soft 和 Coverage "
        "的跨域端到端收益仍未成立。"
    )
    set_run_font(addition, size=12.0)

    keywords = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith("关键词：")
    )
    keyword_run = keywords.add_run("；CLUE-LDS")
    set_run_font(keyword_run, size=12.0)

    dataset_table = document.tables[2]
    append_cloned_row(
        dataset_table,
        [
            "CLUE-LDS",
            "5,389 用户；50.5M 事件；1,910 日",
            "训练至 2022-02-28；评价至 2022-09-29",
            "外部应用级半合成复验",
        ],
    )
    replace_text(
        next(
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.startswith("实验使用 CERT r4.2")
        ),
        "实验使用 CERT r4.2、r5.2、r6.2、SMD 与 CLUE-LDS 五套数据。"
        "前三套 CERT 版本承担主开发、锁定复验与小样本压力审计，SMD "
        "承担跨数据类型验证，CLUE-LDS 承担外部应用级半合成复验。",
    )

    data_anchor = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith(
            "CERT 训练期在首个官方恶意事件前结束"
        )
    )
    add_text_paragraph_before(
        data_anchor,
        "Landauer 等发布的 CLUE-LDS 来自真实云存储环境，包含 "
        "50,522,931 条日志、5,389 个匿名用户和 1,910 个数据日。"
        "原始数据只包含正常行为。本文采用官方切换算法和独立固定 seed "
        "构造 24 个用户对、48 个半合成账号劫持事件，其中训练截止到 "
        "2022-02-28，评价窗口为 2022-03-01 至 2022-09-29。该数据只"
        "用于有限功效的外部应用级复验，不把注入事件写成真实攻击标签。",
    )

    experiment_table = document.tables[4]
    for values in (
        [
            "E12",
            "CERT 固定 10-seed 扩展验证",
            "A0-official、A0-matched、A1-Cov 与 plain/PAI-A1 固定矩阵，"
            "10 个训练 seed，不调参",
            "r4.2/r5.2 的联合主检验与 Coverage-PAI 增量均通过 Holm",
        ],
        [
            "E13",
            "CLUE-LDS 外部应用级复验",
            "真实正常云行为；官方切换算法与独立 seed 构造 24 pairs/"
            "48 events；10 个训练 seed",
            "PAI 点级排序获得外部支持；Coverage 未迁移；K=10 饱和",
        ],
        [
            "E14",
            "受控异常形态压力测试",
            "CERT 正常序列；point/sustained/drift/switch × 3 强度和 "
            "3 seeds；point/context/combined 与 iForest",
            "区分不同异常形态所需信号；temporal-soft 改善部分排序，"
            "但不提高固定 K 召回",
        ],
    ):
        append_cloned_row(experiment_table, values)

    replace_text(
        next(
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.startswith("实验按 E0–E11 的顺序演进")
        ),
        "实验按 E0–E14 的顺序演进，每一步回答一个具体问题。"
        "E0–E11 完成单因素对照、联合评价、外部审计与经典基线；"
        "E12 以固定 10-seed 复核 CERT 稳定性，E13 在 CLUE-LDS 上"
        "开展外部应用级半合成复验，E14 通过受控异常分析评分信号的"
        "适用形态。表 5 列出每项实验的内容、做法与关键结果。",
    )
    replace_text(
        find_exact(document, "表 5  E0–E11 实验一览"),
        "表 5  E0–E14 实验一览",
    )
    replace_text(
        find_exact(document, "图 2  实验演进路线"),
        "图 2  核心实验演进路线（E0–E11）",
    )
    replace_text(
        next(
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.startswith(
                "各实验的训练子集、窗口、K 值与随机种子保持一致"
            )
        ),
        "图 2 概括 E0–E11 的核心实验演进。各实验的训练子集、窗口、"
        "K 值与随机种子保持一致；E12–E14 的 10-seed、CLUE-LDS 与"
        "受控异常扩展见 7.9 节，完整逐 seed 结果见附录 B 和附录 C。",
    )
    pseudocode_intro = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith(
            "以下为三个核心方法的伪代码"
        )
    )
    replace_text(
        pseudocode_intro,
        "以下为三个核心方法的伪代码。E2 是训练端 instance-soft 的"
        "训练流程，E3 是打分端 PAI 幅度评分的构造，E4 是双端联合的"
        "评价流程。其余实验（E0、E1、E5–E14）均为标准流程、同一方法"
        "的重复应用或扩展验证；E12–E14 的详细协议见 7.9 节和附录 C。"
        "完整可运行代码见 Git 仓库的 runtime/cert/ 目录，按 S1–S5 "
        "阅读的代码见 code/by_layer/ 目录。",
    )

    replace_text(
        next(
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.startswith("CERT 主实验固定三随机种子")
        ),
        "CERT 历史 T0–T4 对照固定 seed 20261116、20261117、20261118；"
        "E12 的固定矩阵扩展到 20261116–20261125 共 10 个训练 seed。"
        "SMD 当前使用单一种子 20261116。所有实验报告全部 seed 的均值，"
        "不按结果选择最佳 seed。r6.2 联合实验中 T4 与匹配 T0 使用同一组 "
        "1,000 名训练用户；SMD 联合实验保持窗口、epoch、编码与评分协议"
        "一致。证据等级仍区分多协议配对、单 seed 和小事件压力审计。",
    )

    contribution = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith("多协议外部审计：")
    )
    replace_text(
        contribution,
        "多协议外部审计：在三套 CERT 数据集、SMD 与 CLUE-LDS 上，"
        "联合报告点级排序指标与固定预算事件命中；CLUE-LDS 采用真实"
        "正常云行为和非真实攻击标签的半合成账号劫持，仅作有限外部复验。",
    )
    clone_paragraph_after(
        contribution,
        "机制边界与扩展验证：用 CERT 固定 10-seed 矩阵复核方向稳定性，"
        "并用受控点尖峰、持续偏移、缓慢漂移与账号切换，分析不同异常"
        "形态下 point residual、时间上下文与 temporal-soft 的作用边界。",
    )

    discussion_anchor = find_exact(document, "8 讨论")

    add_heading_before(
        discussion_anchor,
        "7.9 扩展验证与异常形态分析",
        level=2,
    )
    add_text_paragraph_before(
        discussion_anchor,
        "本节在 v5 原有三随机种子结果之外补充三类证据。第一，使用固定 "
        "10 个 seed 复核 CERT 联合方案与覆盖软训练的稳定性。第二，在"
        "独立云存储正常行为上开展外部应用级半合成复验。第三，通过"
        "受控异常注入解释点级残差、上下文聚合与 temporal-soft 的适用"
        "形态。受控实验只回答机制问题，不作为外部验证或方法排名依据。",
    )

    add_heading_before(
        discussion_anchor,
        "7.9.1 CERT 固定 10-seed 验证",
        level=3,
    )
    add_text_paragraph_before(
        discussion_anchor,
        "为避免表 12 中缩写含义不清，三个训练配置按相同输入和评分协议定义如下。"
        "A0-official 使用官方随机 16 用户 batch，实例目标为硬 0/1；"
        "A0-matched 使用与 A1-Cov 相同的覆盖率约束 batch table，但实例目标"
        "仍为硬 0/1，用于消除 batch 组成差异；A1-Cov 保持同一 batch table，"
        "把 47 维训练期画像距离生成的软实例权重作为 instance target，"
        "并令 soft temporal 关闭、lambda=0.5。覆盖率表先把画像距离转换为"
        "行归一化高斯先验 P，带宽取训练期全体非对角距离的中位数；A3 取每个"
        "用户在该先验中的 top-10 邻边并对称化，A5a 在保持 A3 度数序列的约束下"
        "随机重连，A5b 在画像先验的行列联合置换后取 top-10 邻边。每个 batch "
        "使用 8 个设计槽位和 8 个随机填充用户；覆盖目标分别检查 A3、A5a 和 "
        "A5b，并均要求不低于 80%；训练步数取 base steps 与该表达到覆盖目标"
        "所需 batch 数的较大值。",
    )
    add_text_paragraph_before(
        discussion_anchor,
        "固定矩阵使用 A0-official、A0-matched 与 A1-Cov 三种训练方式，"
        "分别与 plain 和固定 PAI-A1 权重组合。所有方法共享同一数据切分、"
        "用户日特征、K=5/10/20、7 天冷却期与评分校准。训练 seed 为 "
        "20261116 至 20261125，共 10 个；4 个预注册主检验采用配对比较"
        "并做 Holm 校正。表 12 汇总联合方案相对官方普通基线的结果。",
    )
    add_caption_before(
        discussion_anchor,
        "表 12  CERT r4.2/r5.2 固定 10-seed 联合结果",
    )
    add_table_before(
        document,
        discussion_anchor,
        [
            ["数据集", "方法", "PR-AUC", "ROC-AUC", "K=10", "主差值"],
            [
                "r4.2",
                "A0-official + plain",
                "0.007349",
                "0.804604",
                "11.3/24",
                "—",
            ],
            [
                "r4.2",
                "A1-Cov + PAI-A1",
                "0.012839",
                "0.876870",
                "13.9/24",
                "+0.005490 / +0.072266 / +2.6",
            ],
            [
                "r5.2",
                "A0-official + plain",
                "0.004929",
                "0.800590",
                "19.5/99",
                "—",
            ],
            [
                "r5.2",
                "A1-Cov + PAI-A1",
                "0.007067",
                "0.869525",
                "32.5/99",
                "+0.002138 / +0.068935 / +13.0",
            ],
        ],
        [0.65, 1.65, 0.85, 0.85, 0.8, 1.45],
        size=9.0,
    )
    add_text_paragraph_before(
        discussion_anchor,
        "在 r4.2 与 r5.2 上，联合方案相对官方普通基线均保持正向。"
        "r4.2 的 PR-AUC、事件召回分别为 +0.005490、+0.108333，"
        "r5.2 分别为 +0.002138、+0.131313；两组比较均为 10/10 seeds "
        "同向并通过 Holm。覆盖软训练在 matched PAI 之上仍有增量："
        "r4.2 的 PR-AUC 为 +0.002505、K=10 为 +0.050000，r5.2 为 "
        "+0.001504 与 +0.144444。该结果加强 v5 原三随机种子结论，"
        "但 r4.2 仍是开发集，r5.2 是已见版本复验，不是盲测。",
    )

    add_heading_before(
        discussion_anchor,
        "7.9.2 CLUE-LDS 外部应用级复验",
        level=3,
    )
    add_text_paragraph_before(
        discussion_anchor,
        "CLUE-LDS 的原始日志来自真实云存储行为，异常标签由官方切换"
        "算法和固定 seed 构造。实验保留 838 个训练期活跃用户，使用 "
        "24 个用户对、48 个目标事件，主统计按 pair 聚类。训练 seed "
        "与标签 seed 分离。由于有效事件簇只有 24 个，80% 功效下可检出"
        "的事件级差异约为 23.0 至 29.5 个百分点；K=10 命中已经接近"
        "饱和，因此点级排序比事件级差异更具解释力。",
    )
    add_caption_before(
        discussion_anchor,
        "表 13  CLUE-LDS 外部测试 10-seed 结果",
    )
    add_table_before(
        document,
        discussion_anchor,
        [
            ["方法", "PR-AUC", "ROC-AUC", "K=10", "ΔPR vs plain", "定位"],
            [
                "A0-official + plain",
                "0.655168",
                "0.490361",
                "46.7/48",
                "—",
                "官方普通基线",
            ],
            [
                "A0-official + PAI-A1",
                "0.685456",
                "0.543051",
                "47.0/48",
                "+0.030288",
                "打分端增益",
            ],
            [
                "A0-matched + PAI-A1",
                "0.685989",
                "0.544035",
                "47.0/48",
                "+0.030821",
                "匹配批次控制",
            ],
            [
                "A1-Cov + plain",
                "0.662905",
                "0.499563",
                "45.5/48",
                "+0.007737",
                "覆盖训练对照",
            ],
            [
                "A1-Cov + PAI-A1",
                "0.684023",
                "0.540662",
                "47.0/48",
                "+0.028855",
                "完整系统",
            ],
        ],
        [1.45, 0.8, 0.8, 0.8, 1.0, 1.25],
        size=8.5,
    )
    add_text_paragraph_before(
        discussion_anchor,
        "PAI 打分使官方训练配置的 PR-AUC 从 0.655168 提高到 0.685456，"
        "提升 +0.030288，10/10 seeds 正向并通过 Holm。完整系统相对"
        "官方普通基线的 PR-AUC 为 +0.028855，增益主要来自打分端。"
        "覆盖软训练在 matched PAI 之上没有复现：PR-AUC 为 -0.001965，"
        "0/10 seeds 为正；K=10 在 47/48 左右饱和，无法区分方法。"
        "因此本数据支持欧氏与幅度评分端的跨应用场景迁移，不支持完整"
        "双端方法的外部普适性。",
    )

    add_heading_before(
        discussion_anchor,
        "7.9.3 受控异常形态与评分信号",
        level=3,
    )
    add_text_paragraph_before(
        discussion_anchor,
        "为解释不同异常形态下的信号响应，本文以 CERT 正常用户日序列"
        "为母体，构造单日尖峰、7 天持续偏移、10 天缓慢漂移和 7 天账号"
        "切换四类事件。每类使用 0.5、1.0、2.0 三档强度，模型训练 seed "
        "为 20261116–20261118，标签 seed 独立固定。评价保留 96 个背景"
        "用户和 24 个事件用户，point residual、5 天因果上下文与二者"
        "组合使用相同 K 和冷却期。该实验只用于机制分析。",
    )
    add_caption_before(
        discussion_anchor,
        "表 14  T4 在不同异常形态下的 PR-AUC 与 K=10（强度 2.0）",
    )
    add_table_before(
        document,
        discussion_anchor,
        [
            ["异常形态", "Point PR / K=10", "Context PR / K=10", "Combined PR / K=10"],
            ["单日尖峰", "0.3420 / 0.4167", "0.0066 / 0.2917", "0.0533 / 0.3750"],
            ["持续偏移", "0.6957 / 1.0000", "0.6404 / 1.0000", "0.7876 / 1.0000"],
            ["缓慢漂移", "0.1710 / 1.0000", "0.1567 / 1.0000", "0.2233 / 1.0000"],
            ["账号切换", "0.0411 / 0.9306", "0.0877 / 0.9167", "0.0838 / 0.9167"],
        ],
        [1.2, 1.6, 1.6, 1.6],
        size=9.0,
    )
    add_text_paragraph_before(
        discussion_anchor,
        "单日尖峰主要由 point residual 捕获，context 与 combined 的"
        "平滑操作降低了该类异常的点级排序；持续偏移和缓慢漂移更适合"
        "上下文聚合，combined 的 PR-AUC 高于单点残差；账号切换则同时"
        "需要即时偏离和上下文证据。没有一种分数覆盖全部形态，这支持"
        "v5 将点级与上下文信号并列报告，而不是压成单一阈值。",
    )
    add_caption_before(
        discussion_anchor,
        "表 15  temporal-soft 在受控异常上的增量（强度 2.0）",
    )
    add_table_before(
        document,
        discussion_anchor,
        [
            ["异常形态", "ΔPR（point）", "ΔPR（combined）", "ΔK=10（point）"],
            ["单日尖峰", "+0.067190", "+0.005393", "-0.013889"],
            ["持续偏移", "+0.076616", "+0.038584", "0.000000"],
            ["缓慢漂移", "+0.022430", "+0.030855", "0.000000"],
            ["账号切换", "+0.007740", "+0.018801", "+0.041667"],
        ],
        [1.2, 1.35, 1.45, 1.4],
        size=9.0,
    )
    add_text_paragraph_before(
        discussion_anchor,
        "T4 相对 T3 在四个形态、三档强度上的 point-residual PR-AUC "
        "差值均为正，12/12 个设置方向为正，其中 10 个设置为 3/3 seeds "
        "同向；平均差值为 +0.015008。然而 12 个设置中只有 3 个固定 "
        "K=10 差值为正，点尖峰还出现小幅下降。该结果与 v5 的判断一致："
        "temporal-soft 改变了表示排序中的局部时间几何，但没有形成"
        "稳定的端到端固定预算收益。",
    )
    add_heading_before(
        discussion_anchor,
        "7.9.4 SMD 5-seed 与深度基线边界",
        level=3,
    )
    add_text_paragraph_before(
        discussion_anchor,
        "为检验原 SMD 单 seed 结论，本文追加连续训练 seed "
        "20261201–20261205，对 28 台机器分别训练与测试。T4 相对匹配 "
        "T0 的 PR-AUC 差值为 +0.004661，5/5 seed 同向，machine-cluster "
        "95% CI 为 [+0.001820, +0.007996]；ROC-AUC 差值为 +0.002310，"
        "PA-F1@Top-5 差值为 +0.003249，二者也均为 5/5 seed 同向。然而 "
        "q99 segment recall 差值为 -0.007530，仅 1/5 seed 为正，区间"
        "不跨零。因此新增证据支持点级排序和 top-fraction 指标，不支持"
        "“所有指标全面提升”。",
    )
    add_text_paragraph_before(
        discussion_anchor,
        "深度基线采用同一 28 机器和同 seed 评估。OmniAnomaly-adapted 的 "
        "PR-AUC/ROC-AUC 为 0.441056/0.813013，T4-full-soft 为 "
        "0.460913/0.846348；T4 的 PR-AUC 和 PA-F1 差值区间均跨零，只有 "
        "ROC-AUC 差值区间为正，不能写成对 OmniAnomaly 的全面胜出。"
        "Anomaly Transformer-adapted 按官方 100 点窗口运行，三项均值为 "
        "0.045713/0.521273/0.081471，明显低于 T0/T4，但这是 SMD 用户级"
        "适配结果，不是该模型在原论文基准上的失败。",
    )
    add_text_paragraph_before(
        discussion_anchor,
        "CERT 深度基线只作为 3-seed 探索性适配：t-NNCLR 的 PR-AUC/"
        "ROC-AUC/K=10 召回为 0.003709/0.643231/0.319444；Anomaly "
        "Transformer 为 0.002324/0.497332/0.055556，增加到 30 epochs "
        "后仍为 0.002344/0.499098/0。它们没有支持深度基线在 CERT "
        "用户日任务上的迁移。PAI 权重方面，项目权重在 r4.2 10-seed 和 "
        "SMD 5-seed 的点级排序上均优于官方标量权重，且 CERT 上多为 "
        "10/10 seed 同向；但事件级差异混合，不能把项目权重描述成"
        "全面最优。",
    )
    add_text_paragraph_before(
        discussion_anchor,
        "综合新增结果，本文可支持的定位仍是：在 CERT 上经过固定 10-seed "
        "和完整对照验证的双端优化方法，外加一套明确的多数据集边界。"
        "它不构成跨所有数据类型、所有指标和所有深度基线的普适 SOTA。",
    )

    main_findings = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith("以下发现值得强调")
    )
    findings_addition = main_findings.add_run(
        "扩展验证进一步收紧了结论。固定 10-seed 矩阵加强了 r4.2 与 "
        "r5.2 的联合收益；CLUE-LDS 支持 PAI 打分端的有限外部迁移，"
        "但不支持 Coverage 的外部增益；受控异常实验说明 point residual "
        "与时间上下文具有不同适用形态，点级排序改善也不自动转化为"
        "固定 K 事件召回。SMD 新增 5-seed 在 PR-AUC、ROC-AUC 和 "
        "PA-F1 上方向一致，但 q99 segment recall 下降；适配的 "
        "OmniAnomaly 与 Anomaly Transformer 没有形成更强的统一证据。"
        "iForest 的优势结论限定于 CERT r4.2 的 E11 主评价；附录 C "
        "的高幅度受控场景中，iForest 可以强于神经表征。"
    )
    set_run_font(findings_addition, size=12.0)

    limitation_prefixes = (
        "（1）统计功效与多重比较：",
        "（2）小样本事件审计的局限：",
        "（3）组件归因与基线对比：",
        "（4）系统集成度：",
    )
    limitations = []
    for prefix in limitation_prefixes:
        matches = [
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.startswith(prefix)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one limitation paragraph for {prefix}, "
                f"got {len(matches)}"
            )
        limitations.append(matches[0])

    replacement_limits = [
        (
            "（1）统计功效与多重比较：",
            "CERT 历史 T0–T4 对照只有 3 个训练 seed；新增固定矩阵将"
            "r4.2 与 r5.2 扩展到 10 seeds 并采用 Holm 校正，二者仍分别"
            "属于开发集和已见复验。SMD 已扩展到 5 个新 seed，PR-AUC、"
            "ROC-AUC 和 PA-F1 方向稳定，但 q99 segment recall 下降，且"
            "OmniAnomaly 是适配实现，后续仍需原论文代码级复验。",
        ),
        (
            "（2）外部标签与事件功效：",
            "CLUE-LDS 的异常是固定 seed 构造的半合成账号劫持，不是真实"
            "攻击标签；24 个有效 pair 对中等事件效应的功效不足，且 K=10 "
            "已接近饱和。r6.2 只有 5 个威胁实例，只用于压力审计。",
        ),
        (
            "（3）组件归因、生成器依赖与基线：",
            "temporal-soft 在当前日粒度设定下没有稳定端到端增量；magG "
            "在点级和事件级上存在分歧。受控异常实验会偏向高幅度、易分"
            "离的表格信号，iForest 在部分注入场景中更强，因此该实验"
            "不能替代真实外部验证。本文新增了 OmniAnomaly-adapted、"
            "Anomaly Transformer 和 t-NNCLR 的适配对照，但窗口、实现"
            "或任务粒度并不完全一致，仍不能替代严格的深度方法同构比较。",
        ),
        (
            "（4）系统集成度：",
            "S3 与 S4 仍是历史系统原型，S4 尚未与最新 47 维 T4 表示"
            "完成端到端对接。风险积累实验没有形成稳定的运营收益，后续"
            "需要更完整的队列、复核成本与案例验证。",
        ),
    ]
    for paragraph, (lead, body) in zip(limitations, replacement_limits):
        clear_content(paragraph)
        lead_run = paragraph.add_run(lead)
        set_run_font(lead_run, size=12.0, bold=True)
        body_run = paragraph.add_run(body)
        set_run_font(body_run, size=12.0, bold=False)

    boundary_first = find_exact(
        document,
        "综合上述局限，本文结论的适用边界如下。双端联合的正向收益在 "
        "r4.2 与 r5.2 的点级排序与固定预算事件召回上得到支持，在 "
        "r6.2 事件级与 SMD 单种子条件下仍需更多证据；结论仅相对于 "
        "TS2Vec 原版、匹配 T0 与 iForest 经典基线成立。",
    )
    replace_text(
        boundary_first,
        "综合上述局限，本文结论的适用边界如下。双端联合的正向收益在 "
        "r4.2、r5.2 的历史三 seed 实验和固定 10-seed 扩展中均得到"
        "支持；CLUE-LDS 为欧氏与幅度评分端提供了有限的外部应用级"
        "半合成支持，但没有复现 Coverage 的外部增益。SMD 新 5-seed "
        "支持点级与 top-fraction 指标，但 q99 segment recall 下降；"
        "r6.2 事件级与受控异常实验仍需更多证据。结论仅相对于 TS2Vec "
        "原版、匹配 T0、iForest 及本文明确标注的深度适配基线成立。",
    )
    boundary_second = find_exact(
        document,
        "需要说明的是，实例软权重对正常邻域结构的改善结论来自 r4.2 的 "
        "T3 对照；r5.2 验证的是 T4 完整 SoftCLT 相对匹配 T0 的联合"
        "收益，不能把该收益单独归因于 instance-soft。",
    )
    replace_text(
        boundary_second,
        "需要说明的是，实例软权重对正常邻域结构的改善结论来自 r4.2 的 "
        "T3 对照；r5.2 验证的是 T4 完整 SoftCLT 相对匹配 T0 的联合"
        "收益，不能把该收益单独归因于 instance-soft。CLUE-LDS 的"
        "正向外部证据主要落在 PAI 和欧氏幅度打分，而不是训练端覆盖"
        "机制。",
    )

    conclusion_first = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith("综上所述，本文工作聚焦于 TS2Vec")
    )
    replace_text(
        conclusion_first,
        "综上所述，本文聚焦 TS2Vec 在内部威胁检测中的方法适配与"
        "实证评价。围绕“表征信息保全”主线，训练端以 instance-soft "
        "和 47 维训练期画像距离缓解硬实例对比造成的正常类碰撞，"
        "打分端以欧氏表征库保留行为幅度，PAI 幅度项作探索性辅助；"
        "系统层面实现固定预算告警、严重度评分与可追溯证据原型。"
        "instance-soft 改善了正常邻域一致率与有效秩；r4.2 与 r5.2 "
        "在三 seed 和固定 10-seed 扩展中支持联合方案的点级及固定 "
        "K 事件收益；SMD 新 5-seed 进一步支持 PR-AUC、ROC-AUC 与 "
        "PA-F1，但 q99 segment recall 下降；CLUE-LDS 为 PAI 打分端"
        "提供有限外部应用级支持；iForest 与深度适配基线对照表明联合"
        "方案在 CERT r4.2 主评价上仍具竞争力，但并非普适 SOTA。",
    )
    conclusion_second = next(
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith("后续工作沿三个方向展开")
    )
    replace_text(
        conclusion_second,
        "后续工作沿三个方向展开。其一，外部证据：获取具备真实事件标签"
        "的独立组织数据，扩大 CLUE-LDS 的事件簇和标签真实性，避免把"
        "半合成复验写成真实攻击确认。其二，方法与基线：把 CERT 的"
        "20-seed 预注册矩阵补齐，统一深度基线的窗口和实现预算，并加入"
        "VUS-PR、PA%K、Affiliation 等更细粒度评价；将"
        "temporal-soft 限定在能够被机制实验支持的场景。其三，系统与"
        "运营：推进 S3/S4 与当前 T4 表示的端到端连接，预注册风险阈值、"
        "人工复核成本和失败案例，形成可复核的告警闭环。",
    )

    references_anchor = find_exact(
        document,
        "附录 A 关键代码",
    )
    reference_entries = [
        "[43] Landauer M, Skopik F, Höld G, Wurzenberger M. A User and "
        "Entity Behavior Analytics Log Data Set for Anomaly Detection in "
        "Cloud Computing[C]//2022 IEEE International Conference on Big "
        "Data. 2022. DOI:10.1109/BigData55660.2022.10020672.",
        "[44] Landauer M, Skopik F, Höld G, Wurzenberger M. Cloud-based "
        "User Entity Behavior Analytics Log Data Set[DS]. Zenodo, 2022. "
        "DOI:10.5281/zenodo.7119953.",
    ]
    for entry in reference_entries:
        paragraph = references_anchor.insert_paragraph_before(
            style="Normal"
        )
        paragraph.paragraph_format.line_spacing = 1.3
        paragraph.paragraph_format.space_after = Pt(4)
        run = paragraph.add_run(entry)
        set_run_font(run, size=10.5)

    replace_text(
        find_exact(document, "表 12  实验目录与内容"),
        "表 16  实验目录与内容",
    )
    replace_text(
        next(
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.startswith("仓库中，runtime/cert/")
        ),
        "仓库中，runtime/cert/ 保留原实验导入关系，code/by_layer/ "
        "提供 S1–S5 分层阅读入口，results/representation_preservation/ "
        "保存 E0–E11 的轻量记录；research/ccf-b/ 保存 E12–E14 的"
        "10-seed、CLUE-LDS 和受控异常汇总。Coverage 批次构造和 A3/A5a/A5b "
        "图构造分别位于 phase0/common_coverage_sampler.py 与 "
        "clue_lds/build_prior_graphs.py，r4.2 的逐 seed 覆盖统计位于 "
        "r4.2_10seed_coverage_summary.csv。SMD 新 5-seed 汇总位于 "
        "smd_multiseed/results/，深度基线与官方 PAI 权重对照位于 "
        "deep_baselines/results/ 和 official_pai/。",
    )
    for values in (
        [
            "research/ccf-b",
            "CERT 10-seed、CLUE-LDS 外部复验与轻量配对结果",
        ],
        [
            "research/ccf-b/phase1",
            "受控异常注入代码、协议、模拟事件和完整结果矩阵",
        ],
        [
            "research/ccf-b/smd_multiseed",
            "SMD 新 5-seed、机器配对差和 bootstrap 汇总",
        ],
        [
            "research/ccf-b/deep_baselines",
            "深度基线适配代码、CERT pilot、SMD 5-seed 和官方 PAI 权重对照",
        ],
    ):
        append_cloned_row(directory_table, values)

    appendix_c = document.add_paragraph(style="Heading 1")
    run = appendix_c.add_run("附录 C 受控异常实验补充")
    set_run_font(run, size=16.0, bold=True)

    def appendix_body(text: str) -> None:
        paragraph = document.add_paragraph(style="Normal")
        paragraph.paragraph_format.line_spacing = 1.5
        paragraph.paragraph_format.space_after = Pt(6)
        paragraph.paragraph_format.first_line_indent = Inches(0.3333)
        body = paragraph.add_run(text)
        set_run_font(body, size=12.0)

    appendix_body(
        "受控异常数据以 CERT r4.2 的真实正常用户日序列为母体，"
        "不修改训练期标签，只在评价期注入固定起点和持续时间的信号。"
        "四类事件分别为单日尖峰、持续偏移、缓慢漂移和账号切换；强度"
        "为 0.5、1.0、2.0；模型 seed 为 20261116–20261118，标签 seed "
        "独立为 20261119。CERT 配置保留 96 个背景用户和 24 个事件用户，"
        "CLUE 配置保留 54 个满足活跃度门槛的背景用户和 24 个事件用户。"
        "所有方法使用同一事件集合、同一 K 与同一冷却期。"
    )
    appendix_body(
        "表 17 给出最高强度下 T3、T4 与 iForest 的对照。"
        "iForest 在单日尖峰和缓慢漂移上强于神经表征，说明注入信号"
        "具有明显的幅度和单行可分性；T4 在持续偏移上保持最高 PR-AUC，"
        "在点尖峰上改善点级排序但没有提高固定 K 召回。该差异用于"
        "解释评价维度，不用于证明深度模型普遍优于表格基线。"
    )
    caption = document.add_paragraph(style="Normal")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(8)
    caption.paragraph_format.space_after = Pt(4)
    caption.paragraph_format.keep_with_next = True
    run = caption.add_run(
        "表 17  CERT 受控异常主方法对照（强度 2.0，三个 seed 均值）"
    )
    set_run_font(run, size=10.5, bold=True)
    add_table_append(
        document,
        [
            ["异常形态", "T3 PR / K=10", "T4 PR / K=10", "iForest PR / K=10", "主要观察"],
            ["单日尖峰", "0.2748 / 0.4306", "0.3420 / 0.4167", "0.3208 / 0.3333", "point 主导"],
            ["持续偏移", "0.6191 / 1.0000", "0.6957 / 1.0000", "0.5767 / 1.0000", "combined 更强"],
            ["缓慢漂移", "0.1486 / 1.0000", "0.1710 / 1.0000", "0.2299 / 1.0000", "上下文有效"],
            ["账号切换", "0.0334 / 0.8889", "0.0411 / 0.9306", "0.0122 / 0.8750", "T4 局部占优"],
        ],
        [1.0, 1.25, 1.25, 1.4, 1.2],
        size=8.5,
    )
    appendix_body(
        "CLUE 受控实验呈现相同倾向。最高强度下，A0-official 和 "
        "A1-Cov 的点级 PR-AUC 分别为：单日尖峰 0.0266/0.0345、"
        "持续偏移 0.1060/0.1288、缓慢漂移 0.0494/0.0631、账号切换 "
        "0.0179/0.0180；iForest 分别为 0.3762、0.6867、0.4567 和 "
        "0.0344。该结果不能作为 A1-Cov 外部更好的证据，因为真实 "
        "CLUE 复验中 Coverage 在 PAI 之上为负。完整逐 seed 结果见 "
        "research/ccf-b/phase1/results/。"
    )
    caption = document.add_paragraph(style="Normal")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(8)
    caption.paragraph_format.space_after = Pt(4)
    caption.paragraph_format.keep_with_next = True
    run = caption.add_run(
        "表 18  CLUE 受控异常主方法对照（强度 2.0，三个 seed 均值）"
    )
    set_run_font(run, size=10.5, bold=True)
    add_table_append(
        document,
        [
            ["异常形态", "A0 PR / K=10", "A1-Cov PR / K=10", "iForest PR / K=10"],
            ["单日尖峰", "0.0266 / 0.3333", "0.0345 / 0.3889", "0.3762 / 0.4167"],
            ["持续偏移", "0.1060 / 0.8472", "0.1288 / 0.8611", "0.6867 / 0.8472"],
            ["缓慢漂移", "0.0494 / 0.8750", "0.0631 / 0.8750", "0.4567 / 0.8750"],
            ["账号切换", "0.0179 / 0.8472", "0.0180 / 0.8611", "0.0344 / 0.8472"],
        ],
        [1.25, 1.55, 1.65, 1.55],
        size=8.5,
    )
    appendix_body(
        "综上，受控实验只支持边界结论：异常形态决定评分信号的选择；"
        "合成强度会奖励高幅度信号，iForest 的优势不能外推到真实数据。"
    )
    toc_anchor = find_exact(document, "8 讨论\t44")
    toc_reference = find_exact(document, "7.8 iForest 外部基线\t43")
    toc_lines = [
        "7.9 扩展验证与异常形态分析\t0",
        "7.9.1 CERT 固定 10-seed 验证\t0",
        "7.9.2 CLUE-LDS 外部应用级复验\t0",
        "7.9.3 受控异常形态与评分信号\t0",
        "7.9.4 SMD 5-seed 与深度基线边界\t0",
    ]
    insertion_after = toc_reference._p
    for line in toc_lines:
        clone = deepcopy(toc_reference._p)
        insertion_after.addnext(clone)
        insertion_after = clone
        paragraph = Paragraph(clone, toc_reference._parent)
        clear_content(paragraph)
        run = paragraph.add_run(line)
        set_run_font(run, size=12.0)

    appendix_toc_matches = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph.text.startswith(
            "附录 B 实验记录与可复现性说明\t"
        )
    ]
    if len(appendix_toc_matches) != 1:
        raise RuntimeError("Unexpected appendix B TOC entry count")
    appendix_toc_reference = appendix_toc_matches[0]
    appendix_toc = deepcopy(appendix_toc_reference._p)
    appendix_toc_reference._p.addnext(appendix_toc)
    paragraph = Paragraph(appendix_toc, appendix_toc_reference._parent)
    clear_content(paragraph)
    run = paragraph.add_run("附录 C 受控异常实验补充\t62")
    set_run_font(run, size=12.0)

    # Word is the reference renderer for the delivered DOCX. Keep the static
    # TOC synchronized with its pagination rather than a LibreOffice render.
    toc_pages = {
        "摘要": 4,
        "1 引言": 4,
        "术语说明": 6,
        "2 相关工作": 9,
        "2.1 异常检测任务与评价": 9,
        "2.2 自监督时序表示与异常检测": 10,
        "2.3 软对比学习与假负样本问题": 11,
        "2.4 表示打分、记忆库与幅度信息": 12,
        "2.5 内部威胁检测与 CERT 方法": 13,
        "2.6 图关系与集体异常": 14,
        "2.7 研究空白": 14,
        "3 系统总体设计": 14,
        "4 数据准备与实验设置": 16,
        "4.1 数据集": 16,
        "4.2 用户日特征：27 维（S1）": 18,
        "4.3 标签隔离与评价协议": 20,
        "4.4 实验流程与演进": 20,
        "4.4.1 核心算法伪代码": 24,
        "5 方法：双端优化（S2）": 26,
        "5.1 问题定义": 26,
        "5.2 训练端：instance-soft SoftCLT": 27,
        "5.2.1 软对比权重": 27,
        "5.2.2 五种训练配置": 27,
        "5.2.3 训练细节": 27,
        "5.2.4 加权对比损失的梯度分析": 28,
        "5.3 47 维用户画像距离": 28,
        "5.4 打分端：表征库": 29,
        "5.4.1 表征库构造": 29,
        "5.4.2 余弦与欧氏对照": 30,
        "5.5 PAI 幅度评分": 30,
        "5.6 方法边界": 31,
        "6 风险处理与证据报告（S3、S4）": 31,
        "6.1 固定预算 Top-K 告警队列": 31,
        "6.2 严重度评分及其演变": 32,
        "6.3 可追溯证据报告": 32,
        "7 实验结果": 33,
        "7.1 训练端机制验证": 33,
        "7.1.1 时间软权重无独立端到端收益": 33,
        "7.1.2 实例软权重改善": 34,
        "7.1.3 机制指标": 34,
        "7.1.4 三组对照的效应量": 35,
        "7.2 47 维画像构造效度": 36,
        "7.3 CERT r4.2：双端联合": 37,
        "7.3.1 联合结果": 37,
        "7.3.2 结论边界": 38,
        "7.4 CERT r5.2：外部组织验证": 39,
        "7.5 CERT r6.2：五事件压力审计": 40,
        "7.6 SMD：同窗口外部验证": 41,
        "7.6.1 SoftCLT 轻微增量": 41,
        "7.6.2 欧氏库最强统计证据": 42,
        "7.7 magG/T2 组件消融": 43,
        "7.7.1 SMD 消融": 43,
        "7.7.2 CERT 消融": 43,
        "7.7.3 综合结论": 43,
        "7.8 iForest 外部基线": 44,
        "7.9 扩展验证与异常形态分析": 45,
        "7.9.1 CERT 固定 10-seed 验证": 45,
        "7.9.2 CLUE-LDS 外部应用级复验": 46,
        "7.9.3 受控异常形态与评分信号": 47,
        "7.9.4 SMD 5-seed 与深度基线边界": 48,
        "8 讨论": 49,
        "8.1 主要发现": 49,
        "8.2 局限性": 50,
        "8.3 与已有工作的关系及 S3/S4 定位": 50,
        "8.4 结论边界": 51,
        "9 结论与展望": 51,
        "参考文献": 52,
        "附录 A 关键代码": 55,
        "附录 B 实验记录与可复现性说明": 60,
        "附录 C 受控异常实验补充": 62,
    }
    for title, page in toc_pages.items():
        matches = [
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.startswith(f"{title}\t")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one TOC entry {title!r}, got {len(matches)}"
            )
        replace_text(matches[0], f"{title}\t{page}")

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text.startswith(("算法 ", "代码 S")):
            paragraph.paragraph_format.keep_with_next = True

    reference_updates = {
        (
            "[2] Lee S, Park T, Lee K. Soft Contrastive Learning for "
            "Time Series[C]//International Conference on Learning "
            "Representations. 2024. Spotlight. arXiv:2312.16424."
        ): (
            "[2] Lee S, Park T, Lee K. Soft Contrastive Learning for "
            "Time Series[C]//The Twelfth International Conference on "
            "Learning Representations. 2024. Spotlight. "
            "OpenReview:pAsQSWlDUf. arXiv:2312.16424. "
            "DOI:10.48550/arXiv.2312.16424."
        ),
        (
            "[3] Zhang K, Lau W J, Ren S, Lin D, Chung J S, Sun C. "
            "PAI: Preserving Amplitude Information in Representation-"
            "Based Time-Series Anomaly Detection[J]. "
            "arXiv:2606.08935, 2026."
        ): (
            "[3] Zhang K, Lau W J, Ren S, et al. PAI: Preserving "
            "Amplitude Information in Representation-Based Time-Series "
            "Anomaly Detection[EB/OL]. arXiv preprint arXiv:2606.08935, "
            "2026: 1-15. DOI:10.48550/arXiv.2606.08935."
        ),
    }
    for old, new in reference_updates.items():
        matches = [
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.strip() == old
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one reference entry {old!r}, got {len(matches)}"
            )
        replace_text(matches[0], new)

    temporal_phrase = "以时间间隔 |t − t′| 为距离"
    temporal_matches = [
        paragraph
        for paragraph in document.paragraphs
        if temporal_phrase in paragraph.text
    ]
    if len(temporal_matches) != 1:
        raise RuntimeError(
            "Expected one temporal-distance sentence, "
            f"got {len(temporal_matches)}"
        )
    replace_text(
        temporal_matches[0],
        temporal_matches[0].text.replace(
            temporal_phrase,
            "以时间差 |t − t′| 为距离",
        ),
    )

    temporal_table_phrase = "按时间间隔给权重"
    table_matches = [
        paragraph
        for table in document.tables
        for row in table.rows
        for cell in row.cells
        for paragraph in cell.paragraphs
        if temporal_table_phrase in paragraph.text
    ]
    if len(table_matches) != 1:
        raise RuntimeError(
            "Expected one table temporal-distance phrase, "
            f"got {len(table_matches)}"
        )
    replace_text(
        table_matches[0],
        table_matches[0].text.replace(
            temporal_table_phrase,
            "按时间差给权重",
        ),
    )

    forbidden_terms = ("间隔", "双分数", "隔离森林")
    all_paragraphs = list(iter_all_paragraphs(document))
    for section in document.sections:
        all_paragraphs.extend(iter_all_paragraphs(section.header))
        all_paragraphs.extend(iter_all_paragraphs(section.footer))
    remaining = [
        (term, paragraph.text)
        for paragraph in all_paragraphs
        for term in forbidden_terms
        if term in paragraph.text
    ]
    if remaining:
        raise RuntimeError(f"Deprecated terminology remains: {remaining[:3]}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    copyfile(OUTPUT, VERSIONED_OUTPUT)
    print(OUTPUT)
    print(VERSIONED_OUTPUT)


def add_table_append(
    document: Document,
    rows: list[list[str]],
    widths: list[float],
    *,
    size: float,
) -> Table:
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    template = document.tables[9]
    table.style = template.style
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    template_borders = template._tbl.tblPr.find(
        qn("w:tblBorders")
    )
    existing_borders = table._tbl.tblPr.find(qn("w:tblBorders"))
    if existing_borders is not None:
        table._tbl.tblPr.remove(existing_borders)
    if template_borders is not None:
        table._tbl.tblPr.append(deepcopy(template_borders))
    for row_index, row_values in enumerate(rows):
        row = table.rows[row_index]
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if row_index == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))
        for column_index, value in enumerate(row_values):
            cell = row.cells[column_index]
            cell.width = Inches(widths[column_index])
            set_cell_text(
                cell,
                value,
                size=size,
                bold=row_index == 0,
                align=(
                    WD_ALIGN_PARAGRAPH.LEFT
                    if column_index == 0
                    else WD_ALIGN_PARAGRAPH.CENTER
                ),
            )
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    return table


if __name__ == "__main__":
    main()
