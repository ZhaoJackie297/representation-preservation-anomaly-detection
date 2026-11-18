from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "innovation_sim"
RETRAIN = ROOT / "processed_r42" / "innovation_sim"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def draw_relation() -> Path:
    relation = pd.read_csv(OUT / "quota_relationship_formula.csv")
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(38)
    text_font = font(25)
    small_font = font(21)
    left, right, top, bottom = 150, 1450, 150, 850
    max_q = 24

    draw.text(
        (left, 55),
        "候选预算分配关系 q(K)=round(0.4K)",
        font=title_font,
        fill="#111827",
    )
    draw.line((left, bottom, right, bottom), fill="#374151", width=3)
    draw.line((left, top, left, bottom), fill="#374151", width=3)
    draw.text(
        (left, bottom + 30),
        "每日新增审查预算 K",
        font=text_font,
        fill="#111827",
    )
    draw.text(
        (25, top),
        "IForest辅助名额 q",
        font=text_font,
        fill="#111827",
    )

    points = []
    for row in relation.itertuples(index=False):
        x = left + (int(row.k) / 50.0) * (right - left)
        y = bottom - (int(row.q) / max_q) * (bottom - top)
        points.append((x, y, int(row.k), int(row.q)))
        draw.text(
            (x - 22, bottom + 12),
            str(int(row.k)),
            font=small_font,
            fill="#111827",
        )
        draw.text(
            (left - 42, y - 12),
            str(int(row.q)),
            font=small_font,
            fill="#111827",
        )
    draw.line(
        [(point[0], point[1]) for point in points],
        fill="#1F4E78",
        width=5,
    )
    for x, y, k, q in points:
        draw.ellipse(
            (x - 9, y - 9, x + 9, y + 9),
            fill="#1F4E78",
            outline="white",
            width=2,
        )
        draw.text(
            (x - 25, y - 48),
            f"q={q}",
            font=small_font,
            fill="#1F4E78",
        )
    draw.text(
        (left, 925),
        "q(K)=min(max(1,round(0.4K)),floor(K/2))：TS占约60%，IForest占约40%",
        font=small_font,
        fill="#475569",
    )
    path = OUT / "quota_relationship_curve.png"
    image.save(path)
    return path


def draw_performance() -> Path:
    original = pd.read_csv(OUT / "quota_relationship_final_metrics.csv")
    retrained = pd.read_csv(RETRAIN / "r4.2_quota_retrain_metrics.csv")
    width, height = 2000, 950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(32)
    small_font = font(19)

    panels = [
        (
            original[
                (original["dataset"] == "r5.2")
                & (original["split"] == "external_test")
            ],
            "原始模型 r5.2 外部敏感性",
            80,
        ),
        (
            retrained[retrained["split"] == "validation"],
            "独立重训模型 r4.2 验证集",
            1040,
        ),
    ]
    colors = {
        "ts_only": "#5B9BD5",
        "iforest_only": "#ED7D31",
        "quota_model": "#70AD47",
    }
    labels = {
        "ts_only": "完全TS2Vec",
        "iforest_only": "完全IForest",
        "quota_model": "TS主+IF辅 0.4K",
    }
    for frame, title, panel_left in panels:
        panel_width = 860
        plot_left = panel_left + 90
        plot_right = panel_left + panel_width
        plot_top = 170
        plot_bottom = 790
        draw.text(
            (panel_left, 55),
            title,
            font=title_font,
            fill="#111827",
        )
        draw.line(
            (plot_left, plot_bottom, plot_right, plot_bottom),
            fill="#374151",
            width=3,
        )
        draw.line(
            (plot_left, plot_top, plot_left, plot_bottom),
            fill="#374151",
            width=3,
        )
        max_y = 0.9
        ks = [3, 5, 10, 20, 30, 50]
        for index, k in enumerate(ks):
            x = plot_left + index * (plot_right - plot_left) / 5
            draw.text(
                (x - 14, plot_bottom + 10),
                str(k),
                font=small_font,
                fill="#111827",
            )
        for y_tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
            y = plot_bottom - y_tick / max_y * (plot_bottom - plot_top)
            draw.line(
                (plot_left, y, plot_right, y),
                fill="#E5E7EB",
                width=1,
            )
            draw.text(
                (plot_left - 58, y - 10),
                f"{y_tick:.1f}",
                font=small_font,
                fill="#475569",
            )
        for method in ("ts_only", "iforest_only", "quota_model"):
            subset = (
                frame[frame["method"] == method]
                .set_index("k")
                .reindex(ks)
            )
            points = []
            for index, k in enumerate(ks):
                value = float(subset.loc[k, "event_recall_3d"])
                x = plot_left + index * (plot_right - plot_left) / 5
                y = plot_bottom - value / max_y * (plot_bottom - plot_top)
                points.append((x, y))
            draw.line(points, fill=colors[method], width=4)
            for x, y in points:
                draw.ellipse(
                    (x - 7, y - 7, x + 7, y + 7),
                    fill=colors[method],
                    outline="white",
                    width=2,
                )
        legend_y = 850
        for index, method in enumerate(
            ("ts_only", "iforest_only", "quota_model")
        ):
            x = plot_left + index * 245
            draw.line(
                (x, legend_y, x + 42, legend_y),
                fill=colors[method],
                width=5,
            )
            draw.text(
                (x + 52, legend_y - 15),
                labels[method],
                font=small_font,
                fill="#111827",
            )
    draw.text(
        (80, 910),
        "纵轴：K=3/5/10/20/30/50 下的3天事件召回率",
        font=small_font,
        fill="#475569",
    )
    path = OUT / "quota_relationship_performance.png"
    image.save(path)
    return path


def main() -> None:
    print(draw_relation())
    print(draw_performance())


if __name__ == "__main__":
    main()
