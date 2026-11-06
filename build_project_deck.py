"""Build the project overview deck for the vector-valued spiking neuron paper.

The deck is deliberately built from native PowerPoint shapes and charts so it
stays editable and does not depend on the manuscript figures that are still in
preparation.  Run with the bundled Python runtime (python-pptx 1.0.2).
"""

from __future__ import annotations

import math
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt


# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

PAPER = RGBColor(0xF7, 0xF4, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x14, 0x24, 0x2B)
INK_SOFT = RGBColor(0x3E, 0x4F, 0x55)
MUTED = RGBColor(0x6F, 0x7C, 0x7E)
TEAL = RGBColor(0x0E, 0x7C, 0x86)
TEAL_DARK = RGBColor(0x0A, 0x55, 0x5D)
TEAL_PALE = RGBColor(0xDD, 0xEC, 0xEC)
OCHRE = RGBColor(0xC4, 0x6A, 0x2D)
OCHRE_PALE = RGBColor(0xF1, 0xE2, 0xD3)
RED = RGBColor(0xA9, 0x3A, 0x2F)
RED_PALE = RGBColor(0xF4, 0xDF, 0xDA)
BLUE_GRAY = RGBColor(0x5F, 0x7F, 0x8A)
BORDER = RGBColor(0xD8, 0xD1, 0xC2)
GRID_LIGHT = RGBColor(0xE7, 0xE1, 0xD5)
DARK_BG = RGBColor(0x0D, 0x24, 0x2C)
DARK_GRID = RGBColor(0x1C, 0x43, 0x4C)

TITLE_FONT = "Georgia"
BODY_FONT = "Arial"
MATH_FONT = "Cambria Math"

SLIDE_W = 13.333
SLIDE_H = 7.5
MARGIN_L = 0.7
MARGIN_R = 0.7
CONTENT_W = SLIDE_W - MARGIN_L - MARGIN_R


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _style_run(run, size=14, color=INK, bold=False, italic=False, font=BODY_FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def add_text(
    slide,
    x,
    y,
    w,
    h,
    text,
    size=14,
    color=INK,
    bold=False,
    italic=False,
    font=BODY_FONT,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    line_spacing=1.0,
    space_after=0,
    margins=(0, 0, 0, 0),
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(margins[0])
    tf.margin_right = Inches(margins[1])
    tf.margin_top = Inches(margins[2])
    tf.margin_bottom = Inches(margins[3])
    lines = str(text).split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_before = Pt(0)
        p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = line
        _style_run(r, size=size, color=color, bold=bold, italic=italic, font=font)
    return box


def _add_math_runs(paragraph, text, size, color, bold, italic):
    """Emit runs with real subscripts/superscripts for `_x` and `^x` tokens."""
    i = 0
    buf = []

    def flush():
        if not buf:
            return
        run = paragraph.add_run()
        run.text = "".join(buf)
        _style_run(run, size=size, color=color, bold=bold, italic=italic, font=MATH_FONT)
        buf.clear()

    while i < len(text):
        ch = text[i]
        if ch in "_^" and i + 1 < len(text):
            flush()
            mode = ch
            i += 1
            if text[i] == "{":
                end = text.find("}", i)
                content = text[i + 1:end] if end != -1 else text[i + 1:]
                i = end + 1 if end != -1 else len(text)
            elif text[i] == "(":
                end = text.find(")", i)
                content = text[i:end + 1] if end != -1 else text[i:]
                i = end + 1 if end != -1 else len(text)
            else:
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] in ",+-"):
                    j += 1
                content = text[i:j]
                i = j
            if not content:
                continue
            run = paragraph.add_run()
            run.text = content
            _style_run(run, size=size * 0.72, color=color, bold=bold, italic=italic, font=MATH_FONT)
            rPr = run._r.get_or_add_rPr()
            rPr.set("baseline", "-25000" if mode == "_" else "30000")
        else:
            buf.append(ch)
            i += 1
    flush()


def add_math(
    slide,
    x,
    y,
    w,
    h,
    text,
    size=15,
    color=INK,
    bold=False,
    italic=False,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    line_spacing=1.2,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        _add_math_runs(p, line, size, color, bold, italic)
    return box


def add_bullets(slide, x, y, w, h, items, size=14, color=INK, bullet_color=TEAL, gap=6):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(0)
        p.space_after = Pt(gap)
        p.line_spacing = 1.05
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(0.24))))
        pPr.set("indent", str(-int(Inches(0.24))))
        bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
        pPr.append(bu)
        r = p.add_run()
        r.text = item
        _style_run(r, size=size, color=color)
        # Tint the bullet itself.
        buClr = pPr.makeelement(qn("a:buClr"), {})
        srgb = buClr.makeelement(qn("a:srgbClr"), {"val": "%02X%02X%02X" % (bullet_color[0], bullet_color[1], bullet_color[2])})
        buClr.append(srgb)
        pPr.insert(0, buClr)
    return box


def add_line(slide, x1, y1, x2, y2, color=BORDER, width=0.75, dash=None, arrow=False):
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    if dash is not None:
        conn.line.dash_style = dash
    if arrow:
        ln = conn.line._get_or_add_ln()
        tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        ln.append(tail)
    return conn


def add_card(slide, x, y, w, h, fill=WHITE, line=BORDER, radius=0.06, line_width=0.75):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line
    shp.line.width = Pt(line_width)
    try:
        shp.adjustments[0] = radius
    except Exception:
        pass
    try:
        shp.shadow.inherit = False
    except Exception:
        pass
    return shp


def add_rect(slide, x, y, w, h, fill, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    return shp


def add_chip(slide, x, y, w, h, text, fill=TEAL_PALE, color=TEAL_DARK, size=9, bold=True):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    try:
        shp.adjustments[0] = 0.25
    except Exception:
        pass
    tf = shp.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    _style_run(r, size=size, color=color, bold=bold)
    return shp


def add_number_circle(slide, x, y, d, n, fill=TEAL, color=WHITE, size=12):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    tf = shp.text_frame
    tf.clear()
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    _style_run(r, size=size, color=color, bold=True)
    return shp


def vector_grid(slide, x, y, w, h, rows=7, cols=7, gap=0.055, color=TEAL, pale=TEAL_PALE):
    cw = (w - gap * (cols - 1)) / cols
    ch = (h - gap * (rows - 1)) / rows
    for r in range(rows):
        for c in range(cols):
            cell = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(x + c * (cw + gap)),
                Inches(y + r * (ch + gap)),
                Inches(cw),
                Inches(ch),
            )
            cell.fill.solid()
            if (r * cols + c) % 11 == 0:
                cell.fill.fore_color.rgb = OCHRE
            elif (r + c) % 4 == 0:
                cell.fill.fore_color.rgb = pale
            elif (r * 3 + c) % 7 == 0:
                cell.fill.fore_color.rgb = TEAL_DARK
            else:
                cell.fill.fore_color.rgb = color
            cell.line.fill.background()
    return None


def spike_train(slide, x, y, w, h, color=INK_SOFT, ticks=18, seed=7):
    add_line(slide, x, y + h * 0.66, x + w, y + h * 0.66, color=color, width=0.8)
    # Deterministic irregular ticks; a schematic, not data.
    for i in range(ticks):
        t = (i * 0.61803398875 + 0.07 * seed) % 1.0
        xx = x + 0.04 + t * (w - 0.08)
        length = h * (0.30 if i % 3 else 0.44)
        add_line(slide, xx, y + h * 0.66 - length, xx, y + h * 0.66 + 0.02, color=color, width=1.1)


def add_footer(slide, page, dark=False, source=None):
    if dark:
        add_text(slide, MARGIN_L, 7.13, 8.2, 0.2, "Project overview  ·  November 2026", size=8, color=RGBColor(0x9F, 0xB3, 0xB4))
        add_text(slide, 12.0, 7.13, 0.65, 0.2, str(page), size=8, color=RGBColor(0x9F, 0xB3, 0xB4), align=PP_ALIGN.RIGHT)
        return
    add_line(slide, MARGIN_L, 7.09, SLIDE_W - MARGIN_R, 7.09, color=BORDER, width=0.6)
    left = "Neuron as vector  ·  Spiking Heidelberg Digits"
    add_text(slide, MARGIN_L, 7.14, 9.0, 0.2, left, size=8, color=MUTED)
    if source:
        add_text(slide, MARGIN_L, 6.88, 10.8, 0.18, source, size=7.5, color=MUTED)
    add_text(slide, 12.0, 7.14, 0.65, 0.2, str(page), size=8, color=MUTED, align=PP_ALIGN.RIGHT)


def add_header(slide, kicker, title, subtitle=None):
    add_rect(slide, MARGIN_L, 0.51, 0.11, 0.11, TEAL)
    add_text(slide, MARGIN_L + 0.21, 0.43, 8.0, 0.24, kicker.upper(), size=9, color=TEAL, bold=True)
    add_text(slide, MARGIN_L, 0.71, CONTENT_W, 0.8, title, size=28, color=INK, bold=True, font=TITLE_FONT)
    if subtitle:
        add_text(slide, MARGIN_L, 1.43, CONTENT_W, 0.32, subtitle, size=12.5, color=MUTED)
    add_line(slide, MARGIN_L, 1.63 if subtitle else 1.55, SLIDE_W - MARGIN_R, 1.63 if subtitle else 1.55, color=BORDER, width=0.7)


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def new_slide(prs, bg=PAPER):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = bg
    return slide


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def style_chart_base(chart, body_size=10):
    chart.has_title = False
    chart.has_legend = False
    chart.font.name = BODY_FONT
    chart.font.size = Pt(body_size)
    chart.font.color.rgb = INK_SOFT
    try:
        chart.format.fill.background()
    except Exception:
        pass
    try:
        chart.format.line.fill.background()
    except Exception:
        pass


def add_width_chart(slide, x, y, w, h):
    data = CategoryChartData()
    data.categories = ["4", "16", "64", "256", "1000"]
    data.add_series("TEST accuracy", (57.20, 63.74, 68.60, 77.61, 76.15))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), data)
    chart = gf.chart
    style_chart_base(chart, 10.5)
    plot = chart.plots[0]
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.font.size = Pt(10)
    dl.font.bold = True
    dl.font.color.rgb = INK
    dl.number_format = '0.00"%"'
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.ABOVE
    series = plot.series[0]
    series.smooth = False
    series.format.line.color.rgb = TEAL
    series.format.line.width = Pt(3.0)
    series.marker.style = XL_MARKER_STYLE.CIRCLE
    series.marker.size = 8
    series.marker.format.fill.solid()
    series.marker.format.fill.fore_color.rgb = TEAL
    series.marker.format.line.color.rgb = WHITE
    series.marker.format.line.width = Pt(1.5)
    va = chart.value_axis
    va.minimum_scale = 50
    va.maximum_scale = 85
    va.major_unit = 10
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = GRID_LIGHT
    va.major_gridlines.format.line.width = Pt(0.6)
    va.tick_labels.font.size = Pt(9.5)
    va.tick_labels.font.color.rgb = MUTED
    va.format.line.color.rgb = BORDER
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(10)
    ca.tick_labels.font.color.rgb = INK_SOFT
    ca.format.line.color.rgb = BORDER
    ca.major_tick_mark = XL_TICK_MARK.NONE
    return gf


def add_controls_chart(slide, x, y, w, h):
    data = CategoryChartData()
    data.categories = ["scalar N=100\n13k", "scalar N=1,411\n2.03M", "broadcast\n812k", "vector N=100\n2.14M"]
    data.add_series("TEST accuracy", (35.91, 33.13, 65.42, 79.64))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    chart = gf.chart
    style_chart_base(chart, 10.5)
    plot = chart.plots[0]
    plot.gap_width = 42
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.font.size = Pt(10.5)
    dl.font.bold = True
    dl.font.color.rgb = INK
    dl.number_format = '0.00"%"'
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    series = plot.series[0]
    colors = [OCHRE, OCHRE, BLUE_GRAY, TEAL]
    for p, c in zip(series.points, colors):
        p.format.fill.solid()
        p.format.fill.fore_color.rgb = c
        p.format.line.fill.background()
    va = chart.value_axis
    va.minimum_scale = 0
    va.maximum_scale = 90
    va.major_unit = 20
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = GRID_LIGHT
    va.major_gridlines.format.line.width = Pt(0.6)
    va.tick_labels.font.size = Pt(9.5)
    va.tick_labels.font.color.rgb = MUTED
    va.format.line.color.rgb = BORDER
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(10)
    ca.tick_labels.font.color.rgb = INK_SOFT
    ca.format.line.color.rgb = BORDER
    return gf


def add_augmentation_chart(slide, x, y, w, h):
    data = CategoryChartData()
    data.categories = ["event\ndropout", "jitter", "insertion", "channel\ndropout", "global\nshift"]
    data.add_series("delta", (1.68, 0.49, 0.35, -0.93, -2.96))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    chart = gf.chart
    style_chart_base(chart, 10.5)
    plot = chart.plots[0]
    plot.gap_width = 55
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.font.size = Pt(10)
    dl.font.bold = True
    dl.font.color.rgb = INK
    dl.number_format = '+0.00;-0.00'
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    series = plot.series[0]
    colors = [TEAL, RGBColor(0x9A, 0xA7, 0xA8), RGBColor(0x9A, 0xA7, 0xA8), OCHRE, RED]
    for p, c in zip(series.points, colors):
        p.format.fill.solid()
        p.format.fill.fore_color.rgb = c
        p.format.line.fill.background()
    va = chart.value_axis
    va.minimum_scale = -3.5
    va.maximum_scale = 2.5
    va.major_unit = 1
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = GRID_LIGHT
    va.major_gridlines.format.line.width = Pt(0.6)
    va.tick_labels.font.size = Pt(9.5)
    va.tick_labels.font.color.rgb = MUTED
    va.format.line.color.rgb = BORDER
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(9.5)
    ca.tick_labels.font.color.rgb = INK_SOFT
    ca.format.line.color.rgb = BORDER
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    return gf


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------


def slide_title(prs):
    s = new_slide(prs, DARK_BG)
    # Decorative vector grid on the right.
    vector_grid(s, 9.55, 1.45, 3.1, 3.1, rows=8, cols=8, gap=0.055, color=DARK_GRID, pale=RGBColor(0x27, 0x59, 0x62))
    add_text(s, 9.55, 4.72, 3.2, 0.25, "one neuron  ·  1,000 coordinates", size=8.5, color=RGBColor(0x9F, 0xB3, 0xB4))
    add_text(s, 0.8, 0.68, 5.0, 0.25, "PROJECT OVERVIEW  ·  2026", size=9.5, color=TEAL_PALE, bold=True)
    add_text(s, 0.8, 1.45, 8.4, 1.95, "What a neuron's\ninternal state buys", size=44, color=WHITE, bold=True, font=TITLE_FONT, line_spacing=0.95)
    add_text(s, 0.82, 3.62, 8.0, 0.8, "A vector-valued spiking neuron on a standard\nevent-based benchmark", size=18, color=RGBColor(0xC9, 0xD6, 0xD6), line_spacing=1.15)
    add_text(s, 0.82, 4.72, 7.5, 0.3, "Mehran Jaberinezhad  ·  November 2026", size=12, color=RGBColor(0xE8, 0xEF, 0xEF))
    chips = [("N = 100", 0.82, 5.55, 1.15), ("D = 1000", 2.12, 5.55, 1.3), ("SHD  ·  20 classes", 3.57, 5.55, 1.95)]
    for text, x, y, w in chips:
        add_chip(s, x, y, w, 0.34, text, fill=RGBColor(0x16, 0x3A, 0x42), color=RGBColor(0xD9, 0xE7, 0xE6), size=9)
    add_text(s, 0.82, 6.7, 8.0, 0.25, "Native diagrams and charts; final manuscript figures in preparation.", size=8.5, color=RGBColor(0x8F, 0xA7, 0xA9))
    add_footer(s, 1, dark=True)
    set_notes(
        s,
        "This deck is a project-level overview built from the current Introduction, Methods and Results drafts. "
        "The visuals are native PowerPoint shapes and charts so the deck is editable before the final manuscript figures land. "
        "The main result is 88.07% (1994/2264) on the official SHD test set; all numbers are tagged confirmed or screening on the relevant slides.",
    )


def slide_scalar(prs):
    s = new_slide(prs)
    add_header(s, "01 · The starting point", "Almost every trainable spiking neuron is a scalar")
    add_text(s, 0.7, 1.92, 6.35, 0.3, "THE POINT NEURON", size=9, color=TEAL, bold=True)
    add_text(
        s,
        0.7,
        2.28,
        6.25,
        2.55,
        "A spike is one bit: fire or not.\n"
        "The state is one number: the membrane potential V.\n"
        "It integrates input and leaks. It can remember how much excitation arrived; "
        "it cannot hold an input vector whose coordinates vary independently.",
        size=16,
        color=INK,
        line_spacing=1.22,
    )
    add_card(s, 7.35, 1.92, 5.25, 3.25, fill=WHITE)
    add_text(s, 7.65, 2.15, 4.6, 0.25, "LEAKY INTEGRATE-AND-FIRE", size=9, color=MUTED, bold=True)
    add_math(s, 7.65, 2.55, 4.6, 0.5, "τ dV/dt  =  −V + I(t)", size=22, color=INK)
    add_math(s, 7.65, 3.18, 4.6, 0.45, "V  ←  (1 − α)V + αI", size=20, color=INK)
    add_math(s, 7.65, 3.78, 4.6, 0.45, "s  =  H(V − θ)", size=20, color=INK)
    add_text(s, 7.65, 4.5, 4.55, 0.35, "one scalar state  ·  one threshold  ·  one output bit", size=9.5, color=MUTED)
    # single-coordinate glyph
    add_rect(s, 12.0, 2.12, 0.09, 0.9, OCHRE_PALE)
    add_rect(s, 12.0, 2.12, 0.09, 0.36, OCHRE)
    add_text(s, 12.14, 2.16, 0.4, 0.2, "V", size=9, color=OCHRE, bold=True)
    add_card(s, 0.7, 5.45, 11.9, 1.05, fill=OCHRE_PALE, line=OCHRE_PALE)
    add_text(
        s,
        0.95,
        5.63,
        11.45,
        0.7,
        "The biological record is richer — dendrites, adaptation, population codes. "
        "The question is not whether that richness exists, but whether a spiking unit needs it to compute.",
        size=13,
        color=INK_SOFT,
        italic=True,
        line_spacing=1.15,
    )
    add_footer(s, 2, source="Baseline used here omits reset and refractory period (Methods §2.3).")
    set_notes(
        s,
        "The scalar abstraction is deliberately modest: one state variable, one threshold, one output bit. "
        "The LIF equation is shown for orientation; the model in this paper keeps the same scalar case at D=1. "
        "The biological motivation is not evidence that vector states are necessary; it motivates the controlled question.",
    )


def slide_question(prs):
    s = new_slide(prs)
    add_header(s, "01 · The question", "If the state is a vector, what does width buy?")
    # scalar panel
    add_card(s, 0.7, 1.95, 5.55, 3.95, fill=WHITE)
    add_text(s, 0.98, 2.15, 4.8, 0.25, "SCALAR STATE", size=9, color=OCHRE, bold=True)
    add_math(s, 0.98, 2.5, 4.8, 0.55, "V ∈ ℝ", size=30, color=INK)
    add_rect(s, 3.05, 3.2, 0.12, 1.4, OCHRE_PALE)
    add_rect(s, 3.05, 3.78, 0.12, 0.24, OCHRE)
    add_text(s, 3.28, 3.72, 2.2, 0.25, "one coordinate", size=11, color=MUTED)
    add_text(s, 0.98, 4.85, 4.85, 0.65, "One leak, one threshold, one number to read. Width is fixed at 1.", size=12.5, color=INK_SOFT, line_spacing=1.15)
    # vector panel
    add_card(s, 6.55, 1.95, 6.1, 3.95, fill=WHITE)
    add_text(s, 6.85, 2.15, 5.3, 0.25, "VECTOR STATE", size=9, color=TEAL, bold=True)
    add_math(s, 6.85, 2.5, 5.3, 0.55, "z ∈ ℝ^D", size=30, color=INK)
    vector_grid(s, 9.8, 2.55, 1.65, 1.65, rows=6, cols=6, gap=0.045)
    add_text(s, 6.85, 3.95, 2.75, 0.55, "D = 1000\ncoordinates", size=13, color=INK, bold=True, line_spacing=1.15)
    add_text(s, 9.75, 4.35, 2.55, 0.3, "all of them\nin the dynamics", size=10.5, color=MUTED, line_spacing=1.1)
    add_text(s, 6.85, 4.85, 5.4, 0.65, "A shared operator mixes coordinates; a spike gates the whole vector into the population.", size=12.5, color=INK_SOFT, line_spacing=1.15)
    # controlled-variable strip
    add_card(s, 0.7, 6.1, 11.95, 0.55, fill=TEAL_DARK, line=TEAL_DARK, radius=0.04)
    add_text(
        s,
        0.95,
        6.19,
        11.45,
        0.3,
        "Hold N, readout, optimizer, data and budget fixed.  Vary only D.  At D = 1 the same code is the scalar case.",
        size=11.5,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    add_footer(s, 3)
    set_notes(
        s,
        "The controlled variable is D, the width of one neuron's state, not N, the number of neurons. "
        "The same code path runs at D=1; the scalar limit is a setting, not a separate implementation. "
        "This is the design that makes a width sweep interpretable.",
    )


def slide_architecture(prs):
    s = new_slide(prs)
    add_header(s, "02 · The model", "One neuron, one vector, one spike")
    # top pipeline boxes
    y = 1.93
    h = 1.38
    boxes = [
        (0.7, 2.25, "SPIKING HEIDELBERG DIGITS", "700 channels · ~1 s", "event stream (channel, time)"),
        (3.18, 1.55, "2 ms bins", "550 bins", "500 signal + 50 silence"),
        (4.95, 1.8, "input projection", "1400 × 1000", "2 delay-line copies"),
        (6.95, 3.45, "100 vector neurons", "z_i ∈ ℝ^1000", "rank-64 mixing inside"),
        (10.62, 2.0, "learned hyperplane", "⟨z_i, w_out⟩ + θ_i", "spike → counts → 20 classes"),
    ]
    for x, w, top, mid, bottom in boxes:
        add_card(s, x, y, w, h, fill=WHITE)
        if top == "100 vector neurons":
            add_text(s, x + 0.15, y + 0.12, w - 0.3, 0.2, top, size=9.5, color=TEAL, bold=True)
            vector_grid(s, x + 0.18, y + 0.38, 0.95, 0.95, rows=5, cols=5, gap=0.04)
            add_math(s, x + 1.28, y + 0.36, w - 1.35, 0.3, mid, size=14, color=INK, bold=True)
            add_text(s, x + 1.28, y + 0.68, w - 1.35, 0.35, bottom, size=8.5, color=MUTED)
            add_text(s, x + 1.28, y + 1.0, w - 1.35, 0.2, "shared mixing · vector broadcast", size=8, color=MUTED)
        else:
            add_text(s, x + 0.15, y + 0.12, w - 0.3, 0.2, top, size=8.2, color=TEAL, bold=True)
            if top == "learned hyperplane":
                add_math(s, x + 0.15, y + 0.38, w - 0.3, 0.55, mid, size=10.5, color=INK, bold=True, line_spacing=1.1)
            else:
                add_text(s, x + 0.15, y + 0.42, w - 0.3, 0.3, mid, size=15, color=INK, bold=True)
            add_text(s, x + 0.15, y + 1.0, w - 0.3, 0.25, bottom, size=8.5, color=MUTED)
    # arrows between boxes
    arrows = [(2.95, 3.18), (4.73, 4.95), (6.75, 6.95), (10.4, 10.62)]
    for x1, x2 in arrows:
        add_line(s, x1, y + h / 2, x2 - 0.03, y + h / 2, color=TEAL, width=1.6, arrow=True)
    # two bottom explanatory cards
    add_card(s, 0.7, 3.65, 6.15, 2.55, fill=WHITE)
    add_text(s, 0.95, 3.82, 5.6, 0.25, "WHY IT STAYS TRACTABLE", size=9, color=TEAL, bold=True)
    add_text(s, 0.95, 4.15, 5.55, 1.8,
             "•  A shared input matrix is 1400 × 1000, not one per neuron.\n"
             "•  Rank-64 mixing replaces a per-neuron D × D matrix.\n"
             "•  One leak, one bias and one readout vector per neuron.",
             size=12.5, color=INK_SOFT, line_spacing=1.2, space_after=4)
    add_card(s, 7.1, 3.65, 5.55, 2.55, fill=WHITE)
    add_text(s, 7.35, 3.82, 5.0, 0.25, "WHAT TRAVELS BETWEEN NEURONS", size=9, color=TEAL, bold=True)
    add_math(s, 7.35, 4.16, 5.0, 0.35, "h_t = (1/√N)(Σ_j s_j z_j) W_emit", size=14, color=INK)
    add_math(s, 7.35, 4.58, 5.0, 0.35, "c_i(t) = Σ_j W_rec[j,i] s_j(t)", size=13, color=INK)
    add_text(s, 7.35, 5.05, 5.0, 0.7, "A spike is still one bit. It gates a whole vector; the receiving weight stays scalar.", size=11.5, color=INK_SOFT, line_spacing=1.15)
    # bottom stats
    chips = [("N = 100", 0.7, 2.3), ("D = 1000", 3.12, 2.55), ("rank = 64", 5.8, 2.25), ("3,069,201 parameters", 8.2, 4.1)]
    for text, cx, cw in chips:
        add_chip(s, cx, 6.42, cw, 0.34, text, fill=TEAL_PALE, color=TEAL_DARK, size=9)
    add_footer(s, 4)
    set_notes(
        s,
        "The architecture is deliberately a chain: event stream, binning, one shared input projection, 100 vector neurons, and a count readout. "
        "The state is D=1000 per neuron; N=100; the mixing rank is 64. "
        "The bottom-right equation is the key departure from a scalar population: a spike is a bit, but it gates the spiking neuron's state vector into the shared signal h_t.",
    )


def slide_where_vector(prs):
    s = new_slide(prs)
    add_header(s, "02 · Where the vector goes", "Three places the vector changes the computation")
    cards = [
        (0.7, "01", "Input", "a_t = x_t W_in\nd_i^in = g_i ⊙ a_t", "The 700-channel event is projected into 1,000 dimensions. A per-neuron gate makes the projection specific."),
        (4.65, "02", "Broadcast", "h_t = (1/√N)(Σ_j s_j z_j) W_emit\nd_i^rec = c_i(t) h_t", "A spike is one bit. It enters the population as a gated state vector, scaled by an ordinary scalar synapse."),
        (8.6, "03", "Spike", "o_i = ⟨z_i, w_out,i⟩ + θ_i − w_ad,i b_i\ns_i = H(o_i)", "The threshold is a learned hyperplane over all coordinates, not a designated coordinate crossing a value."),
    ]
    for x, num, label, formula, body in cards:
        add_card(s, x, 1.95, 3.75, 4.05, fill=WHITE)
        add_text(s, x + 0.22, 2.1, 0.9, 0.55, num, size=26, color=TEAL_PALE, bold=True, font=TITLE_FONT)
        add_text(s, x + 0.24, 2.55, 3.2, 0.3, label.upper(), size=10, color=TEAL, bold=True)
        add_math(s, x + 0.24, 2.88, 3.3, 0.95, formula, size=12.5, color=INK, line_spacing=1.25)
        add_text(s, x + 0.24, 3.95, 3.25, 1.35, body, size=12.5, color=INK_SOFT, line_spacing=1.18)
    row_chips = [("TABLE 2 · ROW 1", 0.94), ("TABLE 2 · ROW 3", 4.89), ("TABLE 2 · ROW 7", 8.84)]
    for label, cx in row_chips:
        add_chip(s, cx, 5.48, 1.75, 0.28, label, fill=TEAL_PALE, color=TEAL_DARK, size=7.5)
    add_card(s, 0.7, 6.35, 11.65, 0.48, fill=TEAL_PALE, line=TEAL_PALE, radius=0.04)
    add_text(s, 0.95, 6.43, 11.15, 0.3, "Rows 1, 3 and 7 of the step-by-step comparison (Table 2). Everything else is scalar or a consequence.", size=10.5, color=TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)
    add_footer(s, 5)
    set_notes(
        s,
        "This is the slide to use when someone asks exactly where the vector is inserted. "
        "First, the input is projected into the vector space and gated per neuron. "
        "Second, a spike gates the whole state vector into the population signal; the synapse itself is still scalar. "
        "Third, the spike threshold is a learned linear functional of the whole state. "
        "The readout is deliberately unchanged, so width is the controlled variable.",
    )


def slide_equations(prs):
    s = new_slide(prs)
    add_header(s, "02 · The dynamics", "The update, in four lines")
    cards = [
        (0.7, 1.9, "1 · INPUT", "a_t = x_t W_in\nd_i^in = g_i ⊙ a_t", "shared projection, per-neuron gate"),
        (6.75, 1.9, "2 · MIXING", "m_i = (z_i W_down) W_up\nm_i^(2) = (z_i W_down^(2)) W_up^(2)", "rank 64 inside each neuron"),
        (0.7, 4.0, "3 · BROADCAST", "h_t = (1/√N)(Σ_j s_j z_j) W_emit\nc_i(t) = Σ_j W_rec[j,i] s_j(t)", "a spike gates a whole vector"),
        (6.75, 4.0, "4 · STATE + SPIKE", "z_i(t+1) = (1−α_i) z_i(t) + α_i z̃_i\nz̃_i = tanh(m_i + λ_i tanh(m_i^(2) + d_i))\no_i = ⟨z_i, w_out,i⟩ + θ_i − w_ad,i b_i\ns_i = H(o_i)", "leak, two-stage target, hyperplane threshold"),
    ]
    for x, y, label, formula, note in cards:
        add_card(s, x, y, 5.9, 1.85, fill=WHITE)
        add_text(s, x + 0.22, y + 0.16, 5.4, 0.22, label, size=9, color=TEAL, bold=True)
        add_math(s, x + 0.22, y + 0.48, 5.45, 1.1, formula, size=12.0 if label.startswith("4") else 15.5, color=INK, line_spacing=1.2)
        add_text(s, x + 0.22, y + 1.62, 5.4, 0.2, note, size=8.2, color=MUTED)
    add_math(s, 0.7, 6.1, 11.9, 0.3, "d_i = d_i^in + d_i^rec + β_i.  Full derivation and the scalar limit: Methods §2.4–2.5.", size=10.5, color=MUTED, italic=True)
    add_footer(s, 6, source="Equations simplified for the slide; adLIF currents are omitted because they are inert in the reported run (Results §3.4).")
    set_notes(
        s,
        "This is not the full Methods; it is the four equations needed to follow the talk. "
        "The two-stage term is the reported target: m plus lambda times a tanh of a second rank-64 mix plus drive. "
        "The adLIF currents are omitted because they are configured but functionally inert under the two-stage branch; the Methods and Results both state this.",
    )


def slide_training(prs):
    s = new_slide(prs)
    add_header(s, "02 · Training", "What it takes to train a 1,000-dimensional neuron")
    chips = [("3.07M parameters", 0.7, 2.05), ("5,055 MiB peak", 2.95, 1.85), ("~505 s / epoch", 4.95, 1.75)]
    for text, cx, cw in chips:
        add_chip(s, cx, 1.78, cw, 0.3, text, fill=TEAL_PALE, color=TEAL_DARK, size=8.5)
    # pipeline
    steps = [
        (0.7, 2.15, "SHD event stream", "700 channels · 20 classes"),
        (3.18, 1.95, "2 ms bins", "550 bins total"),
        (5.37, 2.1, "100 × 1000 state", "3.07M parameters"),
        (7.7, 2.0, "BPTT + checkpointing", "10 chunks per sequence"),
        (9.95, 2.25, "count readout", "20-class logits"),
    ]
    for x, w, top, bottom in steps:
        add_card(s, x, 2.2, w, 0.85, fill=WHITE)
        add_text(s, x + 0.12, 2.32, w - 0.24, 0.25, top, size=11.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
        add_text(s, x + 0.12, 2.62, w - 0.24, 0.2, bottom, size=8, color=MUTED, align=PP_ALIGN.CENTER)
    for x1, x2 in [(2.75, 3.18), (5.13, 5.37), (7.47, 7.7), (9.7, 9.95)]:
        add_line(s, x1, 2.62, x2 - 0.03, 2.62, color=TEAL, width=1.4, arrow=True)
    # three cards
    cards = [
        (0.7, 3.45, "DATA", "Spiking Heidelberg Digits (SHD)\n20 classes: digits 0–9, English + German\n8,156 train · 2,264 test\nclass-stratified 90/10 FIT/VAL\nspeakers 4 & 5 never seen in training"),
        (4.35, 3.45, "OPTIMIZATION", "Adam, lr 1e-3, no weight decay\ncosine schedule, 3% warmup\ngradient norm clipped at 1.0\n30 epochs · batch 96 · seed 0\nmanual fp16 · fp32 masters\nSLAYER surrogate, α = 2"),
        (8.0, 3.45, "REGULARIZATION", "event dropout p = 0.1 (raw events)\nsame-class blend: p = 0.5, keep 0.5\n100 ms silence tail\nthresholds calibrated to 10 Hz\naugmentation only in training"),
    ]
    for x, y, label, body in cards:
        add_card(s, x, y, 4.0, 2.35, fill=WHITE)
        add_text(s, x + 0.22, y + 0.16, 3.6, 0.22, label, size=9, color=TEAL, bold=True)
        add_text(s, x + 0.22, y + 0.48, 3.55, 1.78, body, size=10.5, color=INK_SOFT, line_spacing=1.16)
    add_text(s, 0.7, 6.25, 11.9, 0.3, "One configuration → one registry row → one run directory.  The config file is the authoritative record.", size=9.5, color=MUTED, italic=True)
    add_footer(s, 7, source="Timing: 505 s/epoch and 5,055 MiB peak from the final 30-epoch run (Results §3.3).")
    set_notes(
        s,
        "The protocol is intentionally standard: SHD event stream, 2 ms bins, 500 signal bins plus 50 silence bins, count readout. "
        "Training uses BPTT with ten checkpointed chunks, manual fp16 state with fp32 master weights, and the SLAYER surrogate. "
        "The data split is a class-stratified 90/10 split of the official training file; the official test file is untouched for training. "
        "Speakers 4 and 5 are never in training, which is why the unseen-speaker number matters.",
    )


def stat_card(slide, x, y, w, h, value, label, sub, accent=TEAL, value_size=25):
    add_card(slide, x, y, w, h, fill=WHITE)
    add_rect(slide, x, y, 0.07, h, accent)
    add_text(slide, x + 0.2, y + 0.18, w - 0.35, 0.2, label.upper(), size=8.5, color=accent, bold=True)
    add_text(slide, x + 0.2, y + 0.5, w - 0.35, 0.6, value, size=value_size, color=INK, bold=True)
    add_text(slide, x + 0.2, y + 1.25, w - 0.35, 0.75, sub, size=10, color=MUTED, line_spacing=1.15)


def slide_results_glance(prs):
    s = new_slide(prs)
    add_header(s, "03 · Results", "What the width buys, in one slide")
    stat_card(s, 0.7, 1.95, 2.8, 2.7, "49.43 → 82.51%", "Scalar → vector", "N = 64 · 20 epochs\n+33.1 percentage points", accent=OCHRE, value_size=20)
    stat_card(s, 3.78, 1.95, 2.8, 2.7, "88.07%", "Final model", "1994 / 2264 · 30 epochs\nseed 0 · 3.07M parameters", accent=TEAL, value_size=31)
    stat_card(s, 6.86, 1.95, 2.8, 2.7, "85.87%", "Unseen speakers", "1,840 trials from speakers 4 & 5\nseen speakers: 97.64%", accent=TEAL, value_size=31)
    stat_card(s, 9.94, 1.95, 2.65, 2.7, "7.49 / 100k", "Effective dimensions", "participation ratio\n9 components → 90% variance", accent=BLUE_GRAY, value_size=24)
    add_card(s, 0.7, 5.0, 11.9, 1.35, fill=TEAL_DARK, line=TEAL_DARK, radius=0.04)
    add_text(s, 0.98, 5.15, 11.3, 0.3, "THE SHORT VERSION", size=9, color=RGBColor(0xA9, 0xD3, 0xD2), bold=True)
    add_text(s, 0.98, 5.45, 11.3, 0.75, "The gap is not a parameter-count effect. A 2M-parameter scalar reaches 33.13% (5 ep); a 1000-wide broadcast recovers 29.5 of the 43.7 pp, leaving 14.2 pp for state width.", size=12.5, color=WHITE, line_spacing=1.18)
    add_footer(s, 8, source="Confirmed = 20/30 epochs; screening = 5 epochs; single seed; SHD test n = 2,264.")
    set_notes(
        s,
        "One-slide preview. The scalar-to-vector pair is the cleanest controlled comparison: N=64, 20 epochs, 2,264 test samples, 49.43% vs 82.51%. "
        "The final model is 88.07% at N=100 and 30 epochs. "
        "The unseen-speaker split is the closest thing to a generalisation estimate in this evaluation. "
        "The effective-dimension number is from the frozen final checkpoint: the population mean uses 7.49 effective dimensions out of 100,000.",
    )


def slide_width(prs):
    s = new_slide(prs)
    add_header(s, "03 · Width", "Width helps, then saturates")
    add_card(s, 0.7, 1.85, 8.55, 4.65, fill=WHITE)
    add_width_chart(s, 0.95, 2.05, 8.05, 4.2)
    add_text(s, 0.95, 6.27, 8.0, 0.2, "state width D  ·  fourfold steps", size=8.5, color=MUTED, align=PP_ALIGN.CENTER)
    # right rail
    add_card(s, 9.5, 1.85, 3.15, 4.65, fill=WHITE)
    add_text(s, 9.75, 2.05, 2.7, 0.25, "AT A FIXED N = 64", size=9, color=TEAL, bold=True)
    add_text(s, 9.75, 2.27, 2.7, 0.2, "+20.4 pp from D=4 to 256", size=8.5, color=TEAL, bold=True)
    rail = [("D = 4", "57.20%"), ("D = 16", "63.74%"), ("D = 64", "68.60%"), ("D = 256", "77.61%"), ("D = 1000", "76.15%")]
    for i, (a, b) in enumerate(rail):
        yy = 2.48 + i * 0.47
        add_text(s, 9.75, yy, 1.05, 0.25, a, size=10.5, color=INK_SOFT)
        add_text(s, 10.85, yy, 1.3, 0.25, b, size=11, color=INK, bold=True, align=PP_ALIGN.RIGHT)
    add_line(s, 9.75, 4.9, 12.35, 4.9, color=BORDER, width=0.6)
    add_text(s, 9.75, 5.02, 2.7, 0.3, "D = 256 → 1000", size=9, color=OCHRE, bold=True)
    add_text(s, 9.75, 5.3, 2.7, 0.4, "1.5 pp, within noise", size=9, color=INK_SOFT)
    add_text(s, 9.75, 5.68, 2.7, 0.3, "5-epoch screening", size=9.5, color=OCHRE, bold=True)
    add_text(s, 9.75, 5.98, 2.7, 0.45, "D=1000 reaches 82.51% at 20 epochs.", size=8.5, color=MUTED, line_spacing=1.05)
    add_footer(s, 9, source="Screening runs: N=64, same readout, optimizer and 5-epoch budget; SHD test n = 2,264.")
    set_notes(
        s,
        "The width sweep is monotone from D=4 to D=256: 57.20 to 77.61, +20.4 percentage points. "
        "D=1000 is 76.15 at five epochs, which is 1.5 pp below D=256 and inside the noise band; do not read it as a win for the smaller width. "
        "This is a screening experiment, not a headline result. The 20-epoch D=1000 reference is 82.51%.",
    )


def slide_controls(prs):
    s = new_slide(prs)
    add_header(s, "03 · Controls", "The gap is not a parameter-count effect")
    add_card(s, 0.7, 1.85, 8.4, 4.7, fill=WHITE)
    add_controls_chart(s, 0.95, 2.05, 7.9, 4.3)
    # right rail
    add_card(s, 9.35, 1.85, 3.3, 4.7, fill=WHITE)
    add_text(s, 9.6, 2.05, 2.8, 0.25, "DECOMPOSING THE GAP", size=9, color=TEAL, bold=True)
    rows = [("43.7 pp", "scalar → vector at N=100, 5 ep"), ("29.5 pp", "recovered by a 1000-wide broadcast"), ("14.2 pp", "left for the width of the state")]
    for i, (big, small) in enumerate(rows):
        yy = 2.5 + i * 1.05
        add_text(s, 9.6, yy, 2.8, 0.45, big, size=22, color=TEAL if i < 2 else OCHRE, bold=True)
        add_text(s, 9.6, yy + 0.45, 2.8, 0.45, small, size=9, color=MUTED, line_spacing=1.1)
    add_text(s, 9.6, 5.78, 2.75, 0.55, "The broadcast control uses 38% of the vector model's parameters.", size=8.5, color=MUTED, italic=True, line_spacing=1.1)
    add_text(s, 0.7, 6.68, 11.9, 0.25, "At N=64 and 20 epochs: scalar 49.43% vs vector 82.51% (+33.1 pp).", size=9.5, color=INK_SOFT, italic=True)
    add_footer(s, 10, source="Controls are untuned size/parameter-matched ablations, not competitive baselines (Results §3.2, §3.8).")
    set_notes(
        s,
        "The capacity-matched scalar is the key control: 1,411 scalar neurons, 2.03M parameters, 33.13% at five epochs. "
        "A scalar with 100 neurons is 35.91%; a broadcast model with a scalar state but a 1,000-wide transmission is 65.42%; the full vector model is 79.64%. "
        "So a 1,000-wide transmission recovers 29.5 of the 43.7 pp gap with 38% of the parameters; the remaining 14.2 pp is the state width itself. "
        "The scalar controls are untuned ablations, not competitive baselines.",
    )


def metric_row(slide, x, y, w, label, value, color=TEAL, max_value=100, note=None):
    add_text(slide, x, y, 1.8, 0.22, label, size=9.5, color=INK_SOFT, bold=True)
    add_text(slide, x + 1.85, y - 0.03, 1.0, 0.25, f"{value:.2f}%", size=11, color=INK, bold=True, align=PP_ALIGN.RIGHT)
    track_x = x + 3.0
    track_w = w - 3.0
    add_rect(slide, track_x, y + 0.04, track_w, 0.12, RGBColor(0xEC, 0xE7, 0xDC))
    add_rect(slide, track_x, y + 0.04, max(0.02, track_w * value / max_value), 0.12, color)
    if note:
        add_text(slide, x, y + 0.2, w, 0.2, note, size=8, color=MUTED)


def slide_headline(prs):
    s = new_slide(prs)
    add_header(s, "03 · The trained model", "88.07% on the official SHD test")
    add_text(s, 0.72, 1.95, 5.2, 0.3, "FINAL MODEL · 30 EPOCHS · SEED 0", size=9, color=TEAL, bold=True)
    add_text(s, 0.68, 2.28, 5.4, 1.35, "88.07%", size=78, color=INK, bold=True, font=TITLE_FONT)
    add_text(s, 0.78, 3.55, 4.9, 0.35, "1994 / 2264 correct", size=16, color=INK, bold=True)
    add_text(s, 0.78, 3.95, 5.0, 0.55, "final-epoch weights, not the best epoch", size=10.5, color=MUTED, italic=True)
    add_text(s, 0.78, 4.65, 5.1, 1.0, "The best epoch is 88.74% (2009/2264). The final-epoch number is the one quoted in the manuscript.", size=10.5, color=INK_SOFT, line_spacing=1.15)
    # metric card
    add_card(s, 6.4, 1.95, 6.25, 3.95, fill=WHITE)
    add_text(s, 6.65, 2.1, 5.7, 0.25, "FIT / VAL / TEST", size=9, color=TEAL, bold=True)
    metric_row(s, 6.65, 2.55, 5.75, "FIT", 99.78, color=RGBColor(0x9A, 0xA7, 0xA8), note="training split, 7,340 samples")
    metric_row(s, 6.65, 3.12, 5.75, "VAL", 97.67, color=RGBColor(0x9A, 0xA7, 0xA8), note="speaker-overlapping random 10% of train")
    metric_row(s, 6.65, 3.69, 5.75, "TEST", 88.07, color=TEAL, note="official test, 2,264 samples, speakers 4 & 5 unseen")
    add_line(s, 6.65, 4.48, 12.4, 4.48, color=BORDER, width=0.6)
    add_text(s, 6.65, 4.62, 2.7, 0.25, "UNSEEN SPEAKERS", size=8.5, color=MUTED, bold=True)
    add_text(s, 6.65, 4.9, 2.7, 0.35, "85.87%", size=18, color=TEAL, bold=True)
    add_text(s, 6.65, 5.28, 2.7, 0.2, "1,840 trials", size=8, color=MUTED)
    add_text(s, 9.65, 4.62, 2.7, 0.25, "SEEN SPEAKERS", size=8.5, color=MUTED, bold=True)
    add_text(s, 9.65, 4.9, 2.7, 0.35, "97.64%", size=18, color=INK, bold=True)
    add_text(s, 9.65, 5.28, 2.7, 0.2, "424 trials", size=8, color=MUTED)
    # bottom chips
    chips = [("30 epochs", 0.7, 1.4), ("3,069,201 parameters", 2.25, 2.8), ("5,055 MiB peak", 5.2, 2.0), ("~505 s / epoch", 7.35, 1.9)]
    for text, cx, cw in chips:
        add_chip(s, cx, 6.08, cw, 0.32, text, fill=TEAL_PALE, color=TEAL_DARK, size=8.5)
    add_footer(s, 11, source="Confirmed 30-epoch run. VAL is not a generalisation estimate here (Results §3.3, §3.8).")
    set_notes(
        s,
        "The headline is 1994/2264 = 88.07% on the official test set, final epoch, seed 0, 3,069,201 parameters. "
        "The best epoch is 2009/2264 = 88.74%; the manuscript quotes final-epoch weights for consistency with Methods. "
        "The unseen-speaker number is 85.87% on 1,840 trials from speakers 4 and 5; the 424 seen-speaker trials are 97.64%. "
        "VAL is a speaker-overlapping split, so it is a monitoring signal, not a generalisation estimate.",
    )


def add_dumbbell(slide, x, y, w, h):
    # x range -2 .. 6 pp
    x0 = x + 1.55
    x1 = x + w - 0.3
    def px(v):
        return x0 + (v + 2.0) / 8.0 * (x1 - x0)
    rows = [
        ("M1  adaptation", 5.70, -0.31),
        ("M2  learned z0 + α", 0.71, 2.69),
        ("M4  input delays", 1.99, -1.37),
    ]
    y_top = y + 0.95
    y_gap = 1.05
    # zero line
    add_line(slide, px(0), y + 0.55, px(0), y + h - 0.55, color=BORDER, width=0.8, dash=None)
    # axis
    add_line(slide, x0, y + h - 0.45, x1, y + h - 0.45, color=BORDER, width=0.7)
    for tick in [-2, 0, 2, 4, 6]:
        add_line(slide, px(tick), y + h - 0.45, px(tick), y + h - 0.38, color=BORDER, width=0.7)
        add_text(slide, px(tick) - 0.25, y + h - 0.32, 0.5, 0.2, f"{tick:+d}" if tick != 0 else "0", size=8, color=MUTED, align=PP_ALIGN.CENTER)
    add_text(slide, x1 - 2.0, y + h - 0.1, 2.0, 0.2, "Δ TEST accuracy (pp)", size=8, color=MUTED, align=PP_ALIGN.RIGHT)
    for i, (label, a, b) in enumerate(rows):
        yy = y_top + i * y_gap
        add_text(slide, x + 0.15, yy - 0.12, 1.25, 0.25, label, size=9.5, color=INK_SOFT, bold=True)
        add_line(slide, px(a), yy, px(b), yy, color=RGBColor(0xB9, 0xB2, 0xA5), width=2.2)
        # 5 ep
        d1 = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px(a) - 0.065), Inches(yy - 0.065), Inches(0.13), Inches(0.13))
        d1.fill.solid(); d1.fill.fore_color.rgb = OCHRE; d1.line.color.rgb = WHITE; d1.line.width = Pt(1.2)
        # 20 ep
        d2 = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px(b) - 0.075), Inches(yy - 0.075), Inches(0.15), Inches(0.15))
        d2.fill.solid(); d2.fill.fore_color.rgb = TEAL; d2.line.color.rgb = WHITE; d2.line.width = Pt(1.2)
        # labels
        add_text(slide, px(a) - 0.35, yy - 0.38, 0.7, 0.2, f"{a:+.2f}", size=8.5, color=OCHRE, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, px(b) - 0.35, yy + 0.14, 0.7, 0.2, f"{b:+.2f}", size=8.5, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    # legend
    add_text(slide, x1 - 1.9, y + 0.18, 0.8, 0.2, "5 ep", size=8.5, color=OCHRE, bold=True, align=PP_ALIGN.RIGHT)
    add_text(slide, x1 - 0.95, y + 0.18, 0.8, 0.2, "20 ep", size=8.5, color=TEAL, bold=True, align=PP_ALIGN.RIGHT)


def slide_mechanisms(prs):
    s = new_slide(prs)
    add_header(s, "03 · Mechanisms", "A 5-epoch gain is not a 20-epoch gain")
    add_card(s, 0.7, 1.9, 8.3, 4.55, fill=WHITE)
    add_text(s, 0.95, 2.05, 7.8, 0.25, "DELTA IN TEST ACCURACY FROM THE N=64 BASELINE", size=8.5, color=TEAL, bold=True)
    add_dumbbell(s, 0.85, 2.2, 7.95, 3.85)
    add_text(s, 0.95, 6.05, 7.7, 0.25, "Only the learned initial state (M2) survives at 20 epochs.", size=9.5, color=INK_SOFT, italic=True)
    # right rail
    add_card(s, 9.25, 1.9, 3.4, 4.55, fill=WHITE)
    add_text(s, 9.48, 2.05, 2.95, 0.25, "WHAT ELSE WAS SCREENED", size=8.5, color=TEAL, bold=True)
    lines = [
        "SLAYER surrogate   +2.96",
        "silence tail        +2.03",
        "delay line          +1.94",
        "same-class blend    +1.15",
        "attention           +1.37*",
        "two-stage mixing     0.00",
        "adLIF currents      +0.13†",
    ]
    for i, line in enumerate(lines):
        add_text(s, 9.48, 2.4 + i * 0.36, 2.95, 0.25, line, size=9.5, color=INK_SOFT, font=BODY_FONT)
    add_text(s, 9.48, 5.15, 2.9, 0.8, "* not carried into the final model  ·  † configured but inert (200 params, zero gradient)", size=7.5, color=MUTED, line_spacing=1.08)
    add_text(s, 9.48, 5.95, 2.9, 0.35, "5-epoch block screens; N=64, D=1000", size=8, color=MUTED, italic=True)
    add_footer(s, 12, source="Screening deltas are single-run; baseline 76.15% at 5 ep and 82.51% at 20 ep (Results §3.4).")
    set_notes(
        s,
        "This is the honest mechanism story. Adaptation looked like +5.70 pp at five epochs but was -0.31 at twenty; input delays went from +1.99 to -1.37. "
        "Only the learned initial state and leak survived with +2.69 pp, and the gain came from z0, not from the learned alpha. "
        "The later five-epoch blocks are listed at right: SLAYER +2.96, silence tail +2.03, delay line +1.94, blend +1.15, attention +1.37 (not carried), two-stage 0.00, adLIF +0.13 (inert). "
        "The combined 30-epoch model is 88.07; the simpler 20-epoch dropout-only N=100 run is 88.12, so the extra blocks are not individually validated at 20 epochs.",
    )


def slide_augmentation(prs):
    s = new_slide(prs)
    add_header(s, "03 · Augmentation", "Event dropout is the only augmentation that survives")
    add_card(s, 0.7, 1.85, 8.4, 4.7, fill=WHITE)
    add_text(s, 0.95, 1.98, 7.9, 0.2, "Δ TEST accuracy vs no augmentation (pp)", size=8.5, color=MUTED, bold=True)
    add_augmentation_chart(s, 0.95, 2.15, 7.9, 4.15)
    # right rail
    add_card(s, 9.35, 1.85, 3.3, 4.7, fill=WHITE)
    add_text(s, 9.6, 2.05, 2.8, 0.25, "WHAT SURVIVES", size=9, color=TEAL, bold=True)
    add_text(s, 9.6, 2.42, 2.75, 0.35, "+1.68 pp", size=24, color=TEAL, bold=True)
    add_text(s, 9.6, 2.78, 2.75, 0.35, "dropout, 95% CI [+0.18, +3.14]", size=8.5, color=MUTED, line_spacing=1.1)
    add_text(s, 9.6, 3.35, 2.75, 0.35, "−2.96 pp", size=20, color=RED, bold=True)
    add_text(s, 9.6, 3.68, 2.75, 0.45, "global shift, CI [−4.51, −1.41]", size=8.5, color=MUTED, line_spacing=1.1)
    add_line(s, 9.6, 4.25, 12.4, 4.25, color=BORDER, width=0.6)
    add_text(s, 9.6, 4.38, 2.75, 0.3, "NO COMBINATION GAIN", size=8.5, color=OCHRE, bold=True)
    add_text(s, 9.6, 4.68, 2.75, 1.2, "Jitter + dropout: 76.46%\nDropout alone: 78.66%\nN=64, 5 epochs", size=8.8, color=INK_SOFT, line_spacing=1.2)
    add_text(s, 9.6, 5.85, 2.75, 0.45, "At 20 epochs dropout falls to +0.75 pp (CI −0.75 to +2.25).", size=8, color=MUTED, line_spacing=1.1)
    add_footer(s, 13, source="5-epoch screening, N=100; baseline 1803/2264 = 79.64%. All operations are raw-event, training-only.")
    set_notes(
        s,
        "Five raw-event operations were screened at N=100 and five epochs. Only event dropout has a resolved positive effect: +1.68 pp, 95% CI [+0.18, +3.14]. "
        "Global time shift is significantly negative: -2.96 pp, CI [-4.51, -1.41]. "
        "Jitter and dropout do not combine: at N=64, jitter+dropout is 76.46 versus 78.66 for dropout alone, a -2.21 pp interaction. "
        "At 20 epochs the N=64 dropout effect shrinks to +0.75 pp and is not resolved. The large N=100 apparent gain is confounded by the change from N=64 to N=100.",
    )


def slide_state_probes(prs):
    s = new_slide(prs)
    add_header(s, "03 · What the width is doing", "The state is wide; the readout is a thin slice")
    # card 1
    add_card(s, 0.7, 1.9, 3.85, 3.85, fill=WHITE)
    add_text(s, 0.95, 2.05, 3.3, 0.25, "DECODABILITY", size=9, color=TEAL, bold=True)
    add_text(s, 0.95, 2.35, 3.3, 0.3, "linear probe, TEST", size=8.5, color=MUTED)
    bars = [("spike counts", 87.81, TEAL), ("mean state", 84.67, BLUE_GRAY), ("final state", 27.61, OCHRE)]
    for i, (label, val, color) in enumerate(bars):
        yy = 2.82 + i * 0.78
        add_text(s, 0.95, yy, 1.55, 0.22, label, size=9, color=INK_SOFT)
        add_text(s, 2.55, yy, 0.65, 0.22, f"{val:.2f}%", size=9, color=INK, bold=True, align=PP_ALIGN.RIGHT)
        add_rect(s, 0.95, yy + 0.26, 3.25, 0.12, RGBColor(0xEC, 0xE7, 0xDC))
        add_rect(s, 0.95, yy + 0.26, max(0.02, 3.25 * val / 100), 0.12, color)
    add_text(s, 0.95, 5.25, 3.3, 0.4, "A 100,000-number state does worse than 100 spike counts.", size=8.5, color=MUTED, line_spacing=1.1)
    # card 2
    add_card(s, 4.78, 1.9, 3.85, 3.85, fill=WHITE)
    add_text(s, 5.03, 2.05, 3.3, 0.25, "EFFECTIVE DIMENSION", size=9, color=TEAL, bold=True)
    add_text(s, 5.03, 2.4, 3.3, 0.7, "7.49", size=38, color=INK, bold=True)
    add_text(s, 5.03, 3.08, 3.3, 0.3, "of 100,000 dimensions", size=9, color=MUTED)
    add_text(s, 5.03, 3.55, 3.25, 0.8, "participation ratio of the\npopulation mean state", size=9.5, color=INK_SOFT, line_spacing=1.15)
    add_text(s, 5.03, 4.42, 3.25, 0.65, "9 components → 90% variance\nper neuron: 4.90 of 1,000", size=9, color=MUTED, line_spacing=1.2)
    # card 3
    add_card(s, 8.86, 1.9, 3.8, 3.85, fill=WHITE)
    add_text(s, 9.1, 2.05, 3.25, 0.25, "DISTRIBUTED CODE", size=9, color=TEAL, bold=True)
    add_text(s, 9.1, 2.4, 3.25, 0.6, "0.30 ± 0.63 pp", size=21, color=INK, bold=True)
    add_text(s, 9.1, 2.98, 3.25, 0.3, "mean cost of silencing one neuron", size=8.5, color=MUTED)
    add_text(s, 9.1, 3.45, 3.25, 0.85, "91 / 100 neurons effective\nno critical unit\nworst single deletion: 2.69 pp", size=9.5, color=INK_SOFT, line_spacing=1.25)
    add_text(s, 9.1, 4.55, 3.25, 0.6, "Information is spread; deletion degrades smoothly.", size=8.5, color=MUTED, line_spacing=1.1)
    # bottom
    add_card(s, 0.7, 6.0, 11.95, 0.55, fill=TEAL_DARK, line=TEAL_DARK, radius=0.04)
    add_text(s, 0.95, 6.1, 11.4, 0.3, "The learned dynamics, not raw state dimension, carry the class information.", size=12, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_footer(s, 14, source="Frozen final N=100 checkpoint; probes trained on FIT and evaluated once on TEST (Results §3.6).")
    set_notes(
        s,
        "Three probes on the frozen final checkpoint. First, a linear decoder on spike counts gets 87.81%, the model readout is 88.07, a linear decoder on the 100,000-dimensional mean state gets 84.67, and the final state alone gets 27.61. "
        "The raw state is not extra readout bandwidth. Second, the population mean has a participation ratio of 7.49 out of 100,000; nine components explain 90% of the variance. "
        "Third, the code is distributed: silencing one neuron costs 0.30 pp on average, 91 of 100 neurons are effective, and there is no critical unit. "
        "The caveat is that the probe protocol changed for the final checkpoint: 150 epochs instead of 30, because the 30-epoch counts arm failed its alignment gate.",
    )


def slide_limitations(prs):
    s = new_slide(prs)
    add_header(s, "03 · Bounds", "What these numbers do not establish")
    cards = [
        (0.7, 1.9, "01", "SINGLE SEED", "Every number is seed 0. Run-to-run variation is not quantified, and the five-epoch screens are single-run."),
        (6.75, 1.9, "02", "SELECTION ON TEST", "The block set and dropout probability were chosen using five-epoch test accuracy. The 88.07% carries an unquantified winner's curse."),
        (0.7, 3.8, "03", "NO MATCHED BASELINE", "There is no N=100, 20/30-epoch, no-augmentation run. 88.07% cannot be split into 'more neurons' and 'more augmentation'."),
        (6.75, 3.8, "04", "VAL IS NOT GENERALISATION", "VAL shares speakers with FIT. A speaker-held-out pilot overstates held-out accuracy by roughly 28 pp."),
    ]
    for x, y, num, label, body in cards:
        add_card(s, x, y, 5.85, 1.7, fill=WHITE)
        add_text(s, x + 0.2, y + 0.14, 0.6, 0.35, num, size=18, color=TEAL_PALE, bold=True, font=TITLE_FONT)
        add_text(s, x + 0.82, y + 0.17, 4.7, 0.25, label, size=9, color=TEAL, bold=True)
        add_text(s, x + 0.22, y + 0.52, 5.4, 1.1, body, size=11.5, color=INK_SOFT, line_spacing=1.18)
    add_card(s, 0.7, 5.78, 11.95, 0.5, fill=TEAL_DARK, line=TEAL_DARK, radius=0.04)
    add_text(s, 0.95, 5.88, 11.45, 0.3, "Read 88.07% as a single-seed, test-selected estimate — not a deployment number.", size=11, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, 0.7, 6.5, 11.9, 0.25, "The scalar controls are untuned ablations, not competitive baselines. The test set was read repeatedly during development.", size=9, color=MUTED, italic=True)
    add_footer(s, 15, source="Full disclosure: Methods §2.8.7; Results §3.8; KNOWN_ISSUES.md.")
    set_notes(
        s,
        "These are the bounds that should travel with the headline. Single seed; the block set and dropout were selected on five-epoch test accuracy, so 88.07 carries a winner's curse; there is no matched no-augmentation N=100 baseline, so we cannot decompose the gain into neurons versus augmentation; and VAL is speaker-overlapping, so it is not a generalisation estimate. "
        "A speaker-held-out pilot overstates held-out accuracy by about 28 pp. The scalar controls are untuned ablations, not tuned baselines.",
    )


def slide_takeaways(prs):
    s = new_slide(prs)
    add_header(s, "04 · Takeaways", "What a neuron's internal state buys")
    rows = [
        ("01", "Width is an architectural variable.", "At fixed N=64, D=1 reaches 49.43%; D=1000 reaches 82.51% (20 epochs)."),
        ("02", "The gap is not parameter count.", "A 2M-parameter scalar reaches 33.13%; a 1000-wide broadcast recovers 29.5 of 43.7 pp; 14.2 pp remains for state width."),
        ("03", "The code is distributed and low-dimensional.", "The population mean is linearly decodable; the raw state is not extra bandwidth; no critical unit."),
        ("04", "The practical route is narrow.", "N=100 plus a learned initial state and event dropout; most screened mechanisms did not survive 20 epochs."),
    ]
    for i, (num, head, body) in enumerate(rows):
        yy = 1.95 + i * 1.12
        add_text(s, 0.7, yy, 0.55, 0.35, num, size=13, color=TEAL, bold=True)
        add_text(s, 1.25, yy - 0.02, 3.7, 0.3, head, size=15, color=INK, bold=True)
        add_text(s, 5.05, yy - 0.02, 7.5, 0.7, body, size=11.5, color=INK_SOFT, line_spacing=1.18)
        if i < 3:
            add_line(s, 0.7, yy + 0.95, 12.55, yy + 0.95, color=BORDER, width=0.6)
    add_card(s, 0.7, 6.35, 11.9, 0.55, fill=TEAL_DARK, line=TEAL_DARK, radius=0.04)
    add_text(s, 0.95, 6.45, 11.4, 0.3, "A controlled way to ask what state width buys — and the answer is not 'more parameters'.", size=11.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_footer(s, 16)
    set_notes(
        s,
        "Four takeaways. One: width is an architectural variable, not a free parameter; the 20-epoch scalar/vector pair at N=64 is the cleanest evidence. "
        "Two: the gap survives parameter matching; the broadcast control separates state width from transmission width. "
        "Three: the code is distributed and low-dimensional, and the raw state is not extra readout bandwidth. "
        "Four: the practical route to 88.07 is a larger population plus two confirmed changes, not a stack of mechanisms.",
    )


def slide_next(prs):
    s = new_slide(prs)
    add_header(s, "05 · Next", "Where this goes next")
    cards = [
        (0.7, "MANUSCRIPT", "Final figures: architecture, width/controls, mechanism screens, state probes. The native deck visuals are ready to be swapped 1:1."),
        (4.35, "EXPERIMENTS", "Multi-seed replication. N=100 no-augmentation baseline. 30-epoch drop-one and add-one ablations. Speaker-held-out validation."),
        (8.0, "ENGINEERING", "v3.1 diagonal emission: 1.03M parameters, 77.21% at 5 ep (screening). Dense emission remains the headline configuration."),
    ]
    for x, label, body in cards:
        add_card(s, x, 1.95, 4.0, 2.9, fill=WHITE)
        add_text(s, x + 0.22, 2.12, 3.6, 0.25, label, size=9, color=TEAL, bold=True)
        add_text(s, x + 0.22, 2.5, 3.55, 2.0, body, size=11.5, color=INK_SOFT, line_spacing=1.2)
    add_text(s, 0.7, 5.25, 11.9, 0.5, "Questions / discussion", size=24, color=INK, bold=True, font=TITLE_FONT)
    add_text(s, 0.7, 5.85, 11.9, 0.35, "Draft deck · November 2026 · final manuscript figures in preparation", size=9.5, color=MUTED)
    add_chip(s, 0.7, 6.4, 2.15, 0.32, "SHD  ·  20 classes", fill=TEAL_PALE, color=TEAL_DARK, size=8.5)
    add_chip(s, 3.0, 6.4, 2.45, 0.32, "N = 100  ·  D = 1000", fill=TEAL_PALE, color=TEAL_DARK, size=8.5)
    add_footer(s, 17)
    set_notes(
        s,
        "The deck is complete without the final manuscript figures; the native visuals can be replaced one-for-one when the figures land. "
        "The main experimental gaps are multi-seed replication, the missing N=100 no-augmentation baseline, and the 30-epoch drop-one ablations. "
        "The v3.1 diagonal-emission branch is an engineering variant, not a new headline model: 1.03M parameters and 77.21% at five epochs, screening.",
    )


def build(out_path: Path):
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    # Core properties
    prs.core_properties.title = "What a neuron's internal state buys"
    prs.core_properties.subject = "A vector-valued spiking neuron on the Spiking Heidelberg Digits benchmark"
    prs.core_properties.author = "Mehran Jaberinezhad"
    prs.core_properties.comments = "Project overview deck built with native editable shapes and charts."

    slide_title(prs)
    slide_scalar(prs)
    slide_question(prs)
    slide_architecture(prs)
    slide_where_vector(prs)
    slide_equations(prs)
    slide_training(prs)
    slide_results_glance(prs)
    slide_width(prs)
    slide_controls(prs)
    slide_headline(prs)
    slide_mechanisms(prs)
    slide_augmentation(prs)
    slide_state_probes(prs)
    slide_limitations(prs)
    slide_takeaways(prs)
    slide_next(prs)

    prs.save(out_path)
    print(f"saved {out_path} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build(Path("neuron_as_vector_project_deck.pptx"))
