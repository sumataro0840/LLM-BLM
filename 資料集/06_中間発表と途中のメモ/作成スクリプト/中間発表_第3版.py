"""
中間発表のスライド（第 3 版、10 枚、理系の大学生向け）を作る。前回の中間発表のファイルを土台にして、表紙（ロゴ・題名・所属）と
スライドの大きさ・書体・色づかいを引き継ぎ、2 枚目以降を作り直す。専門用語は使わないか、その場で言い換える。
数値はすべて実験の記録（資料集の全体の記録、実験 2 の図のデータ）から。

    python 中間発表_第3版.py            （python-pptx が要る）

出力：../2026_情報工学工房中間発表資料_第3版.pptx（前回のファイルは変えない）
"""
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '2026_情報工学工房中間発表資料_修正版claude.pptx'
OUT = HERE.parent / '2026_情報工学工房中間発表資料_第3版.pptx'

FONT = 'メイリオ'
BLUE = '1973EB'       # the previous deck's accent
DARK = '0D1C3B'
BLACK = '000000'
CARD = 'F5F2F9'       # the previous deck's card fill
PALE = 'EAF2FD'
GRAY = '5B616B'
LIGHT = 'C9CED6'
ORANGE = 'E0601F'
WHITE = 'FFFFFF'

# 実験 2 の図（資料集/02_実験/実験2_決算発表文/作成スクリプト/figs.py の fig_quintile と fig_car）と同じ計算の値。
# 点数の 5 分位ごとの、発表をはさむ 2 取引日の超過リターンの平均（%）
QUINTILE = [-1.773, -0.198, -0.250, 0.456, 1.164]
# 点数の上位・中位・下位 20% の、発表の 5 日前から 21 日後までの累積の超過リターン（%）
DAYS = list(range(-5, 22))
PATH_TOP = [0.07, 0.245, 0.306, 0.382, 0.542, 1.811, 1.808, 1.916, 2.0, 2.027, 1.992, 2.067, 2.065, 2.12, 2.119,
            2.149, 2.142, 2.104, 2.259, 2.228, 2.178, 2.238, 2.352, 2.333, 2.404, 2.382, 2.329]
PATH_MID = [0.061, 0.027, 0.107, 0.145, 0.139, 0.047, -0.087, -0.133, -0.147, -0.249, -0.307, -0.302, -0.304, -0.453,
            -0.482, -0.528, -0.525, -0.514, -0.518, -0.537, -0.437, -0.429, -0.393, -0.511, -0.454, -0.436, -0.412]
PATH_LOW = [-0.182, -0.194, -0.159, -0.185, -0.245, -2.099, -2.207, -2.24, -2.2, -2.18, -2.159, -2.261, -2.232, -2.274,
            -2.211, -2.278, -2.331, -2.283, -2.323, -2.339, -2.302, -2.298, -2.273, -2.297, -2.302, -2.241, -2.239]


# ---------- helpers ----------

def rgb(h):
    return RGBColor.from_string(h)


def set_font(font, size=None, bold=None, color=None):
    """Size, weight, colour, and the same face for Latin and Japanese text; Japanese line breaking."""
    if size is not None:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    if color is not None:
        font.color.rgb = rgb(color)
    font.name = FONT
    rPr = font._rPr
    rPr.set('lang', 'ja-JP')          # Japanese line breaking (no 。 at the start of a line)
    for tag in ('a:ea', 'a:cs'):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set('typeface', FONT)


def text(slide, x, y, w, h, paras, size=16, color=BLACK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, after=0, name=None):
    """A text box. paras: list of paragraphs; a paragraph is a str or a list of (text, {size, color, bold})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if after:
            p.space_after = Pt(after)
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run()
            r.text = t
            set_font(r.font, o.get('size', size), o.get('bold', bold), o.get('color', color))
    return tb


def box(slide, x, y, w, h, fill=CARD, line=None, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1.25)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def arrow(slide, x1, y1, x2, y2, color=GRAY, width=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    return c


def dot_row(slide, x, y, w, para, size=16, color=BLACK, h=0.45):
    """A bullet row in the previous deck's style: a small blue circle and a text box."""
    box(slide, x + 0.16, y + 0.11, 0.14, 0.14, fill=BLUE, shape=MSO_SHAPE.OVAL)
    text(slide, x + 0.42, y, w - 0.42, h, [para], size=size, color=color)


def numbered(slide, x, y, n, fill=BLUE, d=0.56):
    box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    text(slide, x, y, d, d, [str(n)], size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def header(slide, title, claim):
    text(slide, 0.54, 0.22, 12.2, 0.8, [title], size=38, bold=True, color=BLACK, name='タイトル', anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 0.74, 1.15, 11.9, 0.5, [claim], size=21, bold=True, color=BLUE, name='主張')


def footnote(slide, s):
    text(slide, 0.74, 6.95, 11.9, 0.36, [s], size=11, color=GRAY, name='注')


def style_axis_font(axis, size=14):
    set_font(axis.tick_labels.font, size, False, DARK)
    axis.format.line.color.rgb = rgb(LIGHT)


def chart_title(ch, title):
    ch.has_title = True
    tf = ch.chart_title.text_frame
    tf.text = title
    set_font(tf.paragraphs[0].runs[0].font, 15, True, DARK)


def bar_chart(slide, x, y, w, h, cats, vals, colors, title, fmt='0', horizontal=True, vmin=0, vmax=None, name=None):
    data = CategoryChartData()
    data.categories = cats
    data.add_series('値', vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data)
    if name:
        gf.name = name
    ch = gf.chart
    set_font(ch.font, 14, False, DARK)
    ch.has_legend = False
    chart_title(ch, title)
    plot = ch.plots[0]
    plot.gap_width = 45
    plot.vary_by_categories = False
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    set_font(dl.font, 15, True, DARK)
    ser = plot.series[0]
    ser.invert_if_negative = False
    for i, c in enumerate(colors):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)
        pt.format.line.fill.background()
    # keep the colour of negative bars (PowerPoint draws them white unless each point says otherwise)
    for dpt in ser._element.findall(qn('c:dPt')):
        if dpt.find(qn('c:invertIfNegative')) is None:
            dpt.find(qn('c:idx')).addnext(dpt.makeelement(qn('c:invertIfNegative'), {'val': '0'}))
    va = ch.value_axis
    va.has_major_gridlines = False
    va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    va.visible = False
    ca = ch.category_axis
    style_axis_font(ca)
    ca.has_major_gridlines = False
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW     # labels below negative bars, not on them
    if horizontal:
        ca.reverse_order = True        # first category at the top
    return gf


def line_chart(slide, x, y, w, h, cats, series, title, name=None):
    data = CategoryChartData()
    data.categories = cats
    for label, vals, _ in series:
        data.add_series(label, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(x), Inches(y), Inches(w), Inches(h), data)
    if name:
        gf.name = name
    ch = gf.chart
    set_font(ch.font, 13, False, DARK)
    chart_title(ch, title)
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    set_font(ch.legend.font, 13, False, DARK)
    for ser, (_, _, color) in zip(ch.plots[0].series, series):
        ser.smooth = False
        ser.marker.style = XL_MARKER_STYLE.NONE
        ser.format.line.color.rgb = rgb(color)
        ser.format.line.width = Pt(3)
    va = ch.value_axis
    va.minimum_scale = -3
    va.maximum_scale = 3
    va.major_unit = 1
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb('E6E8EC')
    va.tick_labels.number_format = '+0"%";-0"%";0"%"'
    va.tick_labels.number_format_is_linked = False
    style_axis_font(va, 13)
    ca = ch.category_axis
    style_axis_font(ca, 13)
    ca.has_major_gridlines = False
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    return gf


def blank(prs):
    return prs.slides.add_slide(next(l for l in prs.slide_layouts if l.name == 'Blank'))


# ---------- the deck ----------

def build():
    prs = Presentation(str(BASE))
    # keep the title slide, drop the rest
    ids = prs.slides._sldIdLst
    for sid in list(ids)[1:]:
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)
    prs.slides[0].notes_slide.notes_text_frame.text = (
        'ChatGPT のような AI に株を選ばせたら儲かるか、という研究の中間発表です。前回は、元の論文の方法を再現して、'
        '問題点を見つけたところまででした。今回は、問いを立て直して行った実験の結果を話します。')

    # 2 ---------------------------------------------------------------- one slide summary
    s = blank(prs)
    header(s, 'この研究を一言で', 'AI は決算の文章を読める。でも、市場も同じ日に読んでいる')
    box(s, 0.74, 1.95, 11.9, 0.85, fill=PALE, name='問い')
    text(s, 1.0, 1.95, 11.4, 0.85, [[('問い：', {'bold': True, 'color': BLUE}),
                                    ('ChatGPT のような AI（LLM）に株を選ばせたら、市場の平均より儲かるか？', {})]],
         size=18, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    steps = [('数字だけを見せる', '過去 1 か月の値動きを見せて来月を予想させると、先月の動きをまねするだけ', '×  当たらない', ORANGE),
             ('決算の文章を読ませる', '会社の成績発表の文章を読ませて採点させると、発表直後の株価の動きと合う', '○  読める', BLUE),
             ('その情報で運用する', 'AI の点数でポートフォリオを約 10 年組んでも、市場の平均とほぼ同じ', '×  儲からない', ORANGE)]
    for i, (lab, body, verdict, col) in enumerate(steps):
        x = 0.74 + i * 4.05
        box(s, x, 3.1, 3.8, 3.55, name=f'段階{i + 1}')
        numbered(s, x + 0.25, 3.3, i + 1)
        text(s, x + 0.95, 3.3, 2.75, 0.56, [lab], size=18, bold=True, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.3, 4.1, 3.25, 1.5, [body], size=16, color=DARK)
        text(s, x + 0.3, 5.75, 3.25, 0.6, [verdict], size=22, bold=True, color=col)
        if i < 2:
            arrow(s, x + 3.83, 4.87, x + 4.02, 4.87)
    s.notes_slide.notes_text_frame.text = (
        '最初に結論です。問いは、AI に株を選ばせたら市場の平均より儲かるか、です。3 段階で調べました。'
        '1 つ目、数字だけを見せて予想させると、AI は先月の動きをまねするだけでした。2 つ目、会社の決算の文章を読ませると、'
        '発表直後の株価の動きとよく合う点数を付けられました。3 つ目、ところがその点数で運用すると、市場の平均とほぼ同じ。'
        'AI が読めることは、市場も発表の当日に読んでいた、というのが結論です。')

    # 3 ---------------------------------------------------------------- background
    s = blank(prs)
    header(s, '背景：株をどう組み合わせるか', '市場の平均的な見方から出発し、自分の予想に自信がある分だけずらす')
    defs = [('ポートフォリオ', 'どの株に、どれだけの割合で投資するか。この研究の最終的な出力'),
            ('市場の平均（S&P500）', '米国の大企業約 500 社に、会社の大きさ（時価総額）に比例して投資したもの。プロの多くもこれに勝てない')]
    for i, (lab, body) in enumerate(defs):
        y = 1.95 + i * 2.05
        box(s, 0.74, y, 4.6, 1.85, name=f'用語{i + 1}')
        text(s, 0.99, y + 0.15, 4.1, 0.45, [lab], size=18, bold=True, color=BLUE)
        text(s, 0.99, y + 0.65, 4.1, 1.15, [body], size=15, color=DARK)
    # diagram
    box(s, 5.75, 1.95, 2.9, 1.05, fill=WHITE, line=LIGHT, name='市場の比率')
    text(s, 5.75, 1.95, 2.9, 1.05, ['市場の比率', [('みんなの平均的な見方', {'size': 13, 'bold': False, 'color': GRAY})]],
         size=17, bold=True, color=DARK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 5.75, 3.35, 2.9, 1.05, fill=WHITE, line=LIGHT, name='予想と自信')
    text(s, 5.75, 3.35, 2.9, 1.05, ['AI の予想', [('と、その自信の強さ', {'size': 13, 'bold': False, 'color': GRAY})]],
         size=17, bold=True, color=DARK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 9.3, 2.45, 3.34, 1.45, fill=BLUE, name='BLM')
    text(s, 9.3, 2.45, 3.34, 1.45, ['Black–Litterman', [('モデルで混ぜる', {'size': 15})]], size=18, bold=True,
         color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    arrow(s, 8.7, 2.48, 9.25, 2.95)
    arrow(s, 8.7, 3.87, 9.25, 3.4)
    box(s, 9.3, 4.45, 3.34, 0.95, fill=WHITE, line=BLUE, name='新しい比率')
    text(s, 9.3, 4.45, 3.34, 0.95, ['新しい比率'], size=17, bold=True, color=BLUE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    arrow(s, 10.97, 3.93, 10.97, 4.42)
    text(s, 5.75, 4.6, 3.3, 0.9, ['自信が弱い → 市場のまま', '自信が強い → 予想どおりに動かす'], size=14, color=DARK)
    box(s, 5.75, 5.7, 6.89, 0.95, fill=PALE, name='ベイズ')
    text(s, 6.0, 5.7, 6.5, 0.95, [[('ベイズ推定でいうと：', {'bold': True, 'color': BLUE}),
                                  ('事前分布＝市場の見方、観測＝AI の予想、事後分布＝新しい見方', {})]],
         size=15, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    s.notes_slide.notes_text_frame.text = (
        '用語を 2 つだけ説明します。ポートフォリオは、どの株にどれだけ投資するかの組み合わせです。市場の平均は、'
        'S&P500 という米国の大企業約 500 社を、会社の大きさに比例して持つもので、プロでもこれに勝つのは難しいです。'
        'Black-Litterman モデルは、この市場の見方から出発して、自分の予想に自信がある分だけ比率をずらす方法です。'
        'ベイズ推定でいえば、市場の見方が事前分布、AI の予想が観測、混ぜた結果が事後分布です。')

    # 4 ---------------------------------------------------------------- replication
    s = blank(prs)
    header(s, '１．元の論文の方法を再現した', '数字を見せて予想させると、AI は先月の値動きをまねするだけだった')
    box(s, 0.74, 1.95, 5.0, 4.75, name='元の方法')
    text(s, 0.99, 2.08, 4.5, 0.4, ['元の論文の方法'], size=17, bold=True, color=BLUE)
    flow = ['過去 1 か月の株価の動き（約 20 個の数字）と会社名を見せる',
            'AI（gpt-4o-mini）が来月の値動きを 30 回答える',
            '30 個の平均＝予想、ばらつき＝自信のなさ',
            'Black–Litterman モデルで比率を決める']
    for i, t in enumerate(flow):
        y = 2.6 + i * 1.02
        box(s, 0.99, y, 4.5, 0.74, fill=WHITE, line=LIGHT)
        text(s, 1.14, y, 4.2, 0.74, [t], size=14, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
        if i < 3:
            arrow(s, 3.24, y + 0.76, 3.24, y + 1.0)
    stats = [('0.89', 'AI の予想と「先月の値動き」の相関。ほぼそのまま写している'),
             ('ほぼ 0', '先月の値動きで説明できない、AI ならではの当たり（相関 −0.004）'),
             ('99.9%', '自信が強すぎて、市場の見方をほぼ捨てて予想だけを信じていた')]
    for i, (num, desc) in enumerate(stats):
        y = 1.95 + i * 1.62
        box(s, 6.0, y, 6.64, 1.43, name=f'問題{i + 1}')
        text(s, 6.25, y, 2.0, 1.43, [num], size=32, bold=True, color=ORANGE, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 8.35, y, 4.1, 1.43, [desc], size=15, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    footnote(s, 'S&P500 の約 490 社 × 30 か月（2024-03〜2026-08）、AI の回答 44 万個。')
    s.notes_slide.notes_text_frame.text = (
        '前回までの内容を短くまとめます。元の論文は、過去 1 か月の株価の動きを AI に見せて、来月の値動きを 30 回答えさせ、'
        'その平均を予想、ばらつきを自信のなさとして使います。再現すると、AI の予想は先月の値動きと相関 0.89 で、'
        'ほぼ写しているだけでした。それを取り除いた AI ならではの当たりは、ほぼ 0 でした。しかも自信が強すぎて、'
        '市場の見方を 99.9% 捨てていました。数字を渡して数字を答えさせても、AI の強みは出ない、というのが前回の結論です。')

    # 5 ---------------------------------------------------------------- new question
    s = blank(prs)
    header(s, '２．問いを変えた：決算発表を読ませる', 'まず、AI の読み取りが発表直後の株価の動きと合うかを確かめる')
    box(s, 0.74, 1.95, 11.9, 0.95, fill=PALE, name='決算発表とは')
    text(s, 1.0, 1.95, 11.4, 0.95, [[('決算発表とは：', {'bold': True, 'color': BLUE}),
                                    ('会社が 3 か月ごとに出す成績表。売上・利益と、今後の見通しが書いてある。出た直後に株価が大きく動く', {})]],
         size=16, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    pipe = [('入力', '決算発表の文章（英語）。米国の大企業、2023 年 11 月〜 の 5,374 本'),
            ('AI', 'gpt-4o-mini。学習はさせず、質問するだけ'),
            ('出力', '見通し・語調・業績予想の上げ下げを −2〜+2 で採点'),
            ('比べるもの', '発表をはさむ 2 日間の株価の動き（同じ業種の株の平均との差）')]
    for i, (h, body) in enumerate(pipe):
        x = 0.74 + i * 3.1
        box(s, x, 3.2, 2.8, 2.05, name=f'流れ{i + 1}')
        text(s, x + 0.2, 3.33, 2.4, 0.4, [h], size=17, bold=True, color=BLUE)
        text(s, x + 0.2, 3.8, 2.4, 1.4, [body], size=15, color=DARK)
        if i < 3:
            arrow(s, x + 2.83, 4.22, x + 3.07, 4.22)
    box(s, 0.74, 5.55, 11.9, 1.05, name='ズルをしない')
    text(s, 1.0, 5.55, 11.4, 1.05, [[('ズルをしない工夫（事前登録）：', {'bold': True, 'color': BLUE}),
                                    ('結果を見てから都合のよい分析を選べないよう、計画と解析プログラムを先に固定し、1 回だけ実行した', {})]],
         size=16, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    s.notes_slide.notes_text_frame.text = (
        'そこで問いを変えました。AI が得意なのは文章を読むことです。決算発表は、会社が 3 か月ごとに出す成績表で、'
        '出た直後に株価が大きく動きます。その文章を AI に読ませて採点させ、点数が発表直後の株価の動きと合うかを調べました。'
        '比べるのは、発表をはさむ 2 日間の株価の動きから、同じ業種の株の平均の動きを引いたものです。'
        'また、結果を見てから都合のよい分析を選ばないよう、計画とプログラムを先に固定して 1 回だけ実行しました。')

    # 6 ---------------------------------------------------------------- result 1
    s = blank(prs)
    header(s, '３．結果①：AI の点数で株価の動きが分かれる', '点数の高い発表ほど、発表直後の株価が上がる')
    bar_chart(s, 0.74, 1.85, 7.1, 4.95, ['1（低い）', '2', '3', '4', '5（高い）'], QUINTILE,
              [ORANGE, ORANGE, ORANGE, BLUE, BLUE], 'AI の点数で 5 グループに分けた、2 日間の株価の動き',
              fmt='+0.00"%";-0.00"%"', horizontal=False, vmin=-2.5, vmax=2.0, name='5 グループのグラフ')
    box(s, 8.1, 1.95, 4.54, 2.05, name='差')
    text(s, 8.35, 2.05, 4.1, 0.75, ['約 2.9%'], size=34, bold=True, color=BLUE)
    text(s, 8.35, 2.85, 4.1, 1.1, ['点数が最も高い 2 割と最も低い 2 割の、2 日間の株価の動きの差'], size=15, color=DARK)
    box(s, 8.1, 4.2, 4.54, 2.5, name='偶然ではない')
    text(s, 8.35, 4.3, 4.1, 0.75, ['t = 9.4'], size=34, bold=True, color=BLUE)
    text(s, 8.35, 5.1, 4.1, 1.55, ['偶然では説明できない（目安：2 を超えれば偶然とは考えにくい）。'
                                   '売上・利益の数字、アナリストの予想、昔ながらの文章分析を考えに入れても残る'], size=14, color=DARK)
    footnote(s, '米国の大企業（S&P500）、2023-11〜2026-08 の決算発表 5,374 本。株価の動きは、同じ業種の株をまとめた指数との差。')
    s.notes_slide.notes_text_frame.text = (
        '結果です。AI の点数で発表を 5 つのグループに分けて、発表直後 2 日間の株価の動きを平均しました。'
        '点数の低いグループは約 1.8% 下がり、高いグループは約 1.2% 上がりました。上と下で約 2.9% の差です。'
        't 値は 9.4 で、偶然ではまず起きない大きさです。売上や利益の数字、アナリストの予想、昔ながらの単語を数える'
        '文章分析を考えに入れても、この関係は残りました。AI は、数字だけでは分からない情報を文章から読み取っています。')

    # 7 ---------------------------------------------------------------- look-ahead
    s = blank(prs)
    header(s, '３．結果②：答えを覚えていたのでは？', '2015 年までの文章しか知らない AI でも、昔ながらの文章分析より読めた')
    box(s, 0.74, 1.95, 4.75, 4.75, name='説明')
    text(s, 0.99, 2.1, 4.25, 0.4, ['心配'], size=17, bold=True, color=ORANGE)
    text(s, 0.99, 2.55, 4.25, 1.55, ['gpt-4o-mini は 2023 年までのネットの文章で学習している。昔の決算なら、その後の株価を'
                                     '知っているかもしれない（テストの答えを知っている受験生）'], size=15, color=DARK)
    text(s, 0.99, 4.2, 4.25, 0.4, ['確かめ方'], size=17, bold=True, color=BLUE)
    text(s, 0.99, 4.65, 4.25, 1.95, ['ChronoGPT（2015 年末までの文章だけで学習した AI）に、2020〜2026 年の決算を読ませる。'
                                     '答えを知りようがない'], size=15, color=DARK)
    bar_chart(s, 5.75, 1.85, 6.9, 4.95,
              ['単語の辞書', '金融用の AI（BERT）', '単語の出現頻度＋回帰', '昔ながらの 3 つの合計', 'ChronoGPT', 'BERT（株価で追加学習）'],
              [0.02, 0.24, 0.37, 0.51, 0.59, 1.01], [LIGHT, LIGHT, LIGHT, GRAY, BLUE, GRAY],
              '株価の動きをどれだけ説明できたか（決定係数の増分）', fmt='0.00', vmax=1.3, name='比較のグラフ')
    footnote(s, '2020-01〜2026-08 の決算発表 13,141 本。売上・利益などの数字の情報に足したとき、決定係数 R² が何ポイント増えたか。')
    s.notes_slide.notes_text_frame.text = (
        'ここで心配なのは、AI が答えを覚えていたのではないか、ということです。gpt-4o-mini は 2023 年までのネットの文章で'
        '学習しているので、昔の決算については、その後の株価まで読んでいるかもしれません。テストの答えを知っている受験生です。'
        'そこで、2015 年末までの文章だけで学習した ChronoGPT という AI に、2020 年以降の決算を読ませました。'
        'これなら答えを知りようがありません。それでも、単語を数える方法など昔ながらの 3 つの方法を合わせたものより'
        'よく説明できました。株価で追加学習させた BERT には負けますが、10 年前の言葉しか知らない AI としては十分です。')

    # 8 ---------------------------------------------------------------- no profit
    s = blank(prs)
    header(s, '３．結果③：でも儲からない', '株価は発表の日に動き終わる。後から買っても間に合わない')
    cats = [('発表日' if d == 0 else (f'{d:+d}' if d % 5 == 0 else '')) for d in DAYS]
    line_chart(s, 0.74, 1.85, 7.3, 5.0, cats,
               [('点数が上位 20%', PATH_TOP, BLUE), ('中位 20%', PATH_MID, GRAY), ('下位 20%', PATH_LOW, ORANGE)],
               '発表の前後の株価の動き（同じ業種との差の累積）', name='前後のグラフ')
    text(s, 2.9, 2.4, 4.9, 0.4, ['発表の日に動き終わり、あとは横ばい'], size=14, bold=True, color=DARK)
    res = [('毎月組み替えて約 10 年', '市場の平均とほぼ同じ', '年 +0.04%（手数料の後）'),
           ('毎日組み替える', '手数料で負ける', '年 −20%'),
           ('なぜ？', '大勢のプロと自動売買', '発表から数秒〜数時間で値段に反映される')]
    for i, (lab, big, sub) in enumerate(res):
        y = 1.95 + i * 1.62
        box(s, 8.3, y, 4.34, 1.43, name=f'運用{i + 1}')
        text(s, 8.5, y + 0.12, 4.0, 0.4, [lab], size=15, bold=True, color=GRAY)
        text(s, 8.5, y + 0.5, 4.0, 0.5, [big], size=20, bold=True, color=ORANGE if i < 2 else BLUE)
        text(s, 8.5, y + 0.98, 4.0, 0.4, [sub], size=14, color=DARK)
    footnote(s, '左：点数の上位・中位・下位 20% の発表（各約 1,075 本）。右：AI の点数で組んだポートフォリオの、市場の平均に対する成績。')
    s.notes_slide.notes_text_frame.text = (
        'では、これで儲かるか。左のグラフは、発表の 5 日前から 21 日後までの株価の動きです。点数の高い発表は上がり、'
        '低い発表は下がりますが、動くのはほぼ発表の日だけで、そのあとは横ばいです。つまり、AI が文章を読んでから買っても、'
        'もう値段に入っています。実際に AI の点数で毎月ポートフォリオを組み替えて約 10 年運用すると、市場の平均とほぼ同じ。'
        '毎日組み替えると、手数料で年マイナス 20% でした。市場には大勢のプロと自動売買のプログラムがいて、'
        '発表から数秒から数時間で値段に反映してしまうからです。')

    # 9 ---------------------------------------------------------------- other attempts
    s = blank(prs)
    header(s, '４．ほかに試したこと', 'AI の中身の学習や読ませ方の工夫では、ほとんど良くならなかった')
    tries = [('AI を株価で追加学習（LoRA）', '約 1.4 万本の発表で AI の中身を学習させたが、学習なしで内部の数値を使うほうが強かった',
              '×  弱くなった', ORANGE),
             ('長い文章の読ませ方', '表や定型文の削除、末尾も読む、圧縮など 5 通り。開発用の期間で選び、最後に 1 回だけ比べた',
              '×  ほぼ変わらず', ORANGE),
             ('AI の途中の層を使う', '最後の層ではなく、途中の層の数値を使うと、説明力（決定係数の増分）が 0.49 → 0.80 に',
              '△  有望（要検証）', BLUE),
             ('リスクの約束を守る', '「市場平均からのずれ」を年 3% に抑えるはずが実際は 4.8%。過去の実績で上限を補正すると 3.4% に',
              '○  改善', BLUE)]
    for i, (lab, body, verdict, col) in enumerate(tries):
        x = 0.74 + (i % 2) * 6.05
        y = 1.95 + (i // 2) * 2.45
        box(s, x, y, 5.85, 2.25, name=f'試み{i + 1}')
        text(s, x + 0.25, y + 0.15, 5.35, 0.45, [lab], size=18, bold=True, color=DARK)
        text(s, x + 0.25, y + 0.65, 5.35, 1.0, [body], size=15, color=DARK)
        text(s, x + 0.25, y + 1.68, 5.35, 0.45, [verdict], size=17, bold=True, color=col)
    footnote(s, '「途中の層」は事前に計画していなかった探索の結果なので、次の実験で改めて確かめる。')
    s.notes_slide.notes_text_frame.text = (
        'ほかにも試したことを 4 つ紹介します。AI の中身そのものを株価で追加学習させる方法は、学習させない使い方より弱くなりました。'
        'データが 1 万 4 千本では、AI の多くの重みを合わせるには足りなかったと考えています。長い文章の読ませ方を 5 通り工夫しても、'
        'ほとんど変わりませんでした。一方、AI の途中の層の数値を使うと説明力が大きく上がりました。ただしこれは計画外の'
        '探索なので、改めて確かめます。最後に、市場平均からのずれを年 3% に抑えるはずが 4.8% になっていた問題は、'
        '過去の実績で上限を補正する簡単な方法で 3.4% まで直せました。')

    # 10 --------------------------------------------------------------- conclusion
    s = blank(prs)
    header(s, '５．まとめと今後', 'AI の強みは「市場の読み違い」や「リスクの予測」の側にあるかもしれない')
    text(s, 0.74, 1.95, 5.2, 0.4, ['まとめ'], size=18, bold=True, color=BLUE)
    box(s, 0.74, 2.45, 5.2, 3.6, name='まとめ')
    summ = ['数字を見せて予想させても、AI は先月の値動きをまねするだけ',
            'AI は決算の文章を読める。答えを知らない AI でも、昔ながらの方法より読める',
            'でも株価は発表の日に動き終わるので、その情報では儲からない']
    for i, t in enumerate(summ):
        dot_row(s, 0.94, 2.7 + i * 1.1, 4.85, t, size=15, color=DARK, h=1.0)
    nxt = [('市場が読み違えた発表を探す', '文章は良いのに株価が下がった発表は、その後に戻るか'),
           ('値動きの大きさ（リスク）を当てる', '上がる・下がるより、荒れるかどうかの方が長く続く'),
           ('複数の AI の意見の割れを使う', '意見が割れる株ほど、予想を信じすぎない')]
    text(s, 6.2, 1.95, 6.4, 0.4, ['今後'], size=18, bold=True, color=BLUE)
    for i, (lab, desc) in enumerate(nxt):
        y = 2.45 + i * 1.25
        box(s, 6.2, y, 6.44, 1.1, name=f'今後{i + 1}')
        numbered(s, 6.38, y + 0.27, i + 1)
        text(s, 7.15, y + 0.1, 5.3, 0.45, [lab], size=16, bold=True, color=DARK)
        text(s, 7.15, y + 0.55, 5.3, 0.5, [desc], size=14, color=DARK)
    box(s, 0.74, 6.25, 11.9, 0.6, fill=PALE, name='学んだこと')
    text(s, 1.0, 6.25, 11.4, 0.6, [[('学んだこと：', {'bold': True, 'color': BLUE}),
                                   ('結果を見てから分析を選ばない。単位や日付の小さなミスでも結論が変わる', {})]],
         size=15, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    s.notes_slide.notes_text_frame.text = (
        'まとめです。数字を見せて予想させても、AI は先月の値動きをまねするだけでした。決算の文章は読めて、'
        '答えを知らない AI でも昔ながらの方法より読めました。でも株価は発表の日に動き終わるので、その情報では儲かりません。'
        '今後は、市場が読み違えたかもしれない発表を探すこと、上がる下がるではなく値動きの荒さを当てること、'
        '複数の AI の意見の割れを自信の強さに使うこと、を調べる予定です。以上です。')

    prs.save(str(OUT))
    print('saved', OUT)


if __name__ == '__main__':
    build()
