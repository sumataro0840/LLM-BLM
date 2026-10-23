"""
Figures and numbers of LLM-BLM_小型株のデータ (図/). Reads data/smallcap.sqlite (read only) and the tables of
smallcap/bias.py. No stock reaction to a release is computed.

    python 資料集/02_実験/小型株のデータ/作成スクリプト/figs.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
sys.path.insert(0, str(REPO))
from smallcap import bias  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / '図'
DB = REPO / 'data' / 'smallcap.sqlite'
BLUE, ORANGE, AQUA, GRAY, INK, INK2, VIOLET, YELLOW = ('#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2', '#0b0b0b', '#52514e',
                                                       '#6250d6', '#eda100')
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})
REASONS = {'acquired': '買収', 'bankruptcy': '倒産', 'delisted_or_deregistered': '上場廃止・登録廃止',
           'still_filing': 'まだ提出している', 'other': 'そのほか', 'listed_but_no_price': '上場中だが株価なし'}


def fig_coverage(rel):
    g = rel.groupby('year').agg(kept=('kept', 'sum'), n=('kept', 'size'))
    g['lost'] = g.n - g.kept
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.bar(g.index, g.kept, color=BLUE, width=0.7, label='株価が取れる（残った）')
    ax.bar(g.index, g.lost, bottom=g.kept, color=ORANGE, width=0.7, label='株価が取れない（消えた）')
    for y, r in g.iterrows():
        ax.text(y, r.n + 25, f'{r.lost / r.n:.0%}', ha='center', fontsize=8.5, color=INK)
    ax.set_xticks(g.index)
    ax.set_ylabel('会社の数')
    ax.set_ylim(0, 2500)
    ax.legend(frameon=False, loc='upper center', fontsize=8.5, ncol=2)
    ax.set_xlabel('浮動株時価総額を測った年（数字は消えた割合）')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'coverage.pdf')


def fig_reasons(last_year):
    s = last_year.groupby('reason').size().sort_values()
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    ax.barh([REASONS[k] for k in s.index], s.values, color=[ORANGE if k == 'bankruptcy' else BLUE for k in s.index],
            height=0.6)
    for i, v in enumerate(s.values):
        ax.text(v + 15, i, f'{v:,}（{v / s.sum():.0%}）', va='center', fontsize=8.5, color=INK)
    ax.set_xlim(0, s.max() * 1.25)
    ax.set_xlabel('消えた会社の数')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'reasons.pdf')


def fig_float(rel, last_year):
    groups = [('残った会社（全年）', rel[rel.kept].dlog, BLUE),
              ('買収された会社（最後の年）', last_year[last_year.reason == 'acquired'].dlog, AQUA),
              ('倒産した会社（最後の年）', last_year[last_year.reason == 'bankruptcy'].dlog, ORANGE)]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    bins = np.linspace(-2.5, 2.5, 51)
    for name, x, color in groups:
        x = x.dropna().clip(-2.5, 2.5)
        ax.hist(x, bins=bins, density=True, histtype='step', lw=1.8, color=color,
                label=f'{name}  中央値 {np.expm1(x.median()):+.0%}')
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel('前の年からの浮動株時価総額の変化（対数。−0.69 で半分、+0.69 で 2 倍）')
    ax.set_ylabel('密度')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'float_change.pdf')


def release_numbers(conn):
    rel = pd.read_sql('SELECT r.*, s.symbol AS sample_symbol FROM earnings_releases r JOIN sc_sample s USING (cik)', conn)
    fil = pd.read_sql('SELECT f.* FROM sc_filings f JOIN sc_sample s USING (cik)', conn)
    rel['year'] = rel.filing_date.str[:4]
    with_text = rel[rel.text_path.notna()]
    per_company = with_text.groupby('cik').size()
    numbers = {
        'filings': int(len(fil)), 'companies_with_filings': int(fil.cik.nunique()),
        'releases_stored': int(len(rel)), 'with_text': int(len(with_text)),
        'ex991_share_of_texts': float(with_text.doc_type.isin(['EX-99.1', 'EX-99', 'EX-99.01']).mean()),
        'how_chosen': {k: int(v) for k, v in rel.how_chosen.value_counts().items()},
        'event_date_share': float(with_text.event_date.notna().mean()),
        'n_chars_median': float(with_text.n_chars.median()),
        'session': {k: int(v) for k, v in with_text.session.value_counts().items()},
        'companies_with_4_texts': int((per_company >= 4).sum()),
        'texts_per_year': {k: int(v) for k, v in with_text.groupby('year').size().items()},
    }
    return numbers, with_text


def fig_sample(conn, with_text):
    band = pd.read_sql('SELECT b.year, COUNT(*) AS n FROM sc_band b JOIN sc_sample s USING (cik) WHERE b.in_band = 1 '
                       'GROUP BY b.year', conn)
    per_year = with_text.groupby(with_text.filing_date.str[:4].astype(int)).size()
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.bar(band.year, band.n, color=BLUE, width=0.7, label='小型株の範囲に入っている会社（左の目盛り）')
    ax.set_ylabel('会社の数')
    ax.set_xticks(range(2015, 2027))
    ax2 = ax.twinx()
    ax2.plot(per_year.index, per_year.values, color=ORANGE, marker='o', lw=1.8, label='本文を取れた決算発表文（右の目盛り）')
    ax2.set_ylabel('決算発表文の数')
    ax2.grid(False)
    ax2.spines['right'].set_visible(True)
    ax2.set_ylim(0, per_year.max() * 1.25)
    ax.set_ylim(0, band.n.max() * 1.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc='upper left')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'sample.pdf')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    band, rel, last_year = bias.tables()
    fig_coverage(rel)
    fig_reasons(last_year)
    fig_float(rel, last_year)
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    numbers, with_text = release_numbers(conn)
    fig_sample(conn, with_text)
    sic = pd.read_sql('SELECT c.sic_description, COUNT(*) AS n FROM sc_sample s JOIN sc_companies c USING (cik) '
                      'GROUP BY 1 ORDER BY 2 DESC LIMIT 10', conn)
    numbers['top_industries'] = dict(zip(sic.sic_description, sic.n.astype(int)))
    (OUT / 'numbers.json').write_text(json.dumps(numbers, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(numbers, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
