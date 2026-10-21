"""
Figures of LLM-BLM_実験11 (図/). Reads experiments/F_fixes/results/{dev_details, final_summary, explore_fixes245,
explore_rank_logit}.json.

    python 資料集/02_実験/実験11_読ませ方とまとめ方の修正/作成スクリプト/figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
OUT = Path(__file__).resolve().parents[1] / '図'
RES = REPO / 'experiments' / 'F_fixes' / 'results'
BLUE, ORANGE, AQUA, GRAY, INK, INK2, VIOLET, YELLOW = ('#2a78d6', '#eb6834', '#1baf7a', '#b9b8b2', '#0b0b0b', '#52514e',
                                                       '#6250d6', '#eda100')
plt.rcParams.update({'font.family': 'BIZ UDPGothic', 'font.size': 9.5, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42})


def load(name):
    return json.loads((RES / name).read_text(encoding='utf-8'))


def fig_dev_fix1(dev):
    """Paired t of fix 1 against the current reading, per model and split (the keep rule needs t >= 1)."""
    models = ['dictionary', 'finbert', 'tfidf', 'chrono']
    names = ['辞書', 'FinBERT', 'TF-IDF', 'ChronoGPT']
    splits = ['B', 'A', 'C']
    t = np.zeros((len(models), len(splits)))
    for j, sp in enumerate(splits):
        for m in dev[sp]:
            for st in m['steps']:
                if st.get('variant') == 'fix1':
                    t[models.index(m['model']), j] = st['full']['t']
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    w = 0.25
    for j, (sp, c) in enumerate(zip(splits, [BLUE, VIOLET, AQUA])):
        ax.bar(np.arange(len(models)) + (j - 1) * w, t[:, j], width=w, color=c, label=f'切り方 {sp}')
    ax.axhline(1, color=ORANGE, lw=1.2, ls='--')
    ax.text(len(models) - 0.45, 1.05, '残す基準（t が 1 以上）', color=ORANGE, fontsize=8, ha='right', va='bottom')
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(models)), names)
    ax.set_ylabel('修正前との差の t')
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper left')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'dev_fix1.pdf')


def fig_explore_245(ex):
    """Exploratory: R2 gain over the numeric controls of each reading on the development period of split B."""
    rows = [('chrono', 'chrono_base', '今（平均・最後の層）'), ('chrono', 'chrono_mean', '修正 1'),
            ('chrono', 'chrono_mid', '修正 1＋4（真ん中の層）'), ('chrono', 'chrono_last', '修正 1＋4（最後のトークン）'),
            ('chrono', 'chrono_comp', '修正 1＋5（圧縮）'),
            ('finbert', 'finbert_net_tone', '今（先頭 200 文）'), ('finbert', 'fb_head200', '修正 1'),
            ('finbert', 'fb_headtail', '修正 1＋2（先頭＋末尾）'), ('finbert', 'fb_all', '修正 1＋2（全文）'),
            ('finbert', 'fb_comp', '修正 1＋5（圧縮）')]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, model, title, color in ((axes[0], 'chrono', 'ChronoGPT（埋め込み＋リッジ）', VIOLET),
                                    (axes[1], 'finbert', 'FinBERT（学習なし）', YELLOW)):
        r = ex['B'][model]
        sel = [x for x in rows if x[0] == model]
        vals, ts = [], []
        for _, col, _ in sel:
            if col in r:
                vals.append(100 * r[col]['full']['r2_gain_fix'])
                ts.append(r[col]['full']['t'])
            else:
                first = next(iter(r.values()))
                vals.append(100 * first['full']['r2_gain_base'])
                ts.append(None)
        y = np.arange(len(sel))[::-1]
        ax.barh(y, vals, color=[GRAY if t is None else (color if t >= 1 else '#d9d7d0') for t in ts], height=0.6)
        for yy, v, t in zip(y, vals, ts):
            ax.text(v + 0.01, yy, f'{v:.2f}' + ('' if t is None else f'（t {t:+.1f}）'), va='center', fontsize=7.5,
                    color=INK)
        ax.set_yticks(y, [s[2] for s in sel], fontsize=8)
        ax.set_xlim(0, max(vals) * 1.6)
        ax.set_title(title, fontsize=9, color=INK2)
        ax.set_xlabel('決定係数の増分（ポイント）', fontsize=8)
        ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'explore_245.pdf')


def fig_final_firm(fin):
    """Final comparison, firm level: R2 gain over the numeric controls (split B's final period and gpt-4o-mini's)."""
    order = [('logistic', 'ロジ\nスティック', GRAY), ('dictionary_base', '辞書', GRAY),
             ('finbert_base', 'Fin\nBERT', YELLOW), ('tfidf_base', 'TF-\nIDF', ORANGE), ('nlp_base', 'NLP\n3 つ', ORANGE),
             ('chrono_base', 'Chrono\nGPT', VIOLET), ('bert_ft', 'BERT\n追加\n学習', AQUA), ('gpt4o_base', 'gpt-\n4o-mini', BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), width_ratios=[1, 1.12])
    for ax, res, title in ((axes[0], fin['splits']['B'], '切り方 B の最終の期間（2020-01〜）'),
                           (axes[1], fin['gpt_period'], 'gpt-4o-mini の期間（2023-11〜）')):
        sel = [o for o in order if o[0] in res['firm']]
        v = [100 * res['firm'][k]['r2_gain'] for k, _, _ in sel]
        ax.bar(range(len(sel)), v, color=[c for _, _, c in sel], width=0.65)
        for i, x in enumerate(v):
            ax.text(i, x + 0.03, f'{x:.2f}', ha='center', fontsize=7.5, color=INK)
        ax.set_xticks(range(len(sel)), [n for _, n, _ in sel], fontsize=7.5)
        ax.set_title(title, fontsize=9, color=INK2)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('数字の統制に加えたときの\n決定係数の増分（ポイント）')
    fig.tight_layout()
    fig.savefig(OUT / 'final_firm.pdf')


def fig_final_sector(fin):
    """Final comparison, sector level: monthly excess return of the sector BLM after costs (95% interval)."""
    order = [('logistic', 'ロジスティック', GRAY), ('dictionary_base', '辞書', GRAY), ('finbert_base', 'FinBERT', YELLOW),
             ('tfidf_base', 'TF-IDF', ORANGE), ('chrono_base', 'ChronoGPT', VIOLET), ('bert_ft', 'BERT（追加学習）', AQUA)]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    w = 0.38
    for j, (sp, c, a) in enumerate([('B', INK2, 1.0), ('A', INK2, 0.45)]):
        res = fin['splits'][sp]['sector']
        v = [100 * res[k]['blm']['active_net_monthly'] for k, _, _ in order]
        se = [abs(x / res[k]['blm']['nw_t']) if res[k]['blm']['nw_t'] else 0 for x, (k, _, _) in zip(v, order)]
        ax.bar(np.arange(len(order)) + (j - 0.5) * w, v, width=w, color=[col for _, _, col in order], alpha=a,
               label=f'切り方 {sp}')
        ax.errorbar(np.arange(len(order)) + (j - 0.5) * w, v, yerr=[1.96 * s for s in se], fmt='none', ecolor=INK,
                    capsize=2, lw=0.8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(order)), [n for _, n, _ in order], fontsize=8)
    ax.set_ylabel('超過リターン（費用の後、%／月）')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'final_sector.pdf')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_dev_fix1(load('dev_details.json'))
    fig_explore_245(load('explore_fixes245.json'))
    fin = load('final_summary.json')
    fig_final_firm(fin)
    fig_final_sector(fin)
    print('saved to', OUT)
