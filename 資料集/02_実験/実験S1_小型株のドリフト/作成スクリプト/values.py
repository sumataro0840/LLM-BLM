"""
Numbers and tables of LLM-BLM_実験S1 (図/values.tex, 図/tab_*.tex), read from experiments/S1_smallcap_drift/results.
Nothing is typed by hand into the report: every number there is a \\V{name} from values.tex or a generated table.

    python 資料集/02_実験/実験S1_小型株のドリフト/作成スクリプト/values.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
RES = REPO / 'experiments' / 'S1_smallcap_drift' / 'results'
OUT = Path(__file__).resolve().parents[1] / '図'
NAMES = {'dictionary': '辞書（学習なし）', 'finbert': 'FinBERT（学習なし）', 'finbert_ft': 'FinBERT（追加学習）',
         'chrono_ridge': 'ChronoGPT＋リッジ（主）', 'chrono_mid_ridge': 'ChronoGPT 26 層目＋リッジ',
         'gpt': 'gpt-4o-mini（学習なし）', 'gpt_ridge': 'gpt-4o-mini（重みを学習）', 'band': '範囲の全社（比べる相手）'}
TARGETS = {'abn_r_0_1': '2 日の反応', 'abn_drift_2_60': '2〜60 日', 'abn_drift_2_21': '2〜21 日'}
FREQ = {'quarter': '四半期', 'half': '半年', 'year': '1 年'}


def f(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return '--'
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    return s.replace('-', '$-$') if not sign else s.replace('-', '$-$').replace('+', '$+$')


def pv(x):
    return '$<$0.001' if x < 0.001 else f(x, 3)


def pct(x, nd=2, sign=True):
    return '--' if x is None else f(100 * x, nd, sign) + '\\%'


def stat_macros(prefix, d, v):
    v[f'{prefix}coef'] = f(d['coef_bp_per_sd'], 1, True)
    v[f'{prefix}t'] = f(d['t'], 2, True)
    v[f'{prefix}p'] = pv(d['p_one_sided'])
    v[f'{prefix}lo'] = f(d['ci95_bp'][0], 1, True)
    v[f'{prefix}hi'] = f(d['ci95_bp'][1], 1, True)
    v[f'{prefix}n'] = f"{d['n']:,}"
    v[f'{prefix}dates'] = f"{d['event_dates']:,}"
    v[f'{prefix}rtwo'] = f(d['r2_gain_pt'], 3)
    if 'holm_p' in d:
        v[f'{prefix}holm'] = pv(d['holm_p'])


def main():
    res = RES / ('dry' if '--dry' in sys.argv else '')
    out = OUT if '--dry' not in sys.argv else Path(sys.argv[sys.argv.index('--dry') + 1])
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((res / 'summary.json').read_text(encoding='utf-8'))
    v = {'commit': s['commit'][:7], 'npanel': f"{s['n_panel']:,}", 'neval': f"{s['n_eval']:,}",
         'nevalgpt': f"{s['n_eval_gpt']:,}", 'dry': 'はい' if s['dry'] else 'いいえ'}
    p = s['primary']
    stat_macros('P', p, v)
    v['Pperm'] = f(p['permutation_p_one_sided'], 4)
    v['Pverdict'] = {'supported': '支持', 'not supported': '支持されない'}.get(p['verdict'], p['verdict'])
    stat_macros('R', s['robust_2021'], v)
    stat_macros('E', s['explore_ridge_on_drift'], v)
    for key, port in (('M', s['portfolio']), ('G', s['portfolio_gpt_ridge'])):
        v[f'{key}months'] = str(port['months'])
        for c, tag in (('0.0', 'a'), ('0.003', 'b'), ('0.005', 'c')):
            v[f'{key}ex{tag}'] = pct(port[f'excess_monthly_cost_{c}'])
            v[f'{key}t{tag}'] = f(port[f'nw_t_cost_{c}'], 2, True)
        v[f'{key}turn'] = pct(port['turnover_top_mean'], 0, False)
    for key, name in (('M', 'by_month.csv'), ('G', 'by_month_gpt.csv')):
        bm = pd.read_csv(res / name)
        v[f'{key}top'] = pct(bm.ret_top.mean())
        v[f'{key}all'] = pct(bm.ret_all.mean())
        v[f'{key}start'] = str(bm.month_end.iloc[0])[:7]
        v[f'{key}end'] = str(bm.month_end.iloc[-1])[:7]

    # secondary tests (Holm within each family); macros <model>coefT/tT (2020-) and <model>coefS/tS (2024-)
    for fam, suffix in (('secondary_drift', 'T'), ('secondary_drift_gpt_period', 'S')):
        for m, d in s[fam].items():
            name = m.replace('_', '')
            v[f'{name}coef{suffix}'] = f(d['coef_bp_per_sd'], 1, True)
            v[f'{name}t{suffix}'] = f(d['t'], 2, True)
    rows = []
    for fam, label in (('secondary_drift', '2020 年以降'), ('secondary_drift_gpt_period', '2024 年以降')):
        for m, d in s[fam].items():
            rows.append(f"{label} & {NAMES[m]} & {f(d['coef_bp_per_sd'], 1, True)} & {f(d['t'], 2, True)} & "
                        f"{pv(d['p_one_sided'])} & {pv(d['holm_p'])} & {d['n']:,} \\\\")
    (out / 'tab_secondary.tex').write_text(
        '\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n期間 & 点数 & 係数（bp/SD） & $t$ & 片側 $p$ & Holm の $p$ & 発表の数 \\\\\n'
        '\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')

    # comparison: t of every model x target x period
    c = pd.read_csv(res / 'comparison.csv')
    for period, tag in (('2020-', '2020'), ('2024- (gpt)', '2024')):
        d = c[c.period == period]
        models = [m for m in NAMES if m in set(d.model)]
        lines = []
        for m in models:
            cells = []
            for t in TARGETS:
                r = d[(d.model == m) & (d.target == t)]
                cells.append('--' if r.empty or pd.isna(r.t.iloc[0]) else
                             f"{f(r.coef_bp_per_sd.iloc[0], 1, True)} ({f(r.t.iloc[0], 2, True)})")
            lines.append(f"{NAMES[m]} & " + ' & '.join(cells) + ' \\\\')
        n = d.groupby('target').n.max()
        lines.append('\\midrule\n発表の数（最大） & ' + ' & '.join(f"{int(n.get(t, 0)):,}" for t in TARGETS) + ' \\\\')
        (out / f'tab_compare_{tag}.tex').write_text(
            '\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n点数 & ' + ' & '.join(TARGETS.values()) + ' \\\\\n\\midrule\n'
            + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')

    # BLM paths (addendum 1)
    b = json.loads((res / 'blm_paths.json').read_text(encoding='utf-8')) if (res / 'blm_paths.json').exists() else {}
    for start in (2015, 2023):
        lines = []
        for freq in FREQ:
            keys = [k for k in b if k.startswith(f'{start}/{freq}/')]
            order = ['chrono_ridge', 'finbert_ft', 'gpt', 'gpt_ridge', 'band']
            keys.sort(key=lambda k: order.index(k.split('/')[2]))
            for k in keys:
                r = b[k]
                m = k.split('/')[2]
                status = ', '.join(f'{a} {n}' for a, n in r['status'].items())
                lines.append(f"{FREQ[freq]} & {NAMES[m]} & {f(r['final_value'], 3)} & {pct(r['annual_return'], 1)} & "
                             f"{pct(r.get('annual_excess_over_band'), 1)} & {pct(r['te_realized_mean'], 1, False)} & "
                             f"{pct(r['turnover_mean'], 0, False)} & {r['rebalances']} \\\\")
            lines.append('\\addlinespace')
        (out / f'tab_blm_{start}.tex').write_text(
            '\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\nリバランス & ポートフォリオ & 最終の値 & 年率 & 相手との差 & 実現の TE & 回転率 & 回数 \\\\\n'
            '\\midrule\n' + '\n'.join(lines[:-1]) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
        for k in b:
            if k.startswith(f'{start}/'):
                st, fr, m = k.split('/')
                v[f'B{st}{fr}{m}'.replace('_', '')] = f(b[k]['final_value'], 3)
                v[f'B{st}{fr}{m}ex'.replace('_', '')] = pct(b[k].get('annual_excess_over_band'), 1)
    # exploration (explore.py, after the preregistered run)
    if (res / 'explore.json').exists():
        e = json.loads((res / 'explore.json').read_text(encoding='utf-8'))
        for group in ('days_22_60', 'days_22_60_gpt', 'plus_momentum', 'plus_momentum_gpt', 'by_size'):
            for m, d in e[group].items():
                tag = ''.join(ch for ch in f'X{group}{m}' if ch.isalnum())
                v[f'{tag}c'] = f(d['coef_bp_per_sd'], 1, True)
                v[f'{tag}t'] = f(d['t'], 2, True)
                v[f'{tag}n'] = f"{d['n']:,}"
        for m, r in {**e['corr_score_mom'], **e['corr_score_mom_gpt']}.items():
            v[''.join(ch for ch in f'Xcorr{m}' if ch.isalnum())] = f(r, 2)
        lines = []
        for key, label in (('quintiles_chrono', 'ChronoGPT＋リッジ（2020 年以降）'), ('quintiles_gpt', 'gpt-4o-mini（2024 年以降）')):
            for y, name in (('abn_r_0_1', '2 日の反応'), ('abn_drift_2_21', '2〜21 日'), ('abn_drift_22_60', '22〜60 日'),
                            ('abn_drift_2_60', '2〜60 日')):
                q = e[key][y]
                lines.append(f"{label if y == 'abn_r_0_1' else ''} & {name} & "
                             + ' & '.join(f(round(q[str(i)]) + 0.0, 0, True) if round(q[str(i)]) else '0' for i in range(1, 6))
                             + f" & {f(q['5'] - q['1'], 0, True)} \\\\")
            lines.append('\\addlinespace')
        (out / 'tab_quintiles.tex').write_text(
            '\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n点数 & 期間 & 1（低） & 2 & 3 & 4 & 5（高） & 5 − 1 \\\\\n\\midrule\n'
            + '\n'.join(lines[:-1]) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
    (out / 'values.tex').write_text(
        '% values.py が results から作る。手で直さない。\n'
        + '\n'.join(f'\\expandafter\\def\\csname v@{k}\\endcsname{{{val}}}' for k, val in v.items()) + '\n',
        encoding='utf-8')
    print(len(v), 'values written to', out)


if __name__ == '__main__':
    main()
