"""
Numbers and tables of LLM-BLM_実験S2 (図/values.tex, 図/tab_*.tex), read from experiments/S2_smallcap_risk/results.
Every number in the report is a \\V{name} from values.tex or a generated table.

    python 資料集/02_実験/実験S2_小型株のリスク予測/作成スクリプト/values.py [--dry OUTDIR]
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
RES = REPO / 'experiments' / 'S2_smallcap_risk' / 'results'
OUT = Path(__file__).resolve().parents[1] / '図'
NAMES = {'chrono_risk': 'ChronoGPT＋リッジ（主）', 'chrono_mid_risk': 'ChronoGPT 26 層目＋リッジ',
         'dict_uncertainty': '辞書：不確実さの語', 'dict_negative': '辞書：否定的な語',
         'finbert_negative': 'FinBERT：否定的な文', 'gpt_flags': 'gpt-4o-mini：リスクの印の数',
         'gpt_risk': 'gpt-4o-mini：14 項目＋リッジ'}
TARGETS = {'log_rv_2_60': '変動率 2〜60 日', 'log_rv_2_21': '変動率 2〜21 日', 'log_ivol_2_60': '固有の変動率 2〜60 日',
           'abs_r_0_1': '2 日の反応の大きさ'}


def f(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return '--'
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    return s.replace('-', '$-$').replace('+', '$+$')


def pv(x):
    return '$<$0.001' if x < 0.001 else f(x, 3)


def unit(d):
    return '\\%' if d.get('unit') == '%' else ' bp'


def stat(prefix, d, v):
    v[f'{prefix}coef'] = f(d['coef'], 1 if d['unit'] == 'bp' else 2, True)
    v[f'{prefix}unit'] = unit(d)
    v[f'{prefix}t'] = f(d['t'], 2, True)
    v[f'{prefix}p'] = pv(d['p_one_sided'])
    v[f'{prefix}lo'] = f(d['ci95'][0], 2, True)
    v[f'{prefix}hi'] = f(d['ci95'][1], 2, True)
    v[f'{prefix}n'] = f"{d['n']:,}"
    v[f'{prefix}dates'] = f"{d['event_dates']:,}"
    v[f'{prefix}rtwo'] = f(d['r2_gain_pt'], 3)


def main():
    dry = '--dry' in sys.argv
    res = RES / 'dry' if dry else RES
    out = Path(sys.argv[sys.argv.index('--dry') + 1]) if dry else OUT
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((res / 'summary.json').read_text(encoding='utf-8'))
    v = {'commit': s['commit'][:7], 'npanel': f"{s['n_panel']:,}", 'neval': f"{s['n_eval']:,}",
         'nevalgpt': f"{s['n_eval_gpt']:,}"}
    stat('P', s['primary'], v)
    v['Pperm'] = f(s['primary']['permutation_p_one_sided'], 4)
    v['Pverdict'] = {'supported': '支持', 'not supported': '支持されない'}[s['primary']['verdict']]
    stat('R', s['robust_2021'], v)
    stat('N', s['robust_no_reaction'], v)
    fc = s['forecast']
    v.update({'Fn': f"{fc['n']:,}", 'Fbase': f(100 * fc['r2_oos_base'], 2), 'Fwith': f(100 * fc['r2_oos_with_score'], 2),
              'Fgain': f(100 * (fc['r2_oos_with_score'] - fc['r2_oos_base']), 2, True), 'Fratio': f(fc['mse_ratio'], 4),
              'Ft': f(fc['loss_diff_t'], 2, True)})
    rows = []
    for fam, label in (('secondary', '2020 年以降'), ('secondary_gpt_period', '2024 年以降')):
        for m, d in s[fam].items():
            name = m.replace('_', '')
            if 'coef' not in d:
                rows.append(f"{label} & {NAMES[m]} & \\multicolumn{{5}}{{l}}{{（発表が {d['n']} 件で、推定しない）}} \\\\")
                continue
            v[f'{name}coef{fam[:3]}'] = f(d['coef'], 2, True)
            v[f'{name}t{fam[:3]}'] = f(d['t'], 2, True)
            rows.append(f"{label} & {NAMES[m]} & {f(d['coef'], 2, True)}\\% & {f(d['t'], 2, True)} & "
                        f"{pv(d['p_one_sided'])} & {pv(d['holm_p'])} & {d['n']:,} \\\\")
    (out / 'tab_secondary.tex').write_text(
        '\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n期間 & 点数 & 係数 & $t$ & 片側 $p$ & Holm の $p$ & 発表の数 \\\\\n'
        '\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
    c = pd.read_csv(res / 'comparison.csv')
    for period, tag in (('2020-', '2020'), ('2024- (gpt)', '2024')):
        d = c[c.period == period]
        lines = []
        for m in [m for m in NAMES if m in set(d.model)]:
            cells = []
            for t in TARGETS:
                r = d[(d.model == m) & (d.target == t)]
                if r.empty or pd.isna(r.t.iloc[0]):
                    cells.append('--')
                else:
                    u = '' if t != 'abs_r_0_1' else ''
                    cells.append(f"{f(r.coef.iloc[0], 1 if t == 'abs_r_0_1' else 2, True)}{u} ({f(r.t.iloc[0], 2, True)})")
            lines.append(f'{NAMES[m]} & ' + ' & '.join(cells) + ' \\\\')
        n = d.groupby('target').n.max()
        lines.append('\\midrule\n発表の数（最大） & ' + ' & '.join(f"{int(n.get(t, 0)):,}" for t in TARGETS) + ' \\\\')
        (out / f'tab_compare_{tag}.tex').write_text(
            '\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n点数 & ' + ' & '.join(TARGETS.values()) + ' \\\\\n'
            '単位 & \\% & \\% & \\% & bp \\\\\n\\midrule\n' + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n',
            encoding='utf-8')
    # the out-of-sample forecasts by calendar year (descriptive, added after the run)
    fcs = pd.read_csv(res / 'forecasts.csv')
    fcs['year'] = fcs.event_date.str[:4]
    lines = []
    for year, g in fcs.groupby('year'):
        mb = ((g.log_rv_2_60 - g.pred_base) ** 2).mean()
        mw = ((g.log_rv_2_60 - g.pred_with_score) ** 2).mean()
        lines.append(f"{year} & {len(g):,} & {f(mb, 4)} & {f(mw, 4)} & {f(mw / mb, 3)} & "
                     f"{f((g.log_rv_2_60 - g.pred_base).mean(), 3, True)} \\\\")
    (out / 'tab_forecast_year.tex').write_text(
        '\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n年 & 発表の数 & 二乗誤差（基準） & 二乗誤差（＋点数） & 比 & 平均の誤差（基準） \\\\\n'
        '\\midrule\n' + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
    (out / 'values.tex').write_text(
        '% values.py が results から作る。手で直さない。\n'
        + '\n'.join(f'\\expandafter\\def\\csname v@{k}\\endcsname{{{val}}}' for k, val in v.items()) + '\n',
        encoding='utf-8')
    print(len(v), 'values written to', out)


if __name__ == '__main__':
    main()
