"""
Numbers and tables of LLM-BLM_実験R1 (図/values.tex, 図/tab_*.tex), read from experiments/R1_replication/results
and the results_rep/ of S1-S4.

    python 資料集/02_実験/実験R1_新しい1000社での再現/作成スクリプト/values.py
"""
import json
from pathlib import Path

import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
EXP = REPO / 'experiments'
RES = EXP / 'R1_replication' / 'results'
OUT = Path(__file__).resolve().parents[1] / '図'
DIRS = {'S1': 'S1_smallcap_drift', 'S2': 'S2_smallcap_risk', 'S3': 'S3_smallcap_risk_blm', 'S4': 'S4_te_calibration'}
KEYS = {'2015/quarter/chrono_ridge/B_cal': ('2015 年末', '四半期', 'ChronoGPT'),
        '2015/half/chrono_ridge/B_cal': ('2015 年末', '半年', 'ChronoGPT'),
        '2015/year/chrono_ridge/B_cal': ('2015 年末', '1 年', 'ChronoGPT'),
        '2023/quarter/chrono_ridge/B_cal': ('2023 年末', '四半期', 'ChronoGPT'),
        '2023/half/chrono_ridge/B_cal': ('2023 年末', '半年', 'ChronoGPT'),
        '2023/year/chrono_ridge/B_cal': ('2023 年末', '1 年', 'ChronoGPT'),
        '2023/quarter/gpt_ridge/B_cal': ('2023 年末', '四半期', 'gpt-4o-mini')}


def f(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return '--'
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    return s.replace('-', '$-$').replace('+', '$+$')


def pct(x, nd=1, sign=False):
    return '--' if x is None else f(100 * x, nd, sign) + '\\%'


def pv(x):
    return '--' if x is None else ('$<$0.001' if x < 0.001 else f(x, 3))


def load(name, sub):
    return json.loads((EXP / DIRS[name] / sub / 'summary.json').read_text(encoding='utf-8'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads((RES / 'summary.json').read_text(encoding='utf-8'))
    rep = {k: load(k, 'results_rep') for k in DIRS}
    orig = {k: load(k, 'results') for k in DIRS}
    v = {'commit': s['commit'][:7], 'npanel': f"{rep['S1']['n_panel']:,}", 'neval': f"{rep['S1']['n_eval']:,}",
         'nevalgpt': f"{rep['S1']['n_eval_gpt']:,}", 'npanelorig': f"{orig['S1']['n_panel']:,}",
         'nevalorig': f"{orig['S1']['n_eval']:,}"}
    names = {'R-S1': 'RSone', 'R-S2': 'RStwo', 'R-S3': 'RSthree', 'R-S4': 'RSfour'}
    for k, ok in s['replicated'].items():
        v[names[k]] = '再現した' if ok else '再現しなかった'
    v['nreplicated'] = str(sum(s['replicated'].values()))
    b = s['R-BLM']
    v.update({'Bex': pct(b['annual_excess'], 1, True), 'Bte': pct(b['te_realized']), 'Bir': f(b['ir'], 2, True),
              'Blo': pct(b['ci95_annual'][0], 1, True), 'Bhi': pct(b['ci95_annual'][1], 1, True), 'Bp': pv(b['p_one_sided']),
              'Bdays': f"{b['days']:,}", 'Bverdict': {'supported': '支持', 'not supported': '支持されない'}[b['verdict']]})
    o = s['blm_original_exploratory']['2015/quarter/chrono_ridge/B_cal']
    v.update({'Oex': pct(o['annual_excess'], 1, True), 'Oir': f(o['ir'], 2, True), 'Op': pv(o['p_one_sided'])})
    # primary numbers of S1-S4 in both samples
    for tag, smp in (('o', orig), ('r', rep)):
        p1, p2 = smp['S1']['primary'], smp['S2']['primary']
        t3, t4 = smp['S3']['tests']['primary'], smp['S4']['tests']['primary']
        v.update({f'{tag}Sonec': f(p1['coef_bp_per_sd'], 1, True), f'{tag}Sonet': f(p1['t'], 2, True),
                  f'{tag}Soneperm': f(p1['permutation_p_one_sided'], 4),
                  f'{tag}Stwoc': f(p2['coef'], 2, True), f'{tag}Stwot': f(p2['t'], 2, True),
                  f'{tag}Stwoperm': f(p2['permutation_p_one_sided'], 4),
                  f'{tag}Sthreeb': f(t3['b_ref'], 3), f'{tag}Sthreec': f(t3['b_new'], 3), f'{tag}Sthreep': pv(t3['p_one_sided']),
                  f'{tag}Sfourb': f(t4['b_ref'], 2), f'{tag}Sfourc': f(t4['b_new'], 2), f'{tag}Sfourp': pv(t4['p_one_sided']),
                  f'{tag}Sfourte': pct(0.03 * t4['b_new']),
                  f'{tag}gptc': f(smp['S1']['secondary_drift_gpt_period']['gpt'].get('coef_bp_per_sd'), 1, True),
                  f'{tag}gptt': f(smp['S1']['secondary_drift_gpt_period']['gpt'].get('t'), 2, True),
                  f'{tag}port': pct(smp['S1']['portfolio']['excess_monthly_cost_0.003'], 2, True),
                  f'{tag}portt': f(smp['S1']['portfolio']['nw_t_cost_0.003'], 2, True),
                  f'{tag}final': f(smp['S4']['runs']['2015/quarter/chrono_ridge/B_cal']['final_value'], 3),
                  f'{tag}band': f(smp['S4']['runs']['2015/quarter/band']['final_value'], 3)})
    # BLM tests in both samples
    lines = []
    allrep = {'2015/quarter/chrono_ridge/B_cal': b, **s['blm_secondary']}
    for key, (start, freq, view) in KEYS.items():
        r, oo = allrep.get(key), s['blm_original_exploratory'].get(key)
        cells = []
        for x in (oo, r):
            cells += ['--'] * 3 if x is None else [pct(x['annual_excess'], 1, True), f(x['ir'], 2, True), pv(x['p_one_sided'])]
        mark = '\\textbf{主}' if key == '2015/quarter/chrono_ridge/B_cal' else ''
        lines.append(f'{start} & {freq} & {view} & {mark} & ' + ' & '.join(cells) + ' \\\\')
    (OUT / 'tab_blm.tex').write_text(
        '\\begin{tabular}{@{}llll rrr rrr@{}}\n\\toprule\n & & & & \\multicolumn{3}{c}{元の標本（探索）} & '
        '\\multicolumn{3}{c}{新しい標本} \\\\\n\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n始点 & リバランス & 見通し & & '
        '相手との差 & 情報レシオ & 片側 $p$ & 相手との差 & 情報レシオ & 片側 $p$ \\\\\n\\midrule\n' + '\n'.join(lines)
        + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
    explore(v)
    (OUT / 'values.tex').write_text(
        '% values.py が results から作る。手で直さない。\n'
        + '\n'.join(f'\\expandafter\\def\\csname v@{k}\\endcsname{{{val}}}' for k, val in v.items()) + '\n',
        encoding='utf-8')
    print(len(v), 'values written to', OUT)


THRESHOLD_NAMES = {'none': 'そのまま（S4 と同じ）', 'main_300_80': '$+$300\\%・$-$80\\% を除く（主）',
                   'loose_100_50': '$+$100\\%・$-$50\\% を除く', 'tight_1000_90': '$+$1000\\%・$-$90\\% を除く'}


def explore(v):
    """Numbers and the table of the exploration (experiments/R1_replication/explore.py), when its results exist."""
    ex = {s: RES / f'explore_{s}.json' for s in ('orig', 'rep')}
    if not all(p.exists() for p in ex.values()):
        return
    ex = {s: json.loads(p.read_text(encoding='utf-8')) for s, p in ex.items()}
    for s, tag in (('orig', 'o'), ('rep', 'r')):
        e = ex[s]['extreme_days']
        v.update({f'{tag}nup': f"{e['above_300pct']:,}", f'{tag}nupone': f"{e['above_100pct']:,}",
                  f'{tag}ndown': f"{e['below_-80pct']:,}", f'{tag}nstockdays': f"{e['stock_days']:,}"})
        top = ex[s]['largest_days_of_S4']
        for ver, name in (('A', 'A'), ('B', 'B'), ('B_cal', 'Bcal')):
            t = top[ver]
            v.update({f'{tag}g{name}': f(t['g'], 2), f'{tag}gwo{name}': f(t['g_without_top5_days'], 2),
                      f'{tag}share{name}': pct(t['top5_share_of_squares'], 0)})
        t = top['B_cal']['top_days'][0]
        v.update({f'{tag}topday': t['date'], f'{tag}topactive': pct(t['active_return'], 0, True)})
        runs = ex[s]['runs']
        for th, tname in (('none', 'none'), ('main_300_80', 'main')):
            for ver, name in (('A', 'A'), ('A_cal', 'Acal'), ('B', 'B'), ('B_cal', 'Bcal')):
                r = runs[th][ver]
                m = r['R_BLM_test']
                v.update({f'{tag}{tname}{name}g': f(r['g_eval'], 2), f'{tag}{tname}{name}te': pct(m['te_realized']),
                          f'{tag}{tname}{name}ex': pct(m['annual_excess'], 1, True), f'{tag}{tname}{name}ir': f(m['ir'], 2, True),
                          f'{tag}{tname}{name}p': pv(m['p_one_sided']), f'{tag}{tname}{name}final': f(r['final_value'], 2),
                          f'{tag}{tname}{name}lo': pct(m['ci95_annual'][0], 1, True),
                          f'{tag}{tname}{name}hi': pct(m['ci95_annual'][1], 1, True)})
            v[f'{tag}{tname}band'] = f(runs[th]['band']['final_value'], 2)
        v[f'{tag}nremoved'] = f"{runs['main_300_80']['n_returns_removed']:,}"
        ok = all(abs(c['final_value_S4'] - c['final_value_here']) < 1e-6 for c in ex[s]['reproduces_S4'].values())
        v[f'{tag}reproduces'] = '一致した' if ok else '一致しなかった'
    # the largest jumps of each sample
    for s, tag in (('orig', 'o'), ('rep', 'r')):
        big = ex[s]['extreme_days']['largest'][:3]
        sign = lambda x: f'{100 * x:+,.0f}'.replace('+', '$+$').replace('-', '$-$') + '\\%'
        v[f'{tag}bigjumps'] = '、'.join(f"{b['symbol']}（{b['date']}、{sign(b['return'])}）" for b in big)
    rows = []
    for s, sname in (('orig', '元の標本'), ('rep', '新しい標本')):
        for th, tname in THRESHOLD_NAMES.items():
            r = ex[s]['runs'][th]
            b, a = r['B_cal'], r['A_cal']
            m = b['R_BLM_test']
            rows.append(f"{sname} & {tname} & {r['n_returns_removed']:,} & {f(b['g_eval'], 2)} & {pct(m['te_realized'])} & "
                        f"{pct(m['annual_excess'], 1, True)} & {f(m['ir'], 2, True)} & {pv(m['p_one_sided'])} & "
                        f"{f(a['g_eval'], 2)} & {f(b['final_value'], 2)} & {f(r['band']['final_value'], 2)} \\\\")
        rows.append('\\addlinespace')
    (OUT / 'tab_explore.tex').write_text(
        '\\begin{tabular}{@{}ll r rrrrr r rr@{}}\n\\toprule\n & & & \\multicolumn{5}{c}{B＋校正（R-BLM の主なポートフォリオ）} & '
        'A＋校正 & \\multicolumn{2}{c}{最終の値} \\\\\n\\cmidrule(lr){4-8}\\cmidrule(lr){9-9}\\cmidrule(lr){10-11}\n'
        '標本 & 日々のリターンの扱い & 除いた数 & $g$ & 実現の TE & 相手との差 & 情報レシオ & 片側 $p$ & $g$ & BLM & 相手 \\\\\n'
        '\\midrule\n' + '\n'.join(rows[:-1]) + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')


if __name__ == '__main__':
    main()
