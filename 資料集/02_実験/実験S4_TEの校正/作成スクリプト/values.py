"""
Numbers and tables of LLM-BLM_実験S4 (図/values.tex, 図/tab_*.tex), read from experiments/S4_te_calibration/results.

    python 資料集/02_実験/実験S4_TEの校正/作成スクリプト/values.py [--dry OUTDIR]
"""
import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
RES = REPO / 'experiments' / 'S4_te_calibration' / 'results'
OUT = Path(__file__).resolve().parents[1] / '図'
VIEW = {'chrono_ridge': 'ChronoGPT', 'gpt_ridge': 'gpt-4o-mini'}
VERSION = {'A': 'A 基準', 'A_cal': 'A＋校正', 'B': 'B 株価で補正', 'B_cal': 'B＋校正'}
FREQ = {'quarter': '四半期', 'half': '半年', 'year': '1 年'}


def f(x, nd=2, sign=False):
    if x is None:
        return '--'
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    return s.replace('-', '$-$').replace('+', '$+$')


def pct(x, nd=1, sign=False):
    return '--' if x is None else f(100 * x, nd, sign) + '\\%'


def alnum(s):
    return ''.join(ch for ch in s if ch.isalnum())


def main():
    dry = '--dry' in sys.argv
    res = RES / 'dry' if dry else RES
    out = Path(sys.argv[sys.argv.index('--dry') + 1]) if dry else OUT
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((res / 'summary.json').read_text(encoding='utf-8'))
    v = {'commit': s['commit'][:7]}
    for name, t in s['tests'].items():
        tag = alnum(name)
        v.update({f'{tag}D': f(t['D'], 3, True), f'{tag}lo': f(t['ci95'][0], 3, True), f'{tag}hi': f(t['ci95'][1], 3, True),
                  f'{tag}p': '$<$0.001' if t['p_one_sided'] < 0.001 else f(t['p_one_sided'], 3),
                  f'{tag}gref': f(t['b_ref'], 2), f'{tag}gnew': f(t['b_new'], 2), f'{tag}days': f"{t['days']:,}",
                  f'{tag}teref': pct(0.03 * t['b_ref'], 1), f'{tag}tenew': pct(0.03 * t['b_new'], 1)})
    v['primaryverdict'] = {'supported': '支持', 'not supported': '支持されない'}[s['tests']['primary']['verdict']]
    runs = s['runs']
    for start in ('2015', '2023'):
        lines = []
        for freq in FREQ:
            keys = [k for k in runs if k.startswith(f'{start}/{freq}/') and not k.endswith('/band')]
            for k in keys:
                r = runs[k]
                _, _, model, ver = k.split('/')
                lines.append(f"{FREQ[freq]} & {VIEW[model]} & {VERSION[ver]} & {f(r['final_value'], 3)} & "
                             f"{pct(r['annual_excess_over_band'], 1, True)} & {pct(r['limit_mean'])} & "
                             f"{pct(r['te_realized_eval'])} & {f(r['g_eval'], 2)} & {f(r['ir_eval'], 2, True)} & "
                             f"{pct(r['turnover_mean'], 0)} \\\\")
                for field, fmt in (('final_value', lambda x: f(x, 3)), ('te_realized_eval', pct),
                                   ('limit_mean', pct), ('ir_eval', lambda x: f(x, 2, True)),
                                   ('annual_excess_over_band', lambda x: pct(x, 1, True))):
                    v[alnum(f'{k}{field}')] = fmt(r[field])
            band = runs[f'{start}/{freq}/band']
            v[alnum(f'{start}/{freq}/band')] = f(band['final_value'], 3)
            lines.append(f"{FREQ[freq]} & \\multicolumn{{2}}{{l}}{{範囲の全社（比べる相手）}} & {f(band['final_value'], 3)} & "
                         f"-- & -- & -- & -- & -- & {pct(band['turnover_mean'], 0)} \\\\")
            lines.append('\\addlinespace')
        (out / f'tab_runs_{start}.tex').write_text(
            '\\begin{tabular}{@{}lllrrrrrrr@{}}\n\\toprule\nリバランス & 見通し & 版 & 最終の値 & 相手との差 & 上限の平均 & '
            '実現の TE & $g$ & 情報レシオ & 回転率 \\\\\n\\midrule\n' + '\n'.join(lines[:-1])
            + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
    (out / 'values.tex').write_text(
        '% values.py が results から作る。手で直さない。\n'
        + '\n'.join(f'\\expandafter\\def\\csname v@{k}\\endcsname{{{val}}}' for k, val in v.items()) + '\n',
        encoding='utf-8')
    print(len(v), 'values written to', out)


if __name__ == '__main__':
    main()
