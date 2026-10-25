"""
Numbers and tables of LLM-BLM_実験S3 (図/values.tex, 図/tab_*.tex), read from experiments/S3_smallcap_risk_blm/results.

    python 資料集/02_実験/実験S3_リスクで補正したBLM/作成スクリプト/values.py [--dry OUTDIR]
"""
import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'run_events.py').exists())
RES = REPO / 'experiments' / 'S3_smallcap_risk_blm' / 'results'
OUT = Path(__file__).resolve().parents[1] / '図'
VIEW = {'chrono_ridge': 'ChronoGPT', 'gpt_ridge': 'gpt-4o-mini'}
VERSION = {'A': 'A 基準', 'B': 'B 株価で補正', 'C': 'C 株価＋テキスト', 'C_gpt': 'C 株価＋テキスト（gpt）'}
FREQ = {'quarter': '四半期', 'half': '半年', 'year': '1 年'}


def f(x, nd=2, sign=False):
    if x is None:
        return '--'
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    return s.replace('-', '$-$').replace('+', '$+$')


def pct(x, nd=1, sign=False):
    return '--' if x is None else f(100 * x, nd, sign) + '\\%'


def main():
    dry = '--dry' in sys.argv
    res = RES / 'dry' if dry else RES
    out = Path(sys.argv[sys.argv.index('--dry') + 1]) if dry else OUT
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((res / 'summary.json').read_text(encoding='utf-8'))
    v = {'commit': s['commit'][:7]}
    for name, t in s['tests'].items():
        tag = ''.join(ch for ch in name if ch.isalnum())
        v.update({f'{tag}D': f(t['D'], 4, True), f'{tag}lo': f(t['ci95'][0], 4, True), f'{tag}hi': f(t['ci95'][1], 4, True),
                  f'{tag}p': ('$<$0.001' if t['p_one_sided'] < 0.001 else f(t['p_one_sided'], 3)), f'{tag}bref': f(t['b_ref'], 3), f'{tag}bnew': f(t['b_new'], 3),
                  f'{tag}days': f"{t['days']:,}"})
    v['primaryverdict'] = {'supported': '支持', 'not supported': '支持されない'}[s['tests']['primary']['verdict']]
    for ver, share in s['scaling']['used_share'].items():
        v[f'used{ver.replace("_", "")}'] = pct(share, 0)
    for ver, beta in s['scaling'].get('beta_score_mean', {}).items():
        v[f'beta{ver.replace("_", "")}'] = f(100 * beta, 2, True)
    runs = s['runs']
    for start in ('2015', '2023'):
        lines = []
        for freq in FREQ:
            keys = [k for k in runs if k.startswith(f'{start}/{freq}/') and not k.endswith('/band')]
            for k in keys:
                r = runs[k]
                _, _, model, ver = k.split('/')
                lines.append(f"{FREQ[freq]} & {VIEW[model]} & {VERSION[ver]} & {f(r['final_value'], 3)} & "
                             f"{pct(r['annual_return'], 1, True)} & {pct(r['annual_excess_over_band'], 1, True)} & "
                             f"{pct(r['te_ex_mean'])} & {pct(r['te_realized_mean'])} & {f(r['bias_b'], 2)} & "
                             f"{pct(r['turnover_mean'], 0)} \\\\")
            band = runs[f'{start}/{freq}/band']
            lines.append(f"{FREQ[freq]} & \\multicolumn{{2}}{{l}}{{範囲の全社（比べる相手）}} & {f(band['final_value'], 3)} & "
                         f"{pct(band['annual_return'], 1, True)} & -- & -- & -- & -- & {pct(band['turnover_mean'], 0)} \\\\")
            lines.append('\\addlinespace')
            for k in keys + [f'{start}/{freq}/band']:
                v['B' + ''.join(ch for ch in k if ch.isalnum())] = f(runs[k]['final_value'], 3)
                if not k.endswith('/band'):
                    v['X' + ''.join(ch for ch in k if ch.isalnum())] = pct(runs[k]['annual_excess_over_band'], 1, True)
                    v['b' + ''.join(ch for ch in k if ch.isalnum())] = f(runs[k]['bias_b'], 2)
        (out / f'tab_runs_{start}.tex').write_text(
            '\\begin{tabular}{@{}lllrrrrrrr@{}}\n\\toprule\nリバランス & 見通し & 共分散 & 最終の値 & 年率 & 相手との差 & '
            '事前の TE & 実現の TE & 偏り b & 回転率 \\\\\n\\midrule\n' + '\n'.join(lines[:-1])
            + '\n\\bottomrule\n\\end{tabular}\n', encoding='utf-8')
    (out / 'values.tex').write_text(
        '% values.py が results から作る。手で直さない。\n'
        + '\n'.join(f'\\expandafter\\def\\csname v@{k}\\endcsname{{{val}}}' for k, val in v.items()) + '\n',
        encoding='utf-8')
    print(len(v), 'values written to', out)


if __name__ == '__main__':
    main()
