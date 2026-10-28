r"""
values.json (compute.py) -> values.tex: one macro per number, used by the report as \V{key}, so that no number is
copied by hand. Negative numbers use '-' and are meant to be set in math mode.

    python 資料集/02_実験/計算例_LLMからポートフォリオまで/作成スクリプト/make_tex.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
v = json.loads((HERE.parent / 'values.json').read_text(encoding='utf-8'))
out = {}


def f(x, d):
    s = f'{x:,.{d}f}'
    return '0' + s[2:] if s.startswith('-0') and float(s) == 0 else s


def pct(x, d=2):
    return f(100 * x, d)


r = v['ridge']
out.update({'ntrain': f(r['n_train'], 0), 'trfrom': r['train_from'], 'trto': r['train_to'],
            'wemax': r['window_end_max'], 'dim': f(r['dim'], 0), 'alpha': f(r['alpha'], 0),
            'alphalo': f(r['alphas'][0], 0), 'alphahi': f(r['alphas'][1], 0), 'nalphas': str(r['n_alphas']),
            'mintrain': str(r['min_train']), 'qlo': pct(r['q01']), 'qhi': pct(r['q99']), 'ymean': pct(r['y_mean'], 3),
            'ysd': pct(r['y_sd']), 'bzero': f(r['b0'] * 1e6, 2), 'betanorm': f(r['beta_norm'], 4),
            'nq': f(r['n_quarter'], 0)})
for j in range(4):
    out[f'mu{j + 1}'] = f(r['mu_first'][j], 4)
    out[f'sd{j + 1}'] = f(r['sd_first'][j], 4)
    out[f'beta{j + 1}'] = f(r['beta_first'][j] * 1e3, 3)

for i, st in enumerate(v['stocks'], 1):
    out.update({f'sym{i}': st['symbol'], f'ev{i}': st['event_date'], f'acc{i}': st['accepted_et'].replace('T', ' '),
                f'xnorm{i}': f(st['x_norm'], 2), f'crest{i}': f(st['contrib_rest'], 4),
                f'csum{i}': f(st['contrib_sum'], 4), f'pred{i}': f(st['pred'], 4), f'stored{i}': f(st['pred_stored'], 7),
                f'mine{i}': f(st['pred'], 7), f'predret{i}': pct(st['pred_return']),
                f'rank{i}': f(st['rank'], 0), f'N': f(st['N'], 0), f'quant{i}': f(st['quantile'], 4),
                f'z{i}': f(st['z'], 3), f'sigmazero{i}': pct(st['sigma0'], 3), f's{i}': f(st['s'], 3),
                f'sigma{i}': pct(st['sigma'], 3), f'sigmaann{i}': pct(st['sigma'] * 252 ** 0.5, 1),
                f'wbandfull{i}': pct(st['w_band_full'], 3), f'wblmfull{i}': pct(st['w_blm_full'], 3),
                f'ndays{i}': str(st['n_days'])})
    for j in range(4):
        out[f'x{i}{j + 1}'] = f(st['x_first'][j], 4)
        out[f'xs{i}{j + 1}'] = f(st['xs_first'][j], 3)
        out[f'c{i}{j + 1}'] = f(st['contrib_first'][j], 4)
    for k, (dim, xs, b, c) in enumerate(st['top_dims'], 1):
        out[f'td{i}{k}'] = str(dim)
        out[f'tdx{i}{k}'] = f(xs, 3)
        out[f'tdb{i}{k}'] = f(b * 1e3, 2)
        out[f'tdc{i}{k}'] = f(c, 4)

u = v['universe']
out.update({'nsample': f(u['n_sample'], 0), 'nband': f(u['n_band'], 0), 'ncov': f(u['n_cov'], 0),
            'nrecent': f(u['n_recent'], 0), 'nview': f(u['n_view'], 0), 'viewfrom': u['view_from'],
            'minhist': str(u['min_history'])})

c = v['cov']
out['shrink'] = f(c['shrink'], 3)
for i in range(3):
    for j in range(3):
        out[f'Sa{i + 1}{j + 1}'] = f(c['S3_annual'][i][j], 4)
        out[f'Sd{i + 1}{j + 1}'] = f(c['S3_daily'][i][j] * 1e4, 3)
        out[f'rho{i + 1}{j + 1}'] = f(c['corr'][i][j], 3)

k = v['calib']
out.update({'bpast': f(k['b_past'], 3), 'nhist': f(k['n_hist'], 0), 'limit': pct(k['limit'], 2),
            'TE': pct(k['TE'], 0), 'Llo': pct(k['range'][0], 0), 'Lhi': pct(k['range'][1], 0),
            'histdays': str(k['hist_days']), 'mindays': str(k['min_days']), 'limitdaily': pct(k['limit'] / 252 ** 0.5, 4)})
pr = v['params']
out.update({'IC': f(pr['IC'], 2), 'TAU': f(pr['TAU'], 3), 'DELTA': f(pr['DELTA'], 1), 'viewdays': str(pr['VIEW_DAYS'])})
out.update({'date': v['date'], 'end': v['end'], 'h': str(v['h']), 'sqrth': f(v['h'] ** 0.5, 3)})

b = v['blm3']
for i in range(3):
    n = i + 1
    out.update({f'w{n}': f(b['w3'][i], 3), f'prior{n}': pct(b['prior_daily'][i], 4),
                f'priorann{n}': pct(b['prior_daily'][i] * 252, 1), f'tilt{n}': pct(b['tilt_daily'][i], 4),
                f'tiltann{n}': pct(b['tilt_daily'][i] * 252, 2), f'q{n}': pct(b['q_daily'][i], 4),
                f'om{n}': f(b['omega_diag'][i] * 1e5, 3), f'post{n}': pct(b['post_daily'][i], 4),
                f'postann{n}': pct(b['post_daily'][i] * 252, 1),
                f'gap{n}': pct(b['post_daily'][i] - b['prior_daily'][i], 4),
                f'gapann{n}': pct((b['post_daily'][i] - b['prior_daily'][i]) * 252, 2),
                f'wopt{n}': f(b['w_opt'][i], 4), f'act{n}': pct(b['active'][i], 2), f'wfree{n}': f(b['w_free'][i], 4),
                f'actfree{n}': pct(b['w_free'][i] - b['w3'][i], 2)})
    for j in range(3):
        out[f'M{n}{j + 1}'] = f(b['M'][i][j], 3)
out.update({'tethree': pct(b['te3'], 2), 'tefree': pct(b['te_free'], 2), 'status': b['status'],
            'objband': f(b['obj_band'] * 1e4, 3), 'objopt': f(b['obj_opt'] * 1e4, 3), 'objfree': f(b['obj_free'] * 1e4, 3)})

fu = v['full']
out.update({'teex': pct(fu['te_ex_annual'], 2), 'nheld': f(fu['n_held'], 0), 'nuniv': f(fu['n_universe'], 0)})

pe = v['period']
for i in range(3):
    out[f'g{i + 1}'] = pct(pe['growth'][i], 1)
out.update({'port': pct(pe['port'], 2), 'band': pct(pe['band'], 2), 'active': pct(pe['active'], 2),
            'traded': pct(pe['traded'], 1), 'cost': pct(pe['cost'], 3),
            'net': pct(pe['active'] - pe['cost'], 2)})
ch = v['check']
out.update({'chkday': ch['first_day'], 'chkport': pct(ch['port_replay'], 4), 'chkportst': pct(ch['port_stored'], 4),
            'chkbench': pct(ch['bench_replay'], 4), 'chkbenchst': pct(ch['bench_stored'], 4)})

lines = ['% generated by 作成スクリプト/make_tex.py from values.json; do not edit',
         r'\newcommand{\V}[1]{\csname v@#1\endcsname}']
for key, val in out.items():
    lines.append(r'\expandafter\def\csname v@' + key + r'\endcsname{' + val.replace('_', r'\_') + '}')
(HERE.parent / 'values.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(len(out), 'values')
