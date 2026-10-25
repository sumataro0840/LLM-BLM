"""
S3: does a relative correction of the stock variances by the reading of the earnings release (S2's risk score) make
the ex-ante tracking error of S1's Black-Litterman portfolios more accurate? Paths from 1 at the end of 2015 and 2023,
rebalanced every quarter, half year and year (preregistration.md).

    python experiments/S3_smallcap_risk_blm/run.py --dry    # noise in place of every return (and S1/S2's dry panels)
    python experiments/S3_smallcap_risk_blm/run.py          # once, after preregistration.md and this file are committed

Reuses S1's addendum 1 (experiments/S1_smallcap_drift/blm_paths.py) for everything except the covariance.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from events.returns import load_closes  # noqa: E402
from smallcap.build import open_db  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bp = _load('s1_blm_paths', REPO / 'experiments' / 'S1_smallcap_drift' / 'blm_paths.py')
s1, x02 = bp.s1, bp.x02
S1_RES = REPO / 'experiments' / 'S1_smallcap_drift' / 'results'
S2_RES = REPO / 'experiments' / 'S2_smallcap_risk' / 'results'

COMBOS = {'2015': {'chrono_ridge': ['A', 'B', 'C']},
          '2023': {'chrono_ridge': ['A', 'B', 'C'], 'gpt_ridge': ['A', 'B', 'C_gpt']}}
RISK_SCORE = {'C': 'chrono_risk', 'C_gpt': 'gpt_risk'}
FEATURES = ['log_rv_pre', 'log_rv_long', 'log_float', 'abs_r_0_1', 'abs_sue', 'sue_missing', 'abs_rev', 'rev_missing']
POST_DAYS = 60                 # a stock is "after a release" for 60 trading days after the event date
MIN_TRAIN = 500
S_RANGE = (0.5, 2.0)
BLOCK, N_BOOT = 63, 2000
PRIMARY = ('2015/quarter/chrono_ridge/B', '2015/quarter/chrono_ridge/C')
SECONDARY = {'B_vs_A': ('2015/quarter/chrono_ridge/A', '2015/quarter/chrono_ridge/B'),
             'C_vs_A': ('2015/quarter/chrono_ridge/A', '2015/quarter/chrono_ridge/C'),
             'gpt_C_vs_B': ('2023/quarter/gpt_ridge/B', '2023/quarter/gpt_ridge/C_gpt')}


def panel(dry):
    """S1's release panel (return scores, controls) joined with S2's (volatilities, risk scores)."""
    sub = 'dry' if dry else ''
    a = pd.read_csv(S1_RES / sub / 'panel.csv', dtype={'event_date': str, 'filing_date': str})
    b = pd.read_csv(S2_RES / sub / 'panel.csv', usecols=['accession', 'log_rv_2_60', 'log_rv_pre', 'log_rv_long',
                                                          'abs_r_0_1', 'chrono_risk', 'gpt_risk'])
    p = a.merge(b, on='accession', how='left')
    p['sue_missing'] = p.sue.isna().astype(float)
    p['rev_missing'] = p.revenue_surprise.isna().astype(float)
    p['abs_sue'] = p.sue.abs()
    p['abs_rev'] = p.revenue_surprise.abs()
    p['window_end_60'] = p.window_end_60.fillna('9999-12-31')
    return p


def demean(frame, cols, by):
    return frame[cols] - frame.groupby(by)[cols].transform('mean')


def scale_factors(p, date, days_index, days, cov, score=None):
    """Multipliers s for the stocks after a release on `date` (relative to the other stocks after a release)."""
    k = days_index[date]
    lo = days[max(0, k - POST_DAYS)]
    cur = p[(p.event_date <= date) & (p.event_date > lo) & p.symbol.isin(cov.index) & p.log_rv_pre.notna()]
    cur = cur.sort_values(['event_date', 'accepted_at']).drop_duplicates('symbol', keep='last').set_index('symbol')
    train = p[(p.window_end_60 < date) & p.log_rv_2_60.notna() & p.log_rv_pre.notna()]
    xs = FEATURES + ([score] if score else [])
    if score:
        train = train[train[score].notna()]
        cur = cur[cur[score].notna()]
    if len(train) < MIN_TRAIN or len(cur) < 25:
        return pd.Series(1.0, index=cov.index), {'n_train': int(len(train)), 'n_after': int(len(cur)), 'used': False}
    tr = train[xs + ['log_rv_2_60', 'year_quarter']].copy()
    lo_q, hi_q = tr[xs].quantile(0.01), tr[xs].quantile(0.99)
    tr[xs] = tr[xs].clip(lo_q, hi_q, axis=1).fillna(0.0)
    c = cur[xs].clip(lo_q, hi_q, axis=1).fillna(0.0)
    if score:                      # the score in standard units of its quarter (training) and of the day (current)
        g = tr.groupby('year_quarter')[score]
        tr[score] = ((tr[score] - g.transform('mean')) / g.transform('std')).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        c[score] = (c[score] - c[score].mean()) / c[score].std()
    X = demean(tr, xs, 'year_quarter').to_numpy(float)
    y = (tr.log_rv_2_60 - tr.groupby('year_quarter').log_rv_2_60.transform('mean')).to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = c.to_numpy(float) @ beta
    yhat = yhat - yhat.mean()
    lw = np.log(np.sqrt(np.diag(cov.loc[c.index, c.index].to_numpy())))
    lw = lw - lw.mean()
    s = pd.Series(1.0, index=cov.index)
    s[c.index] = np.clip(np.exp(yhat - lw), *S_RANGE)
    return s, {'n_train': int(len(train)), 'n_after': int(len(cur)), 'used': True, 'scaled': list(c.index),
               'beta_score': float(beta[-1]) if score else None}


def bias(z):
    return float(np.sqrt(np.mean(np.square(z))))


def block_test(z_ref, z_new, seed=0):
    """D = |log b_ref| - |log b_new| and the circular block bootstrap share of D* <= 0 (same days for both)."""
    both = pd.concat([z_ref.rename('r'), z_new.rename('n')], axis=1).dropna()
    r, n = both.r.to_numpy(), both.n.to_numpy()
    d = abs(np.log(bias(r))) - abs(np.log(bias(n)))
    rng = np.random.default_rng(seed)
    N, nb = len(r), int(np.ceil(len(r) / BLOCK))
    stats = []
    for _ in range(N_BOOT):
        idx = (rng.integers(0, N, nb)[:, None] + np.arange(BLOCK)[None, :]).ravel()[:N] % N
        stats.append(abs(np.log(bias(r[idx]))) - abs(np.log(bias(n[idx]))))
    stats = np.array(stats)
    return {'days': int(N), 'b_ref': bias(r), 'b_new': bias(n), 'D': float(d),
            'ci95': [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))],
            'p_one_sided': float(np.mean(stats <= 0))}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = argparse.ArgumentParser()
    a.add_argument('--dry', action='store_true')
    args = a.parse_args()
    res = OUT / ('dry' if args.dry else '')
    if not args.dry and (OUT / 'summary.json').exists():
        raise SystemExit('results/summary.json があります。S3 は実行済みです（事前登録どおり 1 回だけ）')
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    conn = open_db()
    p = panel(args.dry)
    f = s1.float_known(conn)
    comp = pd.read_sql('SELECT cik, symbol FROM sc_sample', conn)
    closes = load_closes(conn, sorted(comp.symbol), '2014-12-01', s1.LATEST_PRICE)
    rets = closes.pct_change()
    if args.dry:
        rets = pd.DataFrame(np.random.default_rng(4).normal(0.0004, 0.025, rets.shape), index=rets.index,
                            columns=rets.columns).where(rets.notna())
    days = list(rets.index)
    index = {d: n for n, d in enumerate(days)}
    cache, s_cache, s_info = {}, {}, {}

    def inputs(date):
        if date not in cache:
            u = comp.assign(m=date)
            u = u.join(s1.attach_float(u, f, 'm'))
            u = u[u.in_band == 1].set_index('symbol')
            i = index[date]
            r252 = rets.iloc[max(0, i - 251):i + 1][u.index.intersection(rets.columns)]
            r252 = r252.loc[:, (r252.notna().sum() >= bp.MIN_HISTORY) & closes.loc[date].reindex(r252.columns).notna()]
            cov = x02.lw(r252.fillna(0.0))
            w = np.exp(u.log_float.reindex(cov.index))
            cache[date] = (cov, w / w.sum(), r252.std())
        return cache[date]

    def factors(date, version, cov):
        """A: none. B: market data. C/C_gpt: market data + the risk score where the score exists, B elsewhere."""
        key = (date, version)
        if key in s_cache:
            return s_cache[key]
        if version == 'A':
            s = pd.Series(1.0, index=cov.index)
        elif version == 'B':
            s, s_info[f'{date}/B'] = scale_factors(p, date, index, days, cov)
        else:
            s = factors(date, 'B', cov).copy()
            sc, info = scale_factors(p, date, index, days, cov, RISK_SCORE[version])
            s_info[f'{date}/{version}'] = info
            if info['used']:
                s[info['scaled']] = sc[info['scaled']]
        s_cache[key] = s
        return s

    rows, summary, zs = [], {}, {}
    for start_name, start in bp.STARTS.items():
        for freq, months in bp.FREQS.items():
            dates, last = bp.schedule(days, start, months)
            runs = [(m, v) for m, vs in COMBOS[start_name].items() for v in vs] + [('band', None)]
            for model, version in runs:
                key = f'{start_name}/{freq}/band' if model == 'band' else f'{start_name}/{freq}/{model}/{version}'
                value, held, path, st_rows, z_all = 1.0, None, [], [], []
                for k, d in enumerate(dates):
                    end = dates[k + 1] if k + 1 < len(dates) else last
                    cov0, w_mkt, sigma0 = inputs(d)
                    h = index[end] - index[d]
                    if model == 'band':
                        w, status, n_views, te_ex, s = w_mkt, 'band', 0, np.nan, None
                    else:
                        s = factors(d, version, cov0)
                        cov = cov0.mul(s, axis=0).mul(s, axis=1)
                        sigma = sigma0.reindex(cov.index) * s
                        z = bp.views(p, model, d, set(cov.index))
                        (w, status), n_views = bp.blm_weights(cov, w_mkt, sigma, z, max(h, 1)), len(z)
                        act_w = w.reindex(cov.index).fillna(0.0) - w_mkt
                        te_ex = float(np.sqrt(act_w.to_numpy() @ cov.to_numpy() @ act_w.to_numpy()))
                    before = pd.Series(dtype=float) if held is None else held
                    idx = w.index.union(before.index)
                    traded = float((w.reindex(idx).fillna(0) - before.reindex(idx).fillna(0)).abs().sum())
                    value *= 1 - bp.COST * traded
                    period = rets.iloc[index[d] + 1:index[end] + 1][w.index].fillna(0.0)
                    cur, bcur = w.copy(), w_mkt.reindex(w.index).fillna(0.0)
                    act = []
                    for day, r in period.iterrows():
                        port, bench = float((cur * r).sum()), float((bcur * r).sum())
                        act.append(port - bench)
                        if model != 'band' and te_ex > 0:
                            z_all.append({'date': day, 'z': (port - bench) / te_ex})
                        value *= 1 + port
                        cur, bcur = cur * (1 + r) / (1 + port), bcur * (1 + r) / (1 + bench)
                        path.append({'date': day, 'value': value})
                    held = cur
                    st_rows.append({'n_universe': len(cov0), 'n_views': n_views, 'status': str(status),
                                    'turnover': traded / 2, 'te_ex': te_ex * np.sqrt(252),
                                    'n_scaled': int((s != 1.0).sum()) if s is not None else 0,
                                    'te_realized': float(np.std(act) * np.sqrt(252)) if len(act) > 1 else np.nan})
                pth = pd.DataFrame(path)
                pth['date'] = pd.to_datetime(pth.date)
                for m, v in pth.groupby(pth.date.dt.to_period('M')).value.last().items():
                    rows.append({'start': start_name, 'freq': freq, 'model': model, 'version': version or '',
                                 'month': str(m), 'value': float(v)})
                st = pd.DataFrame(st_rows)
                years = (pth.date.iloc[-1] - pd.Timestamp(start)).days / 365.25
                out = {'final_value': float(value), 'annual_return': float(value ** (1 / years) - 1),
                       'rebalances': int(len(st)), 'mean_views': float(st.n_views.mean()),
                       'mean_scaled': float(st.n_scaled.mean()), 'turnover_mean': float(st.turnover.mean()),
                       'status': st.status.value_counts().to_dict()}
                if model != 'band':
                    zser = pd.DataFrame(z_all).set_index('date').z
                    zs[key] = zser
                    out.update({'te_ex_mean': float(st.te_ex.mean()), 'te_realized_mean': float(st.te_realized.mean()),
                                'bias_b': bias(zser.to_numpy()), 'days': int(len(zser))})
                summary[key] = out
                print(f'{key}: {value:.3f} ({time.time() - t0:.0f}s)', flush=True)
    for key, v in summary.items():
        parts = key.split('/')
        if parts[2] != 'band':
            v['annual_excess_over_band'] = v['annual_return'] - summary[f'{parts[0]}/{parts[1]}/band']['annual_return']
    tests = {'primary': block_test(zs[PRIMARY[0]], zs[PRIMARY[1]])}
    tests['primary']['verdict'] = 'supported' if tests['primary']['D'] > 0 and tests['primary']['p_one_sided'] < 0.05 \
        else 'not supported'
    for name, (ref, new) in SECONDARY.items():
        tests[name] = block_test(zs[ref], zs[new])
    used = pd.DataFrame([{'key': k, **v} for k, v in s_info.items()])
    result = {'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
              'dry': args.dry, 'tests': tests, 'runs': summary,
              'scaling': {'dates': int(len(used)), 'used_share': used.groupby(used.key.str.split('/').str[1]).used.mean().to_dict()
                          if len(used) else {},
                          'beta_score_mean': used.dropna(subset=['beta_score']).groupby(used.key.str.split('/').str[1])
                          .beta_score.mean().to_dict() if 'beta_score' in used else {}},
              'seconds': round(time.time() - t0)}
    (res / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    pd.DataFrame(rows).to_csv(res / 'paths.csv', index=False)
    pd.concat({k: v for k, v in zs.items()}, names=['key', 'date']).rename('z').reset_index().to_csv(
        res / 'z.csv', index=False, float_format='%.5f')
    print(json.dumps(tests, ensure_ascii=False, indent=1))
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
