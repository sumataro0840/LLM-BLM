"""
S4: calibrating the ex-ante TE limit by the portfolio's own past bias (3% / b of the last 756 days) on S3's
small-cap Black-Litterman portfolios. Does the realized TE come closer to the promised 3%? (preregistration.md)

    python experiments/S4_te_calibration/run.py --dry    # noise in place of every return (and S1/S2's dry panels)
    python experiments/S4_te_calibration/run.py          # once, after preregistration.md and this file are committed

Reuses S3 (experiments/S3_smallcap_risk_blm/run.py) for the panel and the covariance correction B, and S1's addendum 1
for the rest of the portfolio.
"""
import argparse
import importlib.util
import json
import subprocess
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from blm.black_litterman import posterior_returns_general  # noqa: E402
from blm.optimizer import optimize_with_tracking_error  # noqa: E402
from events.returns import load_closes  # noqa: E402
from smallcap.build import open_db  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get('SMALLCAP_RESULTS', 'results')  # results_rep: replication
spec = importlib.util.spec_from_file_location('s3_run', REPO / 'experiments' / 'S3_smallcap_risk_blm' / 'run.py')
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)
bp, s1, x02 = s3.bp, s3.s1, s3.x02

COMBOS = {'2015': {'chrono_ridge': ['A', 'A_cal', 'B', 'B_cal']},
          '2023': {'chrono_ridge': ['A', 'A_cal', 'B', 'B_cal'], 'gpt_ridge': ['B', 'B_cal']}}
TE = bp.TE                     # the promise, 3% a year
HIST_DAYS, MIN_DAYS = 756, 126
L_RANGE = (0.01, 0.06)
EVAL = {'2015': '2017-01-01', '2023': '2024-07-01'}
PRIMARY = ('2015/quarter/chrono_ridge/B', '2015/quarter/chrono_ridge/B_cal')
SECONDARY = {'A_vs_A_cal': ('2015/quarter/chrono_ridge/A', '2015/quarter/chrono_ridge/A_cal'),
             'gpt_B_vs_B_cal': ('2023/quarter/gpt_ridge/B', '2023/quarter/gpt_ridge/B_cal'),
             'A_cal_vs_B_cal': ('2015/quarter/chrono_ridge/A_cal', '2015/quarter/chrono_ridge/B_cal')}


def blm_weights(cov, w_mkt, sigma_daily, z, horizon, te_annual):
    """S1's addendum 1 (blm_paths.blm_weights) with the TE limit as an argument."""
    if len(z) == 0:
        return w_mkt, 'no views'
    prior = bp.DELTA * cov @ w_mkt
    pick = pd.DataFrame(0.0, index=range(len(z)), columns=cov.index)
    for k, s in enumerate(z.index):
        pick.loc[k, s] = 1.0
    q = pd.Series(prior[z.index].values + bp.IC * sigma_daily[z.index].values * z.values / np.sqrt(horizon))
    omega = bp.TAU * np.diag(np.diag(pick.values @ cov.values @ pick.values.T))
    mu = posterior_returns_general(prior, cov, pick, q, omega, bp.TAU)
    try:
        return optimize_with_tracking_error(mu, cov, bp.DELTA, w_mkt, te_annual / np.sqrt(252))
    except RuntimeError as e:
        return w_mkt, str(e)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = argparse.ArgumentParser()
    a.add_argument('--dry', action='store_true')
    args = a.parse_args()
    res = OUT / ('dry' if args.dry else '')
    if not args.dry and (OUT / 'summary.json').exists():
        raise SystemExit('results/summary.json があります。S4 は実行済みです（事前登録どおり 1 回だけ）')
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    conn = open_db()
    p = s3.panel(args.dry)
    f = s1.float_known(conn)
    comp = pd.read_sql('SELECT cik, symbol FROM sc_sample', conn)
    closes = load_closes(conn, sorted(comp.symbol), '2014-12-01', s1.LATEST_PRICE)
    rets = closes.pct_change()
    if args.dry:
        rets = pd.DataFrame(np.random.default_rng(4).normal(0.0004, 0.025, rets.shape), index=rets.index,
                            columns=rets.columns).where(rets.notna())
    days = list(rets.index)
    index = {d: n for n, d in enumerate(days)}
    cache, s_cache = {}, {}

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

    def factors(date, cov):
        if date not in s_cache:
            s_cache[date] = s3.scale_factors(p, date, index, days, cov)[0]
        return s_cache[date]

    rows, summary, us, limits = [], {}, {}, []
    for start_name, start in bp.STARTS.items():
        for freq, months in bp.FREQS.items():
            dates, last = bp.schedule(days, start, months)
            runs = [(m, v) for m, vs in COMBOS[start_name].items() for v in vs] + [('band', None)]
            for model, version in runs:
                key = f'{start_name}/{freq}/band' if model == 'band' else f'{start_name}/{freq}/{model}/{version}'
                value, held, path, st_rows, z_hist, u_rows = 1.0, None, [], [], [], []
                for k, d in enumerate(dates):
                    end = dates[k + 1] if k + 1 < len(dates) else last
                    cov0, w_mkt, sigma0 = inputs(d)
                    h = index[end] - index[d]
                    limit = TE
                    if model == 'band':
                        w, status, n_views, te_ex = w_mkt, 'band', 0, np.nan
                    else:
                        s = factors(d, cov0) if version.startswith('B') else pd.Series(1.0, index=cov0.index)
                        cov = cov0.mul(s, axis=0).mul(s, axis=1)
                        sigma = sigma0.reindex(cov.index) * s
                        if version.endswith('_cal') and len(z_hist) >= MIN_DAYS:
                            b_past = float(np.sqrt(np.mean(np.square(z_hist[-HIST_DAYS:]))))
                            limit = float(np.clip(TE / b_past, *L_RANGE))
                        z = bp.views(p, model, d, set(cov.index))
                        (w, status), n_views = blm_weights(cov, w_mkt, sigma, z, max(h, 1), limit), len(z)
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
                        if model != 'band':
                            if te_ex > 0:
                                z_hist.append((port - bench) / te_ex)
                            u_rows.append({'date': day, 'u': (port - bench) / (TE / np.sqrt(252))})
                        value *= 1 + port
                        cur, bcur = cur * (1 + r) / (1 + port), bcur * (1 + r) / (1 + bench)
                        path.append({'date': day, 'value': value, 'active': port - bench})
                    held = cur
                    st_rows.append({'date': d, 'n_views': n_views, 'status': str(status), 'limit': limit,
                                    'turnover': traded / 2, 'te_ex': te_ex * np.sqrt(252)})
                pth = pd.DataFrame(path)
                pth['date'] = pd.to_datetime(pth.date)
                for m, v in pth.groupby(pth.date.dt.to_period('M')).value.last().items():
                    rows.append({'start': start_name, 'freq': freq, 'model': model, 'version': version or '',
                                 'month': str(m), 'value': float(v)})
                st = pd.DataFrame(st_rows)
                limits += [{'key': key, **r} for r in st_rows]
                years = (pth.date.iloc[-1] - pd.Timestamp(start)).days / 365.25
                out = {'final_value': float(value), 'annual_return': float(value ** (1 / years) - 1),
                       'rebalances': int(len(st)), 'turnover_mean': float(st.turnover.mean()),
                       'status': st.status.value_counts().to_dict()}
                if model != 'band':
                    ev = pth[pth.date >= EVAL[start_name]]
                    useries = pd.DataFrame(u_rows).set_index('date').u
                    us[key] = useries
                    out.update({'limit_mean': float(st.limit.mean()), 'limit_min': float(st.limit.min()),
                                'limit_max': float(st.limit.max()), 'te_ex_mean': float(np.nanmean(st.te_ex)),
                                'te_realized_all': float(pth.active.std() * np.sqrt(252)),
                                'te_realized_eval': float(ev.active.std() * np.sqrt(252)),
                                'g_eval': s3.bias(useries[useries.index >= EVAL[start_name]].to_numpy()),
                                'active_annual_eval': float(ev.active.mean() * 252)})
                    out['ir_eval'] = out['active_annual_eval'] / out['te_realized_eval']
                summary[key] = out
                print(f'{key}: {value:.3f} ({time.time() - t0:.0f}s)', flush=True)
    for key, v in summary.items():
        parts = key.split('/')
        if parts[2] != 'band':
            v['annual_excess_over_band'] = v['annual_return'] - summary[f'{parts[0]}/{parts[1]}/band']['annual_return']

    def test(ref, new):
        start = ref.split('/')[0]
        return s3.block_test(us[ref][us[ref].index >= EVAL[start]], us[new][us[new].index >= EVAL[start]])

    tests = {'primary': test(*PRIMARY)}
    tests['primary']['verdict'] = 'supported' if tests['primary']['D'] > 0 and tests['primary']['p_one_sided'] < 0.05 \
        else 'not supported'
    for name, (ref, new) in SECONDARY.items():
        tests[name] = test(ref, new)
    result = {'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
              'dry': args.dry, 'tests': tests, 'runs': summary, 'seconds': round(time.time() - t0)}
    (res / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    pd.DataFrame(rows).to_csv(res / 'paths.csv', index=False)
    pd.DataFrame(limits).to_csv(res / 'limits.csv', index=False)
    pd.concat(us, names=['key', 'date']).rename('u').reset_index().to_csv(res / 'u.csv', index=False, float_format='%.5f')
    print(json.dumps(tests, ensure_ascii=False, indent=1))
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
