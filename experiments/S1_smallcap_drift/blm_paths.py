"""
S1, addendum 1 (preregistration.md): Black-Litterman portfolios of the small caps rebalanced every quarter, half year
and year, valued from 1 at the end of 2015 and at the end of 2023, against the float-weighted band. Reads
results/panel.csv of run.py (the scores and the learned predictions), so it runs after run.py:

    python experiments/S1_smallcap_drift/blm_paths.py [--dry]

Reported, not judged: the paths, final values, yearly returns, realized tracking error and turnover.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from blm.black_litterman import posterior_returns_general  # noqa: E402
from blm.optimizer import optimize_with_tracking_error  # noqa: E402
from events.returns import load_closes  # noqa: E402
from smallcap.build import open_db  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s1 = _load('s1_run', HERE / 'run.py')
x02 = _load('x02_run', REPO / 'experiments' / 'X02_covariance_window' / 'run.py')

STARTS = {'2015': '2015-12-31', '2023': '2023-12-31'}
FREQS = {'quarter': (3, 6, 9, 12), 'half': (6, 12), 'year': (12,)}
MODELS = {'2015': ['chrono_ridge'], '2023': ['chrono_ridge', 'finbert_ft', 'gpt', 'gpt_ridge']}
VIEW_DAYS = 92                 # a release is a view for 92 calendar days after its event date
IC = 0.05                      # as experiments 7 and 9, fixed
TAU, DELTA = 0.025, 2.5
TE = 0.03                      # ex-ante, annual; no calibration (no past ratio for the small caps)
COST = 0.003                   # one way, of the traded amount (small caps)
MIN_HISTORY = 126              # days with a price among the last 252


def blom(values):
    r = stats.rankdata(values, method='average')
    return stats.norm.ppf((r - 0.375) / (len(r) + 0.25))


def schedule(days, start, months):
    """Last trading day of each scheduled month from start, and the last day with prices."""
    s = pd.Series(pd.to_datetime(days), index=days)
    s = s[s.dt.to_period('M') >= pd.Period(start, 'M')]          # the start month's last trading day is the first date
    ends = s.groupby(s.dt.to_period('M')).apply(lambda x: x.index[-1])
    out = [d for d in ends if pd.Timestamp(d).month in months]
    return out, days[-1]


def views(p, col, date, universe):
    lo = (pd.Timestamp(date) - pd.Timedelta(days=VIEW_DAYS)).strftime('%Y-%m-%d')
    v = p[(p.event_date <= date) & (p.event_date > lo) & p.symbol.isin(universe) & p[col].notna()]
    v = v.sort_values(['event_date', 'accepted_at']).drop_duplicates('symbol', keep='last').set_index('symbol')[col]
    return pd.Series(blom(v.values), index=v.index) if len(v) else pd.Series(dtype=float)


def blm_weights(cov, w_mkt, sigma_daily, z, horizon):
    """One absolute view per stock with a score: q = prior + IC * sigma * z / sqrt(horizon) (Grinold's alpha of the
    holding period, in daily units), He-Litterman Omega, then the optimizer with the ex-ante TE limit."""
    if len(z) == 0:
        return w_mkt, 'no views'
    prior = DELTA * cov @ w_mkt
    pick = pd.DataFrame(0.0, index=range(len(z)), columns=cov.index)
    for k, s in enumerate(z.index):
        pick.loc[k, s] = 1.0
    q = pd.Series(prior[z.index].values + IC * sigma_daily[z.index].values * z.values / np.sqrt(horizon))
    omega = TAU * np.diag(np.diag(pick.values @ cov.values @ pick.values.T))
    mu = posterior_returns_general(prior, cov, pick, q, omega, TAU)
    try:
        return optimize_with_tracking_error(mu, cov, DELTA, w_mkt, TE / np.sqrt(252))
    except RuntimeError as e:
        return w_mkt, str(e)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = argparse.ArgumentParser()
    a.add_argument('--dry', action='store_true')
    args = a.parse_args()
    res = s1.OUT / ('dry' if args.dry else '')
    t0 = time.time()
    conn = open_db()
    p = pd.read_csv(res / 'panel.csv', dtype={'event_date': str, 'filing_date': str})
    days = [r[0] for r in conn.execute('SELECT date FROM trading_days ORDER BY date')]
    f = s1.float_known(conn)
    comp = pd.read_sql('SELECT cik, symbol FROM sc_sample', conn)
    closes = load_closes(conn, sorted(comp.symbol), '2014-12-01', s1.LATEST_PRICE)
    rets = closes.pct_change()
    if args.dry:                   # noise in place of every return
        rets = pd.DataFrame(np.random.default_rng(4).normal(0.0004, 0.025, rets.shape), index=rets.index,
                            columns=rets.columns).where(rets.notna())
    index = {d: n for n, d in enumerate(rets.index)}
    cache = {}

    def inputs(date):
        if date not in cache:
            u = comp.assign(m=date)
            u = u.join(s1.attach_float(u, f, 'm'))
            u = u[u.in_band == 1].set_index('symbol')
            i = index[date]
            r252 = rets.iloc[max(0, i - 251):i + 1][u.index.intersection(rets.columns)]
            r252 = r252.loc[:, (r252.notna().sum() >= MIN_HISTORY) & closes.loc[date].reindex(r252.columns).notna()]
            cov = x02.lw(r252.fillna(0.0))
            w = np.exp(u.log_float.reindex(cov.index))
            cache[date] = (cov, w / w.sum(), r252.std())
        return cache[date]

    rows, summary = [], {}
    for start_name, start in STARTS.items():
        for freq, months in FREQS.items():
            dates, last = schedule(list(rets.index), start, months)
            for model in MODELS[start_name] + ['band']:
                value, held, path, stats_rows = 1.0, None, [], []
                for k, d in enumerate(dates):
                    end = dates[k + 1] if k + 1 < len(dates) else last
                    cov, w_mkt, sigma = inputs(d)
                    h = index[end] - index[d]
                    if model == 'band':
                        w, status, n_views = w_mkt, 'band', 0
                    else:
                        z = views(p, model, d, set(cov.index))
                        (w, status), n_views = blm_weights(cov, w_mkt, sigma, z, max(h, 1)), len(z)
                    before = pd.Series(dtype=float) if held is None else held
                    idx = w.index.union(before.index)
                    traded = float((w.reindex(idx).fillna(0) - before.reindex(idx).fillna(0)).abs().sum())
                    value *= 1 - COST * traded
                    period = rets.iloc[index[d] + 1:index[end] + 1][w.index].fillna(0.0)
                    cur, bcur = w.copy(), w_mkt.reindex(w.index).fillna(0.0)
                    act = []
                    for day, r in period.iterrows():
                        port, bench = float((cur * r).sum()), float((bcur * r).sum())
                        act.append(port - bench)
                        value *= 1 + port
                        cur, bcur = cur * (1 + r) / (1 + port), bcur * (1 + r) / (1 + bench)
                        path.append({'date': day, 'value': value})
                    held = cur
                    stats_rows.append({'date': d, 'n_universe': len(cov), 'n_views': n_views, 'status': str(status),
                                       'turnover': traded / 2,
                                       'te_realized': float(np.std(act) * np.sqrt(252)) if len(act) > 1 else np.nan})
                pth = pd.DataFrame(path)
                pth['date'] = pd.to_datetime(pth.date)
                monthly = pth.groupby(pth.date.dt.to_period('M')).value.last()
                for m, v in monthly.items():
                    rows.append({'start': start_name, 'freq': freq, 'model': model, 'month': str(m), 'value': float(v)})
                st = pd.DataFrame(stats_rows)
                years = (pth.date.iloc[-1] - pd.Timestamp(start)).days / 365.25
                summary[f'{start_name}/{freq}/{model}'] = {
                    'final_value': float(value), 'annual_return': float(value ** (1 / years) - 1),
                    'rebalances': int(len(st)), 'mean_universe': float(st.n_universe.mean()),
                    'mean_views': float(st.n_views.mean()), 'turnover_mean': float(st.turnover.mean()),
                    'te_realized_mean': float(st.te_realized.mean()) if model != 'band' else None,
                    'status': st.status.value_counts().to_dict()}
                print(f'{start_name} {freq} {model}: {value:.3f} ({time.time() - t0:.0f}s)', flush=True)
    pd.DataFrame(rows).to_csv(res / 'blm_paths.csv', index=False)
    for key, v in summary.items():
        start_name, freq, model = key.split('/')
        if model != 'band':
            band = summary[f'{start_name}/{freq}/band']
            v['annual_excess_over_band'] = v['annual_return'] - band['annual_return']
    (res / 'blm_paths.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
