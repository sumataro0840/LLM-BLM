"""
R1 exploration (NOT preregistered; written on 2026-10-25 after R1's results were seen, reported separately as 探索).

Why: in the new sample the calibrated BLM's realized TE was about 10% (the promise is 3%), and one day carried most of
it: on 2020-11-20 the adjusted close of CHRD (Oasis Petroleum, which left Chapter 11 on 2020-11-19 with new shares)
went from 0.07 to 18.8, a return of about +25,700% that no holder of the old shares earned (yfinance joined the old
and the new shares' prices). The original sample has the same kind of jumps (AMPY on 2016-10-24, +16,000%), and both
have days that jump and fall back (NFE, BURU). Checks:
  1. extreme daily returns of the sample stocks (adjusted closes, 2015-): counts and the largest
  2. the bias b of S4's portfolios with and without their largest days (S4's u.csv)
  3. S4's 2015/quarter portfolios (A, A_cal, B, B_cal and the band) recomputed with every daily return beyond a
     threshold treated as missing (which S4 counts as 0, as for a missing price), for three thresholds fixed here before
     any of them was run: (+300%, -80%) the main one, (+100%, -50%) and (+1000%, -90%). The run without a threshold
     must reproduce S4. On each: R-BLM's test (R1's mean_test), g and the realized TE.
     A rebalance whose optimization fails (cvxpy's SolverError, which S4 never met) holds the band, as S4 does when
     its optimizer reports a failure; the failures are counted (solver_failures). Added after the first run of the
     original sample stopped on such a failure at the (+100%, -50%) threshold.

    python experiments/R1_replication/explore.py orig    # one sample per process (sets SMALLCAP_DB / _RESULTS)
    python experiments/R1_replication/explore.py rep
Writes results/explore_<sample>.json.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else 'rep'
if SAMPLE == 'rep':                      # before the imports: smallcap.build and S1-S4 read these when loaded
    os.environ['SMALLCAP_DB'] = str(REPO / 'data' / 'smallcap_rep.sqlite')
    os.environ['SMALLCAP_CACHE'] = str(REPO / 'data' / 'cache' / 'smallcap_rep')
    os.environ['SMALLCAP_RESULTS'] = 'results_rep'
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from events.returns import load_closes  # noqa: E402
from smallcap.build import open_db  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s4 = _load('s4_run', REPO / 'experiments' / 'S4_te_calibration' / 'run.py')
r1 = _load('r1_run', HERE / 'run.py')
s3, bp, s1, x02 = s4.s3, s4.bp, s4.s1, s4.x02
RES = 'results_rep' if SAMPLE == 'rep' else 'results'
VERSIONS = ['A', 'A_cal', 'B', 'B_cal']
THRESHOLDS = {'none': None, 'main_300_80': (3.0, -0.8), 'loose_100_50': (1.0, -0.5), 'tight_1000_90': (10.0, -0.9)}
START, FREQ, MODEL = '2015', 'quarter', 'chrono_ridge'


def extreme_days(rets):
    st = rets.stack()
    big = st[(st > 3) | (st < -0.8)]
    return {'stock_days': int(len(st)), 'above_300pct': int((st > 3).sum()), 'above_100pct': int((st > 1).sum()),
            'below_-80pct': int((st < -0.8).sum()), 'below_-50pct': int((st < -0.5).sum()),
            'largest': [{'date': d, 'symbol': s, 'return': float(v)} for (d, s), v in
                        big.abs().sort_values(ascending=False).head(15).items() for v in [st[(d, s)]]]}


def largest_days():
    u = pd.read_csv(REPO / 'experiments' / 'S4_te_calibration' / RES / 'u.csv')
    out = {}
    for v in VERSIONS:
        s = u[(u.key == f'{START}/{FREQ}/{MODEL}/{v}') & (u.date >= s4.EVAL[START])].set_index('date').u
        sq = (s ** 2).sort_values(ascending=False)
        out[v] = {'g': float(np.sqrt(sq.mean())), 'top5_share_of_squares': float(sq.iloc[:5].sum() / sq.sum()),
                  'g_without_top5_days': float(np.sqrt(sq.iloc[5:].mean())),
                  'top_days': [{'date': d, 'u': float(s[d]), 'active_return': float(s[d] * 0.03 / np.sqrt(252))}
                               for d in sq.index[:5]]}
    return out


def run_portfolios(p, f, comp, closes, rets):
    """S4's loop (experiments/S4_te_calibration/run.py, main) for the 2015/quarter chrono_ridge versions and the band."""
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

    dates, last = bp.schedule(days, bp.STARTS[START], bp.FREQS[FREQ])
    results, failed = {}, {}
    for version in VERSIONS + ['band']:
        value, held, z_hist, u_rows, path = 1.0, None, [], [], []
        for k, d in enumerate(dates):
            end = dates[k + 1] if k + 1 < len(dates) else last
            cov0, w_mkt, sigma0 = inputs(d)
            h = index[end] - index[d]
            limit = s4.TE
            if version == 'band':
                w, te_ex = w_mkt, np.nan
            else:
                s = factors(d, cov0) if version.startswith('B') else pd.Series(1.0, index=cov0.index)
                cov = cov0.mul(s, axis=0).mul(s, axis=1)
                sigma = sigma0.reindex(cov.index) * s
                if version.endswith('_cal') and len(z_hist) >= s4.MIN_DAYS:
                    b_past = float(np.sqrt(np.mean(np.square(z_hist[-s4.HIST_DAYS:]))))
                    limit = float(np.clip(s4.TE / b_past, *s4.L_RANGE))
                z = bp.views(p, MODEL, d, set(cov.index))
                try:
                    w, _ = s4.blm_weights(cov, w_mkt, sigma, z, max(h, 1), limit)
                except Exception:        # cvxpy's SolverError (S4 only catches RuntimeError): the band, as S4 does
                    w = w_mkt
                    failed[version] = failed.get(version, 0) + 1
                act_w = w.reindex(cov.index).fillna(0.0) - w_mkt
                te_ex = float(np.sqrt(act_w.to_numpy() @ cov.to_numpy() @ act_w.to_numpy()))
            before = pd.Series(dtype=float) if held is None else held
            idx = w.index.union(before.index)
            traded = float((w.reindex(idx).fillna(0) - before.reindex(idx).fillna(0)).abs().sum())
            value *= 1 - bp.COST * traded
            period = rets.iloc[index[d] + 1:index[end] + 1][w.index].fillna(0.0)
            cur, bcur = w.copy(), w_mkt.reindex(w.index).fillna(0.0)
            for day, r in period.iterrows():
                port, bench = float((cur * r).sum()), float((bcur * r).sum())
                if version != 'band':
                    if te_ex > 0:
                        z_hist.append((port - bench) / te_ex)
                    u_rows.append({'date': day, 'u': (port - bench) / (s4.TE / np.sqrt(252))})
                value *= 1 + port
                cur, bcur = cur * (1 + r) / (1 + port), bcur * (1 + r) / (1 + bench)
                path.append(value)
            held = cur
        out = {'final_value': float(value)}
        if version != 'band':
            us = pd.DataFrame(u_rows).set_index('date').u
            ev = us[us.index >= s4.EVAL[START]]
            out.update({'g_eval': s3.bias(ev.to_numpy()), 'R_BLM_test': r1.mean_test(ev.to_numpy() * r1.TE_DAILY)})
        results[version] = out
    results['solver_failures'] = failed
    return results


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()
    conn = open_db()
    p = s3.panel(False)
    f = s1.float_known(conn)
    comp = pd.read_sql('SELECT cik, symbol FROM sc_sample', conn)
    closes = load_closes(conn, sorted(comp.symbol), '2014-12-01', s1.LATEST_PRICE)
    raw = closes.pct_change()
    out = {'note': 'exploration after R1 was run; not preregistered', 'sample': SAMPLE,
           'extreme_days': extreme_days(raw[raw.index >= '2015-01-01']), 'largest_days_of_S4': largest_days(),
           'thresholds': {k: v for k, v in THRESHOLDS.items()}, 'runs': {}}
    s4_summary = json.loads((REPO / 'experiments' / 'S4_te_calibration' / RES / 'summary.json').read_text(encoding='utf-8'))
    for name, th in THRESHOLDS.items():
        rets = raw if th is None else raw.where((raw <= th[0]) & (raw >= th[1]))
        n_removed = int((raw.notna() & rets.isna()).sum().sum())
        res = run_portfolios(p, f, comp, closes, rets)
        res['n_returns_removed'] = n_removed
        out['runs'][name] = res
        (HERE / 'results' / f'explore_{SAMPLE}_partial.json').write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                                                    default=float), encoding='utf-8')
        print(name, n_removed, {v: (round(r['final_value'], 3), round(r.get('g_eval', float('nan')), 2),
                                    round(r.get('R_BLM_test', {}).get('annual_excess', float('nan')), 4))
                                for v, r in res.items() if v in VERSIONS + ['band']},
              'solver failures', res['solver_failures'], f'({time.time() - t0:.0f}s)', flush=True)
    # the run without a threshold must reproduce S4
    check = {}
    for v in VERSIONS + ['band']:
        key = f'{START}/{FREQ}/band' if v == 'band' else f'{START}/{FREQ}/{MODEL}/{v}'
        ref = s4_summary['runs'][key]
        got = out['runs']['none'][v]
        check[v] = {'final_value_S4': ref['final_value'], 'final_value_here': got['final_value'],
                    'g_S4': ref.get('g_eval'), 'g_here': got.get('g_eval')}
    out['reproduces_S4'] = check
    out['seconds'] = round(time.time() - t0)
    (HERE / 'results' / f'explore_{SAMPLE}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                                                             encoding='utf-8')
    print(json.dumps(check, indent=1))
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
