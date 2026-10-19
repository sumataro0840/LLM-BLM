"""
X05 (exploratory, after E7 and E9's results were seen): how much does the size of the alpha matter?

E7 and E9 built q = Pi + IC * sigma_daily * z with the daily covariance and held for a month. Grinold's rule uses the
volatility over the forecast horizon: alpha_month = IC * sigma_month * z, i.e. IC * sigma_daily * z / sqrt(21) in
daily units. So the alpha used was sqrt(21) = 4.6 times the horizon-consistent one. Here the same monthly BLM is
rebuilt with the alpha multiplied by k = 0.1, 1/sqrt(21), 0.5, 1 (as used) and 2, for every text score:

    window A (2024-01 .. 2026-07, E7's dates and TE limits): gpt-4o-mini, ChronoGPT ridge (E5), dictionary,
                                                              FinBERT (zero-shot), TF-IDF ridge, fine-tuned FinBERT (E4)
    window L (2016-03 .. 2026-07, E9's dates and TE limits):  ChronoGPT ridge, dictionary, FinBERT, TF-IDF ridge

Everything else is E7's code (views, Omega, covariance, costs). Not pre-registered: report every cell, choose nothing.

    python experiments/X05_alpha_scale/run.py         # k = 0.1 .. 2
    python experiments/X05_alpha_scale/run.py small   # k = 0.001 .. 0.03 (added after the first run)
"""
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from blm.black_litterman import posterior_returns_general
from blm.optimizer import optimize_with_tracking_error
from events import config
from experiments.common import data as D
from marketdb.db import events as db
from marketdb.db.connection import connect


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e7 = _load('e7_run', 'experiments/E7_monthly_blm/run.py')
e9 = _load('e9_run', 'experiments/E9_monthly_blm_long/run.py')
x02 = e7.x02

OUT = Path(__file__).resolve().parent / 'results'
SCALES = [0.1, 1 / np.sqrt(21), 0.5, 1.0, 2.0]
if sys.argv[1:] == ['small']:              # second run: where does the TE limit stop binding?
    SCALES = [0.001, 0.003, 0.01, 0.03]
    OUT = OUT / 'small'
SIGNALS_A = ['gpt4o_mini', 'chrono_ridge', 'dictionary', 'finbert', 'tfidf', 'finbert_ft']
SIGNALS_L = ['chrono_ridge', 'dictionary', 'finbert', 'tfidf']


def signal_frames():
    """For each score: one row per release with symbol, event_date, accepted_at and the value (as llm_composite,
    the column name E7's view selection reads)."""
    p = pd.read_csv(e7.PANEL, usecols=['accession', 'symbol', 'event_date', 'accepted_at', 'filing_date',
                                       'llm_composite', 'dictionary_net_tone', 'finbert_net_tone', 'tfidf_pred'])
    base = p[['accession', 'symbol', 'event_date', 'accepted_at']]
    out = {'gpt4o_mini': p.llm_composite, 'dictionary': p.dictionary_net_tone, 'finbert': p.finbert_net_tone,
           'tfidf': p.tfidf_pred}
    out['chrono_ridge'] = e9.views_frame().set_index('accession').llm_composite.reindex(p.accession).to_numpy()
    conn = connect(config.DB_PATH)
    ft_run = json.loads((REPO / 'experiments' / 'E4_finbert_finetune' / 'results' / 'summary.json')
                        .read_text(encoding='utf-8'))['score_run']
    ft = db.load_scores(conn, ft_run, ['predicted_reaction'])['predicted_reaction']
    out['finbert_ft'] = p.accession.map(ft).to_numpy()
    frames = {}
    for name, v in out.items():
        f = base.assign(llm_composite=np.asarray(v, dtype=float))
        frames[name] = f[f.llm_composite.notna()]
    return frames


def weights(cov, w_mkt, sigma, z, limit, k):
    """E7's blm_weights with the alpha multiplied by k; also returns the ex-ante TE (annual)."""
    prior = e7.DELTA * cov @ w_mkt
    names = z.index
    pick = pd.DataFrame(np.eye(len(cov))[[cov.index.get_loc(s) for s in names]], columns=cov.index)
    q = pd.Series(prior[names].values + k * e7.IC * sigma[names].values * z.values)
    omega = e7.TAU * np.diag(np.diag(pick.values @ cov.values @ pick.values.T))
    mu = posterior_returns_general(prior, cov, pick, q, omega, e7.TAU)
    w, status = optimize_with_tracking_error(mu, cov, e7.DELTA, w_mkt, limit / np.sqrt(252))
    act = (w - w_mkt).values
    return w, status, float(np.sqrt(252 * act @ cov.values @ act))


def one_date(args):
    date, next_end, frames, jobs = args          # jobs: [(window, signal, limit)]
    conn = D.connect()
    returns, _, future, w_mkt = x02.inputs(conn, date, next_end)
    r252 = returns.iloc[-252:]
    cov = x02.lw(r252)
    sigma = r252.std()
    out = {'month': (1 + future).prod() - 1, 'w_mkt': w_mkt, 'daily': future}
    for window, signal, limit in jobs:
        scores = e7.latest_scores(frames[signal], date, set(cov.index))
        z = pd.Series(e7.blom(scores.values), index=scores.index)
        for k in SCALES:
            w, status, te = weights(cov, w_mkt, sigma, z, limit, k)
            out[(window, signal, k)] = (w, status, te, limit, len(z))
    return date, out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(exist_ok=True)
    frames = signal_frames()
    lim_a = e7.te_limits()
    e9m = pd.read_csv(REPO / 'experiments' / 'E9_monthly_blm_long' / 'results' / 'corrected_dates' / 'by_month.csv')
    lim_l = e9m.set_index('date').limit
    dates = e9.decision_dates(D.connect())
    nxt = dict(zip(dates[:-1], dates[1:]))
    tasks = []
    for d in lim_l.index:
        jobs = [('L', s, float(lim_l[d])) for s in SIGNALS_L]
        if e7.FIRST <= d[:7] <= e7.LAST:
            jobs += [('A', s, float(lim_a.loc[d])) for s in SIGNALS_A]
        tasks.append((d, nxt[d], frames, jobs))
    print(f'{len(tasks)} dates, {sum(len(t[3]) for t in tasks) * len(SCALES)} optimizations', flush=True)
    with ProcessPoolExecutor(5) as pool:
        results = list(pool.map(one_date, tasks))

    rows, monthly = [], []
    for window, signals in (('A', SIGNALS_A), ('L', SIGNALS_L)):
        for signal in signals:
            for k in SCALES:
                key = (window, signal, k)
                res = [(d, {'x': x[key][:2], 'month': x['month'], 'w_mkt': x['w_mkt'], 'daily': x['daily']})
                       for d, x in results if key in x]
                bt = e7.backtest(res, 'x')
                te_pred = np.array([x[key][2] for _, x in results if key in x])
                limit = np.array([x[key][3] for _, x in results if key in x])
                s = e7.summarize(bt)
                rows.append({'window': window, 'signal': signal, 'k': k, 'months': len(bt),
                             'active_net_monthly': s['mean_active_net_monthly'], 'nw_t': s['nw_t'],
                             'active_gross_monthly': s['mean_active_gross_monthly'],
                             'ir_net': s['information_ratio_net'], 'turnover': s['turnover_mean'],
                             'te_realized': s['realized_te_mean'], 'te_ex_ante': float(te_pred.mean()),
                             'te_limit': float(limit.mean()), 'share_binding': float((te_pred >= 0.99 * limit).mean()),
                             'views': float(np.mean([x[key][4] for _, x in results if key in x]))})
                monthly.append(bt[['active_net', 'active_gross']].assign(window=window, signal=signal, k=k))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'summary.csv', index=False)
    pd.concat(monthly).to_csv(OUT / 'by_month.csv')
    print(table.round(5).to_string())


if __name__ == '__main__':
    main()
