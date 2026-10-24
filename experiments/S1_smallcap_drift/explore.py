"""
S1 exploration (NOT preregistered; written after the preregistered run, reported separately as 探索).

Why: the primary score predicted the 2-60 day drift but hardly the 2-21 day drift, so the effect sits in days 22-60.
That pattern could be a known factor picked up by the text score rather than a slow reaction to the release.
Checks (no permutation tests; t-values only, clustered by event date as in run.py):
  1. days 22-60 alone (a window not in the preregistration)
  2. the primary regression plus past-year momentum (abnormal return over days -251..-22 against IWM)
  3. the same for the gpt-4o-mini scores in their period
  4. halves by public float (smaller / larger than the median of the evaluated releases)
  5. mean abnormal returns by quintile of the score (descriptive)

    python experiments/S1_smallcap_drift/explore.py      # reads results/panel.csv, writes results/explore.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('s1', HERE / 'run.py')
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)

EXTRA = {'drift_22_60': (22, 60), 'mom': (-251, -22)}


def winsor(x):
    lo, hi = x.quantile([0.01, 0.99])
    return x.clip(lo, hi)


def fit_with(sample, focal, y, extra_controls=()):
    base = list(s1.CONTROLS)
    s1.CONTROLS = base + list(extra_controls)
    try:
        return s1.fit(sample, focal, y)
    finally:
        s1.CONTROLS = base


def quintiles(sample, focal, ys):
    d = sample.dropna(subset=[focal]).copy()
    d['q'] = pd.qcut(d[focal].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    out = {}
    for y in ys:
        v = d.dropna(subset=[y])
        v = v.assign(w=winsor(v[y]))
        out[y] = {int(q): float(g.w.mean() * 1e4) for q, g in v.groupby('q', observed=True)}
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = pd.read_csv(HERE / 'results' / 'panel.csv', dtype={'cik': str})
    conn = s1.open_db()
    days = [r[0] for r in conn.execute('SELECT date FROM trading_days ORDER BY date')]
    closes = s1.load_closes(conn, sorted(set(p.symbol)) + [s1.BENCH, '^GSPC'], '2014-01-01', s1.LATEST_PRICE)
    r = s1.reactions(p[['accession', 'symbol', 'event_date']], closes, days, {s: s1.BENCH for s in p.symbol}, EXTRA)
    p = p.merge(r[['accession', 'abn_drift_22_60', 'abn_mom']], on='accession', how='left')
    p.loc[p.window_end_60.isna(), 'abn_drift_22_60'] = np.nan
    p['abn_mom'] = winsor(p.abn_mom)
    ev = p[(p.filing_date >= s1.EVAL_START) & (p.in_band == 1)]
    ev_gpt = ev[(ev.filing_date >= s1.GPT_START) & ev[['gpt', 'gpt_ridge']].notna().all(axis=1)]
    out = {'note': 'exploration after the preregistered run; not preregistered',
           'mom_missing_share': float(ev.abn_mom.isna().mean())}
    out['days_22_60'] = {m: fit_with(ev, m, 'abn_drift_22_60') for m in ('chrono_ridge', 'finbert_ft', 'dictionary')}
    out['days_22_60_gpt'] = {m: fit_with(ev_gpt, m, 'abn_drift_22_60') for m in ('gpt', 'gpt_ridge', 'chrono_ridge')}
    out['plus_momentum'] = {m: fit_with(ev, m, 'abn_drift_2_60', ['abn_mom']) for m in ('chrono_ridge', 'finbert_ft')}
    out['plus_momentum_gpt'] = {m: fit_with(ev_gpt, m, 'abn_drift_2_60', ['abn_mom'])
                                for m in ('gpt', 'gpt_ridge', 'chrono_ridge')}
    med = ev.log_float.median()
    out['by_size'] = {half: fit_with(ev[(ev.log_float <= med) if half == 'smaller' else (ev.log_float > med)],
                                     'chrono_ridge', 'abn_drift_2_60') for half in ('smaller', 'larger')}
    out['quintiles_chrono'] = quintiles(ev, 'chrono_ridge', ['abn_r_0_1', 'abn_drift_2_21', 'abn_drift_22_60', 'abn_drift_2_60'])
    out['quintiles_gpt'] = quintiles(ev_gpt, 'gpt', ['abn_r_0_1', 'abn_drift_2_21', 'abn_drift_22_60', 'abn_drift_2_60'])
    out['corr_score_mom'] = {m: float(ev[[m, 'abn_mom']].corr(method='spearman').iloc[0, 1]) for m in ('chrono_ridge', 'finbert_ft')}
    out['corr_score_mom_gpt'] = {m: float(ev_gpt[[m, 'abn_mom']].corr(method='spearman').iloc[0, 1]) for m in ('gpt', 'gpt_ridge')}
    (HERE / 'results' / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
