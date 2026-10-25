"""
S1: in small caps, does the reading of earnings releases predict the drift after the release? FinBERT, ChronoGPT and
gpt-4o-mini, each without and with learning, compared on the same releases (preregistration.md).

    python experiments/S1_smallcap_drift/run.py --dry    # the whole pipeline with noise in place of every return
    python experiments/S1_smallcap_drift/run.py          # once, after preregistration.md and this file are committed

Data: data/smallcap.sqlite (設計_小型株.md) and the outcome-free scores of smallcap/score.py. Results: results/.
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
from events.evaluate import holm, permutation_test, regress  # noqa: E402
from events.returns import load_closes, reactions  # noqa: E402
from smallcap.build import open_db  # noqa: E402

OUT = Path(__file__).resolve().parent / os.environ.get('SMALLCAP_RESULTS', 'results')  # results_rep: replication
CACHE = Path(os.environ.get('SMALLCAP_CACHE', REPO / 'data' / 'cache' / 'smallcap'))

EVAL_START = '2020-01-01'          # releases filed from this day are evaluated; earlier ones only train
GPT_START = '2024-01-01'           # the gpt-4o-mini table (its learned weights need two months of 2023-11/12 first)
TRAIN_FIRST = '2015-01-01'
LATEST_PRICE = '2026-09-30'
WINDOWS = {'r_0_1': (0, 1), 'drift_2_60': (2, 60), 'drift_2_21': (2, 21)}
BENCH = 'IWM'
CUT = 0.01
N_PERMUTATIONS = 1000
CONTROLS = ['sue', 'revenue_surprise', 'sue_missing', 'revenue_surprise_missing', 'abn_pre', 'log_float']
FIXED = ['sector', 'year_quarter']
FLOAT_LAG_MONTHS = 9               # a public float is known 9 months after the day it was measured (設計_小型株.md)
RISK_FLAGS = ['demand_weakness', 'margin_pressure', 'cost_inflation', 'restructuring_or_layoffs', 'impairment_or_writedown',
              'legal_or_regulatory', 'supply_chain', 'currency_or_macro', 'debt_or_liquidity', 'management_change']
MODELS = {                         # column: name in the tables
    'dictionary': '辞書（学習なし）', 'finbert': 'FinBERT（学習なし）', 'finbert_ft': 'FinBERT（追加学習）',
    'chrono_ridge': 'ChronoGPT 最後の層＋リッジ', 'chrono_mid_ridge': 'ChronoGPT 26 層目＋リッジ',
    'gpt': 'gpt-4o-mini（学習なし）', 'gpt_ridge': 'gpt-4o-mini（点数に重みを学習）'}
PRIMARY = 'chrono_ridge'
PORTFOLIO_COST = 0.003             # one way; 0.005 reported as well
TOP = 0.2


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e4 = _load('e4_run', 'experiments/E4_finbert_finetune/run.py')
e5 = _load('e5_run', 'experiments/E5_chrono_embeddings/run.py')


# ---------- the release panel ----------

def releases(conn):
    rel = pd.read_sql('SELECT accession, symbol, cik, filing_date, accepted_at, accepted_et, event_date, doc_type, text_path '
                      'FROM earnings_releases', conn)
    rel = rel[rel.text_path.notna() & rel.doc_type.fillna('').str.startswith('EX-99') & rel.event_date.notna()]
    rel = rel.sort_values('accepted_at').drop_duplicates(['symbol', 'event_date'], keep='first')
    return rel[rel.filing_date >= TRAIN_FIRST].reset_index(drop=True)


def float_known(conn):
    """Valid public floats with the day each became known and whether that year's rank was in the band."""
    f = pd.read_sql('SELECT f.cik, f.year, f.float_end, f.public_float, COALESCE(b.in_band, 0) AS in_band '
                    'FROM sc_float f LEFT JOIN sc_band b USING (year, cik) WHERE f.valid = 1', conn)
    f['known'] = (pd.to_datetime(f.float_end) + pd.DateOffset(months=FLOAT_LAG_MONTHS)).dt.strftime('%Y-%m-%d')
    return f


def attach_float(frame, f, date_col):
    """The latest public float known on frame[date_col] (not older than 21 months): log_float and in_band."""
    a = frame[['cik', date_col]].reset_index().rename(columns={date_col: 'd'})
    a['d_ts'] = pd.to_datetime(a.d)
    b = f.assign(known_ts=pd.to_datetime(f.known), end_ts=pd.to_datetime(f.float_end)).sort_values('known_ts')
    m = pd.merge_asof(a.sort_values('d_ts'), b[['cik', 'known_ts', 'end_ts', 'public_float', 'in_band']],
                      left_on='d_ts', right_on='known_ts', by='cik', direction='backward')
    fresh = (m.d_ts - m.end_ts) <= pd.Timedelta(days=640)
    m['log_float'] = np.where(fresh, np.log(m.public_float), np.nan)
    m['in_band'] = np.where(fresh, m.in_band, 0).astype(int)
    return m.set_index('index')[['log_float', 'in_band']].reindex(frame.index)


def sectors(conn, rel):
    sic = pd.read_sql('SELECT cik, sic FROM sc_companies', conn).set_index('cik').sic
    s = rel.cik.map(sic).fillna('').astype(str).str[:2].replace('', '00')
    counts = s.value_counts()
    return s.where(s.map(counts) >= 50, '99')


def outcomes(conn, rel, days, dry):
    """Abnormal returns against IWM over the windows, the 21 days before, and the end day of each window."""
    index = {d: n for n, d in enumerate(days)}
    k = rel.event_date.map(index)
    end = lambda h: k.map(lambda i: days[int(i) + h] if pd.notna(i) and int(i) + h < len(days) else None)
    out = pd.DataFrame({'accession': rel.accession, 'window_end': end(1), 'window_end_60': end(60)})
    if dry:
        rng = np.random.default_rng(0)
        for name in [*WINDOWS, 'pre']:
            out[f'abn_{name}'] = rng.normal(0, 0.05, len(rel))
        out.loc[out.window_end_60.isna(), 'abn_drift_2_60'] = np.nan
        return out
    closes = load_closes(conn, sorted(set(rel.symbol)) + [BENCH, '^GSPC'], '2014-12-01', LATEST_PRICE)
    r = reactions(rel[['accession', 'symbol', 'event_date']], closes, days, {s: BENCH for s in rel.symbol}, WINDOWS)
    return out.merge(r[['accession'] + [f'abn_{n}' for n in [*WINDOWS, 'pre']]], on='accession', how='left')


def scores(conn):
    runs = dict(pd.read_sql('SELECT method, score_run_id FROM release_score_runs', conn).values)
    out = pd.DataFrame()
    for method in ('dictionary', 'finbert'):
        if method in runs:
            out = out.join(pd.read_sql('SELECT accession, value_num AS ' + method + ' FROM release_scores '
                                       'WHERE score_run_id = ? AND item = ?', conn, params=(int(runs[method]), 'net_tone'))
                           .set_index('accession'), how='outer')
    if 'llm' in runs:
        llm = pd.read_sql('SELECT accession, item, value_num, value_text FROM release_scores WHERE score_run_id = ?', conn,
                          params=(int(runs['llm']),))
        w = llm.pivot_table(index='accession', columns='item', values='value_num', aggfunc='first')
        flags = llm[llm.item == 'risk_flags'].set_index('accession').value_text.map(lambda t: json.loads(t) if t else [])
        g = pd.DataFrame(index=w.index.union(flags.index))
        g['tone'], g['outlook'] = w.get('tone'), w.get('outlook')
        g['guidance_null'] = w.get('guidance_change').isna().astype(float)
        g['guidance'] = w.get('guidance_change').fillna(0.0)
        for f in RISK_FLAGS:
            g[f'flag_{f}'] = flags.reindex(g.index).map(lambda x: float(f in x) if isinstance(x, list) else np.nan)
        g['gpt'] = g.tone + g.outlook + g.guidance          # H06's composite, not fitted
        out = out.join(g, how='outer')
    return out


def chrono(name, key, p, dry):
    """The vectors of a finished ChronoGPT step, one row per release of p (NaN when missing)."""
    parts = [np.load(f, allow_pickle=True) for f in sorted((CACHE / name).glob('part*.npz'))]
    if not parts and dry:            # the dry run may come before the scoring has finished
        return np.random.default_rng(2).normal(size=(len(p), 16))
    acc = np.concatenate([q['accession'] for q in parts])
    X = np.concatenate([q[key] for q in parts]).astype(np.float32)
    return pd.DataFrame(X, index=acc).reindex(p.accession).to_numpy()


def rolling(p, X, target='abn_r_0_1', end='window_end'):
    """E5's rolling ridge (quarterly, trained on releases whose window closed before the quarter began). Releases
    without a vector get no prediction; releases without the target are predicted but never trained on."""
    ok = np.isfinite(X).all(axis=1)
    frame = p.loc[ok, ['accession', 'accepted_et', end, target]].rename(columns={end: 'window_end', target: 'abn_r_0_1'})
    frame['window_end'] = frame.window_end.fillna('9999-12-31')
    pred, info = e5.rolling_ridge(frame.reset_index(drop=True), X[ok])
    return pred.reindex(p.accession.to_numpy()).to_numpy(), info


def finbert_ft(p, dry):
    """E4's fine-tuning on the 2-day reactions of the releases whose window closed before EVAL_START; predictions for
    the releases filed from EVAL_START."""
    train = p[(p.window_end < EVAL_START) & p.abn_r_0_1.notna()][['accession', 'abn_r_0_1', 'text_path']]
    test = p[p.filing_date >= EVAL_START][['accession', 'text_path']]
    if dry:                          # the GPU is busy with the scoring during the dry run: noise in place of the model
        pred = pd.Series(np.random.default_rng(3).normal(size=len(test)), index=test.accession)
        return pred.reindex(p.accession.to_numpy()).to_numpy(), {'n_train': int(len(train)), 'dry': True}
    model, tok, info = e4.train(train.reset_index(drop=True))
    pred = e4.predict(model, tok, test.reset_index(drop=True))
    del model
    return pred.reindex(p.accession.to_numpy()).to_numpy(), info


def panel(conn, dry):
    rel = releases(conn)
    days = [r[0] for r in conn.execute('SELECT date FROM trading_days ORDER BY date')]
    f = float_known(conn)
    p = rel.join(attach_float(rel, f, 'filing_date'))
    p['sector'] = sectors(conn, p)
    p['year_quarter'] = pd.PeriodIndex(p.accepted_et.str[:10], freq='Q').astype(str)
    p = p.merge(outcomes(conn, p, days, dry), on='accession', how='left')
    sur = pd.read_sql('SELECT accession, sue, revenue_surprise FROM release_surprises', conn)
    p = p.merge(sur, on='accession', how='left').merge(scores(conn), left_on='accession', right_index=True, how='left')
    return p.sort_values(['filing_date', 'accession']).reset_index(drop=True), days, f


def learned(p, dry):
    info = {}
    X = chrono('chrono_raw_v2015', 'mean_last', p, dry)
    p['chrono_ridge'], info['chrono_ridge'] = rolling(p, X)
    p['chrono_ridge_drift'], info['chrono_ridge_drift'] = rolling(p, X, 'abn_drift_2_60', 'window_end_60')
    X = chrono('chrono_clean_v2015', 'mean_mid', p, dry)
    p['chrono_mid_ridge'], info['chrono_mid_ridge'] = rolling(p, X)
    gcols = ['tone', 'outlook', 'guidance', 'guidance_null'] + [f'flag_{f}' for f in RISK_FLAGS]
    has = p[gcols].notna().all(axis=1)
    gp = p[has].reset_index(drop=True)
    pred, info['gpt_ridge'] = rolling(gp, gp[gcols].to_numpy(float))
    p['gpt_ridge'] = pd.Series(pred, index=gp.accession).reindex(p.accession).to_numpy()
    p['finbert_ft'], info['finbert_ft'] = finbert_ft(p, dry)
    return info


# ---------- regressions ----------

def winsorize(x):
    lo, hi = x.quantile(CUT), x.quantile(1 - CUT)
    return x.clip(lo, hi)


def prepare(d, focal, y, controls):
    d = d.copy()
    d['y'] = winsorize(d[y])
    for col in ('sue', 'revenue_surprise'):
        d[f'{col}_missing'] = d[col].isna().astype(float)
    for col in ('sue', 'revenue_surprise', 'abn_pre', 'abn_r_0_1'):
        if col in d:
            d[col] = winsorize(d[col])
    d[focal] = (d[focal] - d[focal].mean()) / d[focal].std()
    d['log_float'] = (d.log_float - d.log_float.mean()) / d.log_float.std()
    for col in controls:
        if not col.endswith('_missing'):
            d[col] = d[col].fillna(0.0)
    return d


def r_squared(d, xs):
    from events.evaluate import design
    X = design(d, xs, FIXED).to_numpy(float)
    yv = d['y'].to_numpy(float)
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ beta
    return 1 - resid @ resid / ((yv - yv.mean()) @ (yv - yv.mean()))


def fit(sample, focal, y, permute=0):
    controls = CONTROLS + (['abn_r_0_1'] if y != 'abn_r_0_1' else [])
    sample = sample.dropna(subset=[y, focal])
    if len(sample) < 100:                # too few releases to fit (reported, not fitted)
        return {'n': int(len(sample)), 'skipped': 'fewer than 100 releases'}
    d = prepare(sample, focal, y, controls)
    controls = [c for c in controls if d[c].std() > 0]
    row = regress(d, 'y', [focal, *controls], FIXED).loc[focal]
    p_one = row.p / 2 if row.coef > 0 else 1 - row.p / 2
    out = {'coef_bp_per_sd': float(row.coef * 1e4), 't': float(row.t), 'p_one_sided': float(p_one),
           'ci95_bp': [float((row.coef - 1.96 * row.se) * 1e4), float((row.coef + 1.96 * row.se) * 1e4)],
           'n': int(row.n), 'event_dates': int(row.clusters),
           'r2_gain_pt': float((r_squared(d, [focal, *controls]) - r_squared(d, controls)) * 100)}
    if permute:
        perm = permutation_test(d, 'y', focal, controls, FIXED, n=permute, seed=0)
        out['permutation_p_one_sided'] = float(perm['p_one_sided'])
    return out


# ---------- portfolio (secondary) ----------

def month_ends(days, start, end):
    s = pd.Series(pd.to_datetime(days), index=days)
    s = s[(s >= start) & (s <= end)]
    return list(s.groupby(s.dt.to_period('M')).apply(lambda x: x.index[-1]))


def portfolio(conn, p, days, f, col, start, dry):
    """Each month end: the band companies of the sample (latest known float), the latest release of the last 92 days
    with a score. Top fifth by the score, weighted by public float, against all band companies weighted by float.
    Held to the next month end; a missing next price counts as 0%."""
    ends = month_ends(days, start, LATEST_PRICE)
    comp = pd.read_sql('SELECT s.cik, s.symbol FROM sc_sample s', conn)
    if not dry:
        closes = load_closes(conn, sorted(comp.symbol), start, LATEST_PRICE)
    rng = np.random.default_rng(1)
    rows, prev = [], {'top': pd.Series(dtype=float), 'all': pd.Series(dtype=float)}
    for m0, m1 in zip(ends[:-1], ends[1:]):
        u = comp.copy()
        u['m'] = m0
        u = u.join(attach_float(u.set_index(u.index), f, 'm'))
        u = u[u.in_band == 1]
        recent = p[(p.filing_date <= m0) & (p.event_date <= m0) & (p.filing_date > (pd.Timestamp(m0) - pd.Timedelta(days=92))
                                                                    .strftime('%Y-%m-%d')) & p[col].notna()]
        recent = recent.sort_values('filing_date').groupby('symbol').tail(1).set_index('symbol')[col]
        if dry:
            ret = pd.Series(rng.normal(0.01, 0.1, len(u)), index=u.symbol)
        else:
            c0, c1 = closes.loc[:m0].iloc[-1], closes.loc[:m1].iloc[-1]
            ret = (c1 / c0 - 1).reindex(u.symbol).fillna(0.0)
            u = u[u.symbol.isin(c0.dropna().index)]
        w_all = np.exp(u.set_index('symbol').log_float)
        scored = recent.reindex(u.symbol).dropna()
        if len(scored) < 25:
            continue
        top = scored[scored >= scored.quantile(1 - TOP)].index
        w_top = w_all.reindex(top)
        w = {'top': w_top / w_top.sum(), 'all': w_all / w_all.sum()}
        r = {k: float((w[k] * ret.reindex(w[k].index).fillna(0.0)).sum()) for k in w}
        turnover = {k: 0.5 * float(w[k].sub(prev[k], fill_value=0).abs().sum()) for k in w}
        rows.append({'month_end': m0, 'n_universe': len(u), 'n_scored': len(scored), 'n_top': len(top),
                     'ret_top': r['top'], 'ret_all': r['all'], 'turnover_top': turnover['top'],
                     'turnover_all': turnover['all']})
        prev = {k: w[k] * (1 + ret.reindex(w[k].index).fillna(0.0)) / (1 + r[k]) for k in w}
    df = pd.DataFrame(rows)
    out = {'months': int(len(df))}
    for cost in (0.0, PORTFOLIO_COST, 0.005):
        ex = (df.ret_top - 2 * cost * df.turnover_top) - (df.ret_all - 2 * cost * df.turnover_all)
        out[f'excess_monthly_cost_{cost}'] = float(ex.mean())
        out[f'nw_t_cost_{cost}'] = newey_west_t(ex.to_numpy())
    out['turnover_top_mean'] = float(df.turnover_top.mean())
    return out, df


def newey_west_t(x, lag=3):
    x = np.asarray(x, float)
    n = len(x)
    if n < 5:
        return float('nan')
    e = x - x.mean()
    v = e @ e / n
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(v / n))


# ---------- main ----------

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = argparse.ArgumentParser()
    a.add_argument('--dry', action='store_true')
    args = a.parse_args()
    out_dir = OUT / ('dry' if args.dry else '')
    if not args.dry and (OUT / 'summary.json').exists():
        raise SystemExit('results/summary.json があります。S1 は実行済みです（事前登録どおり 1 回だけ）')
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    conn = open_db()
    p, days, f = panel(conn, args.dry)
    print(f'panel: {len(p)} releases ({time.time() - t0:.0f}s)', flush=True)
    info = learned(p, args.dry)
    print(f'learned predictions done ({time.time() - t0:.0f}s)', flush=True)
    n_perm = 20 if args.dry else N_PERMUTATIONS
    ev = p[(p.filing_date >= EVAL_START) & (p.in_band == 1)]
    ev_gpt = ev[(ev.filing_date >= GPT_START) & ev[['gpt', 'gpt_ridge']].notna().all(axis=1)]
    summary = {'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
               'dry': args.dry, 'n_panel': int(len(p)), 'n_eval': int(len(ev)), 'n_eval_gpt': int(len(ev_gpt)),
               'learning': info}
    # primary
    summary['primary'] = fit(ev, PRIMARY, 'abn_drift_2_60', permute=n_perm)
    pr = summary['primary']
    summary['primary']['verdict'] = 'supported' if pr['p_one_sided'] < 0.05 and pr['permutation_p_one_sided'] < 0.05 \
        else 'not supported'
    # secondary: drift, every other model (2020- and the gpt period), Holm within each family
    sec = {m: fit(ev, m, 'abn_drift_2_60') for m in ['dictionary', 'finbert', 'finbert_ft', 'chrono_mid_ridge']}
    sec_gpt = {m: fit(ev_gpt, m, 'abn_drift_2_60') for m in ['gpt', 'gpt_ridge', 'chrono_ridge']}
    for fam in (sec, sec_gpt):
        fitted = {k: v for k, v in fam.items() if 'p_one_sided' in v}
        for (k, v), ph in zip(fitted.items(), holm([v['p_one_sided'] for v in fitted.values()])):
            v['holm_p'] = float(ph)
    summary['secondary_drift'], summary['secondary_drift_gpt_period'] = sec, sec_gpt
    # the comparison tables (both targets, both periods; not judged)
    rows = []
    for period, sample, models in [('2020-', ev, ['dictionary', 'finbert', 'finbert_ft', 'chrono_ridge', 'chrono_mid_ridge']),
                                   ('2024- (gpt)', ev_gpt, list(MODELS))]:
        for y in ('abn_r_0_1', 'abn_drift_2_60', 'abn_drift_2_21'):
            for m in models:
                r = fit(sample, m, y)
                rows.append({'period': period, 'target': y, 'model': m, 'name': MODELS[m], **r})
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / 'comparison.csv', index=False)
    # robustness (not judged): 2021 onwards (fewer companies lost), the ridge trained on the drift itself (exploratory)
    summary['robust_2021'] = fit(ev[ev.filing_date >= '2021-01-01'], PRIMARY, 'abn_drift_2_60')
    summary['explore_ridge_on_drift'] = fit(ev, 'chrono_ridge_drift', 'abn_drift_2_60')
    # portfolio (secondary)
    port, by_month = portfolio(conn, p, days, f, PRIMARY, EVAL_START, args.dry)
    by_month.to_csv(out_dir / 'by_month.csv', index=False)
    summary['portfolio'] = port
    if not ev_gpt.empty:
        port_gpt, by_month_gpt = portfolio(conn, p, days, f, 'gpt_ridge', GPT_START, args.dry)
        by_month_gpt.to_csv(out_dir / 'by_month_gpt.csv', index=False)
        summary['portfolio_gpt_ridge'] = port_gpt
    summary['seconds'] = round(time.time() - t0)
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    p.drop(columns=['text_path']).to_csv(out_dir / 'panel.csv', index=False)
    print(json.dumps({k: summary[k] for k in ('n_panel', 'n_eval', 'n_eval_gpt', 'primary')}, ensure_ascii=False, indent=1))
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
