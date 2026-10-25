"""
S2: in small caps, does the reading of earnings releases predict the volatility after the release, beyond past
volatility and the size of the 2-day reaction? (preregistration.md)

    python experiments/S2_smallcap_risk/run.py --dry    # the whole pipeline with noise in place of every price
    python experiments/S2_smallcap_risk/run.py          # once, after preregistration.md and this file are committed

Reuses S1's release panel, ChronoGPT vectors and rolling ridge (experiments/S1_smallcap_drift/run.py). Results: results/.
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
from events.returns import load_closes  # noqa: E402
from smallcap.build import open_db  # noqa: E402

OUT = Path(__file__).resolve().parent / os.environ.get('SMALLCAP_RESULTS', 'results')  # results_rep: replication
spec = importlib.util.spec_from_file_location('s1_run', REPO / 'experiments' / 'S1_smallcap_drift' / 'run.py')
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)

EVAL_START, GPT_START, LATEST_PRICE, BENCH = s1.EVAL_START, s1.GPT_START, s1.LATEST_PRICE, s1.BENCH
FIXED = s1.FIXED
CUT = 0.01
N_PERMUTATIONS = 1000
ANNUAL = np.sqrt(252)
# windows in trading days relative to day 0 (inclusive) and the minimum number of daily returns
VOL_WINDOWS = {'rv_2_60': (2, 60, 40), 'rv_2_21': (2, 21, 15), 'rv_pre': (-60, -1, 40), 'rv_long': (-252, -61, 120)}
TARGETS = {'log_rv_2_60': '2〜60 日の変動率', 'log_rv_2_21': '2〜21 日の変動率', 'log_ivol_2_60': '2〜60 日の固有の変動率',
           'abs_r_0_1': '2 日の反応の大きさ'}
BASE = ['log_rv_pre', 'log_rv_long', 'rv_long_missing', 'log_float', 'abs_sue', 'sue_missing', 'abs_rev', 'rev_missing']
SCORES = {'chrono_risk': 'ChronoGPT 最後の層＋リッジ（主）', 'chrono_mid_risk': 'ChronoGPT 26 層目＋リッジ',
          'dict_uncertainty': '辞書：不確実さの語', 'dict_negative': '辞書：否定的な語',
          'finbert_negative': 'FinBERT：否定的な文', 'gpt_flags': 'gpt-4o-mini：リスクの印の数',
          'gpt_risk': 'gpt-4o-mini：14 項目＋リッジ'}
PRIMARY = 'chrono_risk'
GCOLS = ['tone', 'outlook', 'guidance', 'guidance_null'] + [f'flag_{f}' for f in s1.RISK_FLAGS]


# ---------- outcomes and controls ----------

def vol_outcomes(conn, p, days, dry):
    """Annualized sd of daily log returns over each window (NaN when too few returns or the window passes
    LATEST_PRICE), and the same for the return minus IWM's over days 2-60."""
    out = pd.DataFrame({'accession': p.accession})
    if dry:
        rng = np.random.default_rng(10)
        for name in VOL_WINDOWS:
            out[name] = np.exp(rng.normal(np.log(0.5), 0.4, len(p)))
        out['ivol_2_60'] = np.exp(rng.normal(np.log(0.4), 0.4, len(p)))
        out.loc[p.window_end_60.isna().to_numpy(), ['rv_2_60', 'ivol_2_60']] = np.nan
        return out
    closes = load_closes(conn, sorted(set(p.symbol)) + [BENCH], '2013-12-01', LATEST_PRICE).reindex(days)
    r = np.log(closes).diff()
    arr, bench = r.to_numpy(float), r[BENCH].to_numpy(float)
    col = {s: i for i, s in enumerate(r.columns)}
    index = {d: n for n, d in enumerate(days)}
    last = max(i for d, i in index.items() if d <= LATEST_PRICE)
    res = {name: np.full(len(p), np.nan) for name in [*VOL_WINDOWS, 'ivol_2_60']}

    def sd(x, n_min):
        x = x[np.isfinite(x)]
        return np.std(x, ddof=1) * ANNUAL if len(x) >= n_min else np.nan

    for row, (sym, ev) in enumerate(zip(p.symbol, p.event_date)):
        k, j = index.get(ev), col.get(sym)
        if k is None or j is None:
            continue
        for name, (a, b, n_min) in VOL_WINDOWS.items():
            if k + a >= 1 and k + b <= last:
                res[name][row] = sd(arr[k + a:k + b + 1, j], n_min)
        if k + 60 <= last:
            res['ivol_2_60'][row] = sd(arr[k + 2:k + 61, j] - bench[k + 2:k + 61], 40)
    for name, v in res.items():
        out[name] = v
    return out


def extra_scores(conn):
    """Outcome-free scores not used by S1: dictionary uncertainty and negative shares, FinBERT's negative share."""
    runs = dict(pd.read_sql('SELECT method, score_run_id FROM release_score_runs', conn).values)
    out = pd.DataFrame()
    for method, item, name in (('dictionary', 'uncertainty', 'dict_uncertainty'), ('dictionary', 'negative', 'dict_negative'),
                               ('finbert', 'negative', 'finbert_negative')):
        if method in runs:
            s = pd.read_sql('SELECT accession, value_num AS v FROM release_scores WHERE score_run_id = ? AND item = ?', conn,
                            params=(int(runs[method]), item)).set_index('accession').v.rename(name)
            out = out.join(s, how='outer') if len(out) else s.to_frame()
    return out


def build(conn, dry):
    p, days, _ = s1.panel(conn, dry)
    p = p.merge(vol_outcomes(conn, p, days, dry), on='accession', how='left')
    p = p.merge(extra_scores(conn), left_on='accession', right_index=True, how='left')
    for name in ('rv_2_60', 'rv_2_21', 'rv_pre', 'ivol_2_60'):
        p[f'log_{name}'] = np.log(p[name])
    p['rv_long_missing'] = p.rv_long.isna().astype(float)
    p['log_rv_long'] = np.log(p.rv_long).fillna(p.log_rv_pre)
    p['abs_r_0_1'] = p.abn_r_0_1.abs()
    p['sue_missing'] = p.sue.isna().astype(float)
    p['rev_missing'] = p.revenue_surprise.isna().astype(float)
    p['abs_sue'] = p.sue.abs()
    p['abs_rev'] = p.revenue_surprise.abs()
    p['vol_surprise'] = p.log_rv_2_60 - p.log_rv_pre          # the ridge's target (learned on closed windows only)
    flags = [f'flag_{f}' for f in s1.RISK_FLAGS]
    p['gpt_flags'] = p[flags].sum(axis=1, min_count=len(flags))
    return p, days


def learned(p, dry):
    info = {}
    X = s1.chrono('chrono_raw_v2015', 'mean_last', p, dry)
    p['chrono_risk'], info['chrono_risk'] = s1.rolling(p, X, 'vol_surprise', 'window_end_60')
    X = s1.chrono('chrono_clean_v2015', 'mean_mid', p, dry)
    p['chrono_mid_risk'], info['chrono_mid_risk'] = s1.rolling(p, X, 'vol_surprise', 'window_end_60')
    has = p[GCOLS].notna().all(axis=1)
    gp = p[has].reset_index(drop=True)
    pred, info['gpt_risk'] = s1.rolling(gp, gp[GCOLS].to_numpy(float), 'vol_surprise', 'window_end_60')
    p['gpt_risk'] = pd.Series(pred, index=gp.accession).reindex(p.accession).to_numpy()
    return info


# ---------- regressions ----------

def winsorize(x):
    return x.clip(x.quantile(CUT), x.quantile(1 - CUT))


def prepare(sample, focal, y, controls):
    d = sample.copy()
    d['y'] = winsorize(d[y])
    for c in ('log_rv_pre', 'log_rv_long', 'abs_sue', 'abs_rev', 'abs_r_0_1'):
        if c in controls:
            d[c] = winsorize(d[c])
    d[focal] = (d[focal] - d[focal].mean()) / d[focal].std()
    d['log_float'] = (d.log_float - d.log_float.mean()) / d.log_float.std()
    for c in controls:
        d[c] = d[c].fillna(0.0)
    return d


def fit(sample, focal, y, permute=0, with_reaction=True):
    """Coefficient of the standardized score: in % (log points x 100) for the log volatilities, in bp for |abn_r_0_1|."""
    controls = BASE + (['abs_r_0_1'] if with_reaction and y != 'abs_r_0_1' else [])
    sample = sample.dropna(subset=[y, focal, 'log_rv_pre'])
    if len(sample) < 100:
        return {'n': int(len(sample)), 'skipped': 'fewer than 100 releases'}
    d = prepare(sample, focal, y, controls)
    controls = [c for c in controls if d[c].std() > 0]
    row = regress(d, 'y', [focal, *controls], FIXED).loc[focal]
    unit = 1e4 if y == 'abs_r_0_1' else 100
    p_one = row.p / 2 if row.coef > 0 else 1 - row.p / 2
    out = {'coef': float(row.coef * unit), 'unit': 'bp' if unit == 1e4 else '%', 't': float(row.t),
           'p_one_sided': float(p_one), 'ci95': [float((row.coef - 1.96 * row.se) * unit),
                                                 float((row.coef + 1.96 * row.se) * unit)],
           'n': int(row.n), 'event_dates': int(row.clusters),
           'r2_gain_pt': float((s1.r_squared(d, [focal, *controls]) - s1.r_squared(d, controls)) * 100)}
    if permute:
        out['permutation_p_one_sided'] = float(permutation_test(d, 'y', focal, controls, FIXED, n=permute, seed=0)['p_one_sided'])
    return out


def with_holm(family):
    fitted = {k: v for k, v in family.items() if 'p_one_sided' in v}
    for (k, v), ph in zip(fitted.items(), holm([v['p_one_sided'] for v in fitted.values()])):
        v['holm_p'] = float(ph)
    return family


# ---------- out-of-sample forecasts (secondary 4) ----------

def forecasts(p, ev):
    """Quarterly OLS on the releases whose 60-day window closed before the quarter; base vs base + chrono_risk."""
    xs_base = BASE + ['abs_r_0_1']
    need = ['log_rv_2_60', 'log_rv_pre', PRIMARY]
    q = pd.PeriodIndex(p.accepted_et.str[:10], freq='Q')
    q_ev = pd.PeriodIndex(ev.accepted_et.str[:10], freq='Q')
    rows = []
    for quarter in sorted(q_ev.unique()):
        start = quarter.start_time.strftime('%Y-%m-%d')
        tr = p[(p.window_end_60.fillna('9999') < start)].dropna(subset=need)
        te = ev[q_ev == quarter].dropna(subset=need)
        if len(tr) < 500 or te.empty:
            continue
        lo, hi = tr[xs_base + [PRIMARY]].quantile(CUT), tr[xs_base + [PRIMARY]].quantile(1 - CUT)
        clip = lambda f: f[xs_base + [PRIMARY]].clip(lo, hi, axis=1).fillna(0.0)
        Xtr, Xte = clip(tr), clip(te)
        for name, cols in (('base', xs_base), ('with_score', xs_base + [PRIMARY])):
            A = np.column_stack([np.ones(len(Xtr)), Xtr[cols].to_numpy(float)])
            beta = np.linalg.lstsq(A, tr.log_rv_2_60.to_numpy(float), rcond=None)[0]
            te = te.assign(**{f'pred_{name}': np.column_stack([np.ones(len(Xte)), Xte[cols].to_numpy(float)]) @ beta})
        rows.append(te[['accession', 'event_date', 'log_rv_2_60', 'pred_base', 'pred_with_score']])
    f = pd.concat(rows, ignore_index=True)
    sst = ((f.log_rv_2_60 - f.log_rv_2_60.mean()) ** 2).sum()
    loss = {k: (f.log_rv_2_60 - f[f'pred_{k}']) ** 2 for k in ('base', 'with_score')}
    f['loss_diff'] = loss['base'] - loss['with_score']
    f['one'] = 1.0
    dm = regress(f, 'loss_diff', [], cluster='event_date').loc['const']
    return {'n': int(len(f)), 'quarters': int(len(rows)),
            'r2_oos_base': float(1 - loss['base'].sum() / sst), 'r2_oos_with_score': float(1 - loss['with_score'].sum() / sst),
            'mse_ratio': float(loss['with_score'].sum() / loss['base'].sum()), 'loss_diff_mean': float(f.loss_diff.mean()),
            'loss_diff_t': float(dm.t)}, f


# ---------- main ----------

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = argparse.ArgumentParser()
    a.add_argument('--dry', action='store_true')
    args = a.parse_args()
    out_dir = OUT / ('dry' if args.dry else '')
    if not args.dry and (OUT / 'summary.json').exists():
        raise SystemExit('results/summary.json があります。S2 は実行済みです（事前登録どおり 1 回だけ）')
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    conn = open_db()
    p, days = build(conn, args.dry)
    print(f'panel: {len(p)} releases ({time.time() - t0:.0f}s)', flush=True)
    info = learned(p, args.dry)
    print(f'learned predictions done ({time.time() - t0:.0f}s)', flush=True)
    n_perm = 20 if args.dry else N_PERMUTATIONS
    ev = p[(p.filing_date >= EVAL_START) & (p.in_band == 1)]
    ev_gpt = ev[(ev.filing_date >= GPT_START) & ev[['gpt_flags', 'gpt_risk']].notna().all(axis=1)]
    summary = {'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
               'dry': args.dry, 'n_panel': int(len(p)), 'n_eval': int(len(ev)), 'n_eval_gpt': int(len(ev_gpt)),
               'learning': info}
    pr = summary['primary'] = fit(ev, PRIMARY, 'log_rv_2_60', permute=n_perm)
    pr['verdict'] = 'supported' if pr['p_one_sided'] < 0.05 and pr['permutation_p_one_sided'] < 0.05 else 'not supported'
    summary['secondary'] = with_holm({m: fit(ev, m, 'log_rv_2_60')
                                      for m in ['chrono_mid_risk', 'dict_uncertainty', 'dict_negative', 'finbert_negative']})
    summary['secondary_gpt_period'] = with_holm({m: fit(ev_gpt, m, 'log_rv_2_60') for m in ['gpt_flags', 'gpt_risk', PRIMARY]})
    rows = []
    for period, sample, models in [('2020-', ev, [m for m in SCORES if not m.startswith('gpt')]),
                                   ('2024- (gpt)', ev_gpt, list(SCORES))]:
        for y in TARGETS:
            for m in models:
                rows.append({'period': period, 'target': y, 'model': m, 'name': SCORES[m], **fit(sample, m, y)})
    pd.DataFrame(rows).drop(columns=['ci95'], errors='ignore').to_csv(out_dir / 'comparison.csv', index=False)
    summary['forecast'], fc = forecasts(p, ev)
    fc.drop(columns=['one']).to_csv(out_dir / 'forecasts.csv', index=False, float_format='%.5f')
    summary['robust_2021'] = fit(ev[ev.filing_date >= '2021-01-01'], PRIMARY, 'log_rv_2_60')
    summary['robust_no_reaction'] = fit(ev, PRIMARY, 'log_rv_2_60', with_reaction=False)
    summary['seconds'] = round(time.time() - t0)
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    keep = ['accession', 'symbol', 'filing_date', 'event_date', 'in_band', 'sector', 'year_quarter', 'window_end_60',
            *TARGETS, 'log_rv_pre', 'log_rv_long', 'vol_surprise', *SCORES]
    p[keep].to_csv(out_dir / 'panel.csv', index=False)
    print(json.dumps({k: summary[k] for k in ('n_panel', 'n_eval', 'n_eval_gpt', 'primary', 'forecast')},
                     ensure_ascii=False, indent=1))
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
