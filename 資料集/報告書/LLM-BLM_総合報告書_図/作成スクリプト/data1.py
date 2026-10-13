"""Numbers and tables for the report, straight from the database."""
import sys, json, sqlite3, numpy as np, pandas as pd
sys.path.insert(0, '/Users/kazuki/develop/工房/LLM-BLM')
from blm.evaluation import performance, turnover, effective_holdings
OUT = '/private/tmp/claude-501/-Users-kazuki-develop---/9d2180f2-ca7f-42f9-a43d-a8ea8ccec7c4/scratchpad/report/build/'
conn = sqlite3.connect("file:/Users/kazuki/develop/工房/LLM-BLM/data/blm.sqlite?mode=ro", uri=True)
N = {}
# --- row counts
names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
N['rows'] = {n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}
N['views'] = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")]
# --- portfolios
runs = pd.read_sql("SELECT portfolio_id, strategy, params, note FROM portfolio_runs", conn)
ret = pd.read_sql("SELECT portfolio_id, as_of_date, end_date, period_return, n_holdings FROM portfolio_returns", conn)
R = ret.pivot(index='as_of_date', columns='portfolio_id', values='period_return'); R.columns = [runs.set_index('portfolio_id').strategy[c] for c in R.columns]
ends = ret.drop_duplicates('as_of_date').set_index('as_of_date').end_date
irx = pd.read_sql("SELECT date, close FROM prices WHERE symbol='^IRX'", conn, index_col='date').close
gs = pd.read_sql("SELECT date, close FROM prices WHERE symbol='^GSPC'", conn, index_col='date').close
rf = pd.Series({d: irx[:d].iloc[-1] / 100 / 12 for d in R.index})
R['sp500_index'] = [gs[ends[d]] / gs[d] - 1 for d in R.index]
perf = {}
W = pd.read_sql("SELECT portfolio_id, as_of_date, symbol, weight FROM portfolio_weights", conn)
for pid, s in runs.set_index('portfolio_id').strategy.items():
    w = W[W.portfolio_id == pid].pivot(index='as_of_date', columns='symbol', values='weight')
    p = performance(R[s], rf, R['market_cap']).to_dict(); p['turnover'] = float(turnover(w)); p['effective_holdings'] = float(effective_holdings(w)); p['max_weight'] = float(w.max(axis=1).mean())
    perf[s] = p
perf['sp500_index'] = performance(R['sp500_index'], rf, R['market_cap']).to_dict()
N['perf'] = perf
R.to_csv(OUT + 'monthly_returns.csv')
eff = {s: (1 / (W[W.portfolio_id == pid].pivot(index='as_of_date', columns='symbol', values='weight').fillna(0) ** 2).sum(axis=1)) for pid, s in runs.set_index('portfolio_id').strategy.items()}
pd.DataFrame(eff).to_csv(OUT + 'effective_holdings.csv')
# --- LLM views per month
v = pd.read_sql("SELECT as_of_date, symbol, mean_return, variance, n_answers FROM llm_view_stats WHERE run_id = 1", conn)
month_ends = [r[0] for r in conn.execute("SELECT MAX(date) FROM prices WHERE symbol='^GSPC' GROUP BY substr(date,1,7) ORDER BY 1")]
px = pd.read_sql("SELECT date, symbol, adj_close FROM prices WHERE symbol IN (SELECT symbol FROM index_membership)", conn).pivot(index='date', columns='symbol', values='adj_close')
daily = px.pct_change()
rows = []; scat = None
for d in sorted(v.as_of_date.unique()):
    g = v[v.as_of_date == d].set_index('symbol'); i = month_ends.index(d); prev, nxt = month_ends[i - 1], month_ends[i + 1]
    inp = daily.loc[(daily.index > prev) & (daily.index <= d), g.index]
    in_mean, in_sd = inp.mean(), inp.std()
    real = (px.loc[nxt, g.index] / px.loc[d, g.index] - 1)
    ndays = ((daily.index > d) & (daily.index <= nxt)).sum()
    ok = in_mean.notna()
    rows.append({'as_of_date': d, 'n': len(g), 'view_mean': g.mean_return.mean(), 'view_sd_cs': g.mean_return.std(), 'share_pos': (g.mean_return > 0).mean(),
                 'input_mean': in_mean.mean(), 'corr_input': g.mean_return[ok].corr(in_mean[ok]), 'slope': np.polyfit(in_mean[ok], g.mean_return[ok], 1)[0],
                 'rank_sd_vol': np.sqrt(g.variance[ok]).rank().corr(in_sd[ok].rank()), 'realized_daily': (real / ndays).mean(),
                 'rank_ic': g.mean_return.rank().corr(real.rank())})
    if d == '2025-06-30': scat = pd.DataFrame({'input_mean': in_mean, 'view': g.mean_return, 'realized': real / ndays})
M = pd.DataFrame(rows); M.to_csv(OUT + 'views_monthly.csv', index=False); scat.to_csv(OUT + 'scatter_2025-06.csv')
N['views'] = {'corr_input_mean': M.corr_input.mean(), 'corr_input_min': M.corr_input.min(), 'corr_input_max': M.corr_input.max(), 'slope_mean': M.slope.mean(), 'rank_sd_vol': M.rank_sd_vol.mean(),
              'rank_ic_mean': M.rank_ic.mean(), 'rank_ic_pos': int((M.rank_ic > 0).sum()), 'view_mean': v.mean_return.mean(), 'share_pos_views': (v.mean_return > 0).mean(), 'realized_mean': M.realized_daily.mean(),
              'n_ticker_months': len(v), 'n_min': int(M.n.min()), 'n_max': int(M.n.max())}
allv = pd.read_sql("SELECT expected_return FROM llm_views WHERE run_id = 1", conn).expected_return
N['answers'] = {'n': len(allv), 'null': int(allv.isna().sum()), 'min': allv.min(), 'p1': allv.quantile(.01), 'median': allv.median(), 'p99': allv.quantile(.99), 'max': allv.max(), 'share_pos': (allv > 0).mean(), 'mean': allv.mean(), 'abs_gt_5pct': int((allv.abs() >= 0.05).sum())}
np.save(OUT + 'answers_sample.npy', allv.sample(60000, random_state=0).values)
# --- index check
iw = pd.read_sql("SELECT date, symbol, weight FROM index_weights", conn).pivot(index='date', columns='symbol', values='weight')
cl = pd.read_sql("SELECT date, symbol, close FROM prices WHERE symbol IN (SELECT symbol FROM index_membership)", conn).pivot(index='date', columns='symbol', values='close')
dr = cl.pct_change()
idx = (iw.shift(1) * dr[iw.columns]).sum(axis=1).iloc[1:]; g = gs.pct_change().reindex(idx.index)
N['index_check'] = {'corr': idx.corr(g), 'mean_abs_diff': (idx - g).abs().mean(), 'total_db': (1 + idx).prod() - 1, 'total_gspc': (1 + g).prod() - 1, 'members_min': int(iw.notna().sum(axis=1).min()), 'members_max': int(iw.notna().sum(axis=1).max())}
pd.DataFrame({'db': (1 + idx).cumprod(), 'gspc': (1 + g).cumprod()}).to_csv(OUT + 'index_check.csv')
# --- reports
rep = pd.read_sql("SELECT r.report_month, r.published_date, r.n_pages, v.item, v.value_num FROM reports r LEFT JOIN report_values v ON v.report_id = r.report_id AND v.report_run_id = 2 AND v.value_num IS NOT NULL", conn)
rp = rep.pivot_table(index=['report_month', 'published_date', 'n_pages'], columns='item', values='value_num').reset_index(); rp.to_csv(OUT + 'reports.csv', index=False)
# --- sector metrics example and heatmap
sm = pd.read_sql("SELECT as_of_date, gics_sector, metric, value, z_score FROM sector_metric_scores", conn); sm.to_csv(OUT + 'sector_metrics.csv', index=False)
N['stock_metrics'] = pd.read_sql("SELECT metric, COUNT(*) n, COUNT(DISTINCT as_of_date) dates FROM stock_metrics GROUP BY 1", conn).set_index('metric').to_dict('index')
N['stock_metrics_20260529'] = pd.read_sql("SELECT metric, COUNT(*) n, MIN(value) lo, AVG(value) mean, MAX(value) hi FROM stock_metrics WHERE as_of_date='2026-05-29' GROUP BY 1", conn).set_index('metric').round(4).to_dict('index')
ex = pd.read_sql("SELECT symbol, metric, value, sector_percentile FROM stock_metric_scores WHERE as_of_date='2026-05-29' AND symbol IN ('NVDA','JPM','PLD','XOM')", conn); ex.to_csv(OUT + 'stock_example.csv', index=False)
# --- filing age
q = pd.read_sql("SELECT symbol, period_end, filed_date FROM fundamentals_quarterly WHERE filed_date IS NOT NULL", conn, parse_dates=['period_end', 'filed_date'])
ages = []
for d in sorted(v.as_of_date.unique()):
    t = pd.Timestamp(d); mem = v[v.as_of_date == d].symbol
    k = q[(q.filed_date <= t) & q.symbol.isin(mem)].sort_values('period_end').groupby('symbol').tail(1)
    ages += list((t - k.period_end).dt.days)
ages = np.array(ages); np.save(OUT + 'ages.npy', ages)
N['ages'] = {'n': len(ages), 'median': float(np.median(ages)), 'p95': float(np.quantile(ages, .95)), 'p99': float(np.quantile(ages, .99)), 'max': float(ages.max()), 'gt150': int((ages > 150).sum()), 'share_gt150': float((ages > 150).mean())}
N['membership'] = {'tickers': N['rows']['tickers'], 'excluded': conn.execute("SELECT COUNT(*) FROM tickers WHERE excluded_reason IS NOT NULL").fetchone()[0], 'stocks': conn.execute("SELECT COUNT(DISTINCT symbol) FROM index_membership").fetchone()[0],
                   'changes_in_period': conn.execute("SELECT COUNT(*) FROM index_changes WHERE date BETWEEN '2023-03-01' AND '2026-09-30'").fetchone()[0]}
N['llm_run'] = dict(zip(['model', 'model_version', 'temperature', 'n_samples', 'created_at'], conn.execute("SELECT model, model_version, temperature, n_samples, created_at FROM llm_runs WHERE run_id=1").fetchone()))
N['system_prompt'] = conn.execute("SELECT system_prompt FROM llm_runs WHERE run_id=1").fetchone()[0]
N['report_prompt'] = conn.execute("SELECT system_prompt FROM report_runs WHERE report_run_id=2").fetchone()[0]
N['portfolio_runs'] = runs.to_dict('records')
json.dump(N, open(OUT + 'numbers.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: N[k] for k in ['rows', 'perf', 'views', 'answers', 'index_check', 'ages', 'membership', 'llm_run']}, indent=1, default=float)[:6000])
