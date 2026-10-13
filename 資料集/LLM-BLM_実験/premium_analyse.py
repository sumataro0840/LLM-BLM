import sys, json, sqlite3, numpy as np, pandas as pd
H = '/Users/kazuki/develop/工房/LLM-BLM_実験/'
conn = sqlite3.connect("file:/Users/kazuki/develop/工房/LLM-BLM/data/blm.sqlite?mode=ro", uri=True)
t = pd.read_pickle(H + 'premium_results.pkl'); a = t.answer.map(json.loads)
P = np.array([x['probs'] for x in a]); t['mass'] = P.sum(1); P = P / P.sum(1, keepdims=True); k = np.arange(1, 6)
t['score'] = P @ k; t['sd'] = np.sqrt(P @ k ** 2 - t.score ** 2); t['p_up'] = P[:, 3] + P[:, 4]; t['p_down'] = P[:, 0] + P[:, 1]
t['sampled'] = [np.mean([int(s) for s in x['samples'] if s in list('12345')]) for x in a]
print("requests", len(t), "| probability mass on 1-5: min", t.mass.min().round(3), "| corr(score from probabilities, mean of 30 samples):", round(t.score.corr(t.sampled), 3))
print("share of each digit (probability):", dict(zip(k, P.mean(0).round(3))))

# realized returns over the month after each decision date
dates = sorted(t.date.unique()); ends = [r[0] for r in conn.execute("SELECT MAX(date) FROM prices WHERE symbol='^GSPC' GROUP BY substr(date,1,7) ORDER BY 1")]
nxt = {d: ends[ends.index(d) + 1] for d in dates}
px = pd.read_sql("SELECT date, symbol, adj_close FROM prices WHERE date IN (%s)" % ",".join("'%s'" % x for x in set(dates) | set(nxt.values())), conn).pivot(index='date', columns='symbol', values='adj_close')
etf = dict(conn.execute("SELECT gics_sector, symbol FROM tickers WHERE gics_sector IS NOT NULL AND symbol NOT IN (SELECT symbol FROM index_membership)").fetchall())
real = []
for d in dates:
    r = px.loc[nxt[d]] / px.loc[d] - 1
    m = pd.read_sql("SELECT symbol, metric, value FROM stock_metrics WHERE as_of_date = ?", conn, params=(d,)).pivot(index='symbol', columns='metric', values='value')
    m['cap'] = pd.read_sql("SELECT symbol, market_cap FROM index_weights WHERE date = ?", conn, params=(d,), index_col='symbol').market_cap
    def spread(col, high_is_top):
        x = m[col].dropna(); x = x[x.index.intersection(r.dropna().index)]; q = x.rank(pct=True)
        top, bot = r[q[q > 2 / 3].index].mean(), r[q[q <= 1 / 3].index].mean()
        return top - bot if high_is_top else bot - top
    tb = conn.execute("SELECT close FROM prices WHERE symbol='^IRX' AND date <= ? ORDER BY date DESC LIMIT 1", (d,)).fetchone()[0] / 100 / 12
    real += [(d, 'market', r['^GSPC'] - tb), (d, 'beta', spread('beta_1y', True)), (d, 'volatility', spread('volatility_30d', True)), (d, 'value', spread('pe_trailing', False)),
             (d, 'momentum', spread('momentum_12m', True)), (d, 'safety', spread('debt_to_equity', False)), (d, 'size', spread('cap', False))]
    real += [(d, s, r[e] - r['^GSPC']) for s, e in etf.items()]
t = t.merge(pd.DataFrame(real, columns=['date', 'name', 'realized']), on=['date', 'name'])
t.to_pickle(H + 'premium_scored.pkl')

f = t[t.kind == 'factor']
rows = {}
for name, g in f.groupby('name', sort=False):
    call = np.sign(g.score - 3); timed = (call * g.realized)
    rows[name] = {'mean score': g.score.mean(), 'sd over months': g.score.std(), 'min': g.score.min(), 'max': g.score.max(), 'sd of the answer': g.sd.mean(),
                  'months score>3': (g.score > 3).sum(), 'months realized>0': (g.realized > 0).sum(), 'hit rate': (np.sign(g.realized) == call).mean(),
                  'rank corr score~realized': g.score.rank().corr(g.realized.rank()), 'mean realized %': g.realized.mean() * 100, 'mean timed %': timed.mean() * 100}
pd.set_option('display.width', 250)
print("\n=== factors (30 months each) ===\n" + pd.DataFrame(rows).T.round(2).to_string())
print("\nwhat the factor scores follow (rank corr over 30 months):")
print(pd.DataFrame({n: {c: g.score.rank().corr(g[c].rank()) for c in ['sp_1m', 'sp_12m', 'vix', 'vol_premium', 'vix_ratio', 'skew', 'tbill']} for n, g in f.groupby('name', sort=False)}).round(2).to_string())

s = t[t.kind == 'sector'].copy()
zm = pd.read_sql("SELECT as_of_date date, gics_sector name, metric, z_score FROM sector_metric_scores", conn).pivot_table(index=['date', 'name'], columns='metric', values='z_score').reset_index()
s = s.merge(zm, on=['date', 'name'])
ic = s.groupby('date').apply(lambda g: g.score.rank().corr(g.realized.rank()))
print(f"\n=== sectors: 11 per month, 30 months ===\nmean rank corr score~realized (per month): {ic.mean():+.3f}  t = {ic.mean() / ic.std() * np.sqrt(len(ic)):.2f}  positive in {(ic > 0).sum()}/{len(ic)} months")
for c in ['momentum_3m', 'momentum_6m', 'momentum_12m', 'volatility_30d', 'pe_trailing']:
    x = s.groupby('date').apply(lambda g: g.score.rank().corr(g[c].rank())); y = s.groupby('date').apply(lambda g: g[c].rank().corr(g.realized.rank()))
    print(f"  score ~ sector {c}: {x.mean():+.2f}   | (reference) {c} ~ realized: {y.mean():+.3f}")
top = s.groupby('date').apply(lambda g: g.nlargest(3, 'score').realized.mean() - g.nsmallest(3, 'score').realized.mean())
print(f"top 3 minus bottom 3 sectors by score, next month: mean {top.mean() * 100:+.2f}% per month, t = {top.mean() / top.std() * np.sqrt(len(top)):.2f}")
print("score by sector (mean over months):", s.groupby('name').score.mean().round(2).sort_values().to_dict())
print("sector score: sd across sectors within a month", s.groupby('date').score.std().mean().round(2), "| sd of the answer", s.sd.mean().round(2), "| digit shares", dict(zip(k, np.array([json.loads(x)['probs'] for x in s.answer]).mean(0).round(2))))
