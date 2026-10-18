import sys, json, numpy as np, pandas as pd
H = sys.argv[1]
base = pd.read_pickle(H + 'results.pkl'); base = base[(base.group == 'industrials') & base.cond.isin(['with']) & base.date.isin(['2025-06-30', '2025-12-31', '2026-05-29'])].assign(cond='baseline')
t = pd.concat([base, pd.read_pickle(H + 'weights.pkl')], ignore_index=True)
ans = t.answers.map(json.loads)
t['mean'] = [np.mean([x for x in a if x is not None]) if isinstance(a, list) else np.nan for a in ans]
t['sd'] = [np.std([x for x in a if x is not None]) if isinstance(a, list) else np.nan for a in ans]
print("failed:", t['mean'].isna().sum(), "| stock-dates per condition:", t.groupby('cond').size().to_dict())
def rc(df, m):
    cs = [g['mean'].rank().corr(g[m].rank()) for _, g in df.groupby('date')]
    return np.nanmean(cs)
M = [('return_1m', '1m return'), ('momentum_12m', 'momentum'), ('volatility_30d', 'volatility'), ('beta_1y', 'beta'),
     ('pe_trailing', 'P/E'), ('pb', 'P/B'), ('revenue_growth_yoy', 'revenue growth'), ('net_income_growth_yoy', 'income growth'), ('net_margin', 'net margin'), ('roe', 'ROE'), ('debt_to_equity', 'debt/equity')]
order = ['baseline', 'p50_f50', 'p20_f80', 'p0_f100']
print(pd.DataFrame({c: {label: rc(t[t.cond == c], m) for m, label in M} for c in order}).round(2).to_string())
print(pd.DataFrame({c: {'mean prediction': t[t.cond == c]['mean'].mean(), 'cross-sectional sd': t[t.cond == c].groupby('date')['mean'].std().mean(), 'sd of 30 answers': t[t.cond == c].sd.mean()} for c in order}).round(5).to_string())
w = t.pivot_table(index=['date', 'symbol'], columns='cond', values='mean')
print("rank agreement with baseline:", {c: round(np.mean([g['baseline'].rank().corr(g[c].rank()) for _, g in w.groupby('date')]), 2) for c in order[1:]})
