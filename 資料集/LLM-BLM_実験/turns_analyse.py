import sys, json, numpy as np, pandas as pd
H = '/Users/kazuki/develop/工房/LLM-BLM_実験/'
D3 = ['2025-06-30', '2025-12-31', '2026-05-29']
base = pd.read_pickle(H + 'industry_results.pkl'); base = base[base.group.isin(['industrials', 'reit']) & base.date.isin(D3)]
base = base.replace({'cond': {'with': 'one_number', 'na': 'one_number_na', 'na2': 'one_number_na2'}})
t = pd.concat([base, pd.read_pickle(H + 'turns_results.pkl')], ignore_index=True)
ans = t.answers.map(json.loads)
vals = lambda a: [x for x in a if x is not None] if isinstance(a, list) else []
t['mean'] = [np.mean(vals(a)) if vals(a) else np.nan for a in ans]; t['sd'] = [np.std(vals(a)) if vals(a) else np.nan for a in ans]
t['n'] = [len(vals(a)) for a in ans]
print("failed requests:", (t.n == 0).sum(), "| unparsed answers:", (30 - t.n[t.n > 0]).sum())
def rc(df, m):
    cs = [g['mean'].rank().corr(g[m].rank()) for _, g in df.groupby('date') if g[m].notna().sum() >= 8]
    return np.nanmean(cs) if cs else np.nan
M = ['return_1m', 'momentum_12m', 'volatility_30d', 'beta_1y', 'pe_trailing', 'pb', 'revenue_growth_yoy', 'net_income_growth_yoy', 'net_margin', 'roe', 'debt_to_equity']
order = ['one_number', 'three_number', 'one_rating', 'three_rating']
res = {}
for g in ['industrials', 'reit']:
    d = t[t.group == g]
    tab = pd.DataFrame({c: {m: rc(d[d.cond == c], m) for m in M} for c in order}).round(2)
    print(f"\n=== {g}: {d[d.cond == 'one_number'].shape[0]} stock-dates ===\n" + tab.to_string())
    w = d.pivot_table(index=['date', 'symbol'], columns='cond', values='mean')
    agree = lambda a, b: np.mean([x[a].rank().corr(x[b].rank()) for _, x in w.groupby('date')])
    print("rank agreement: " + " | ".join(f"{a}~{b} {agree(a, b):.2f}" for a, b in [('one_number_na', 'one_number_na2'), ('one_number', 'three_number'), ('one_number', 'one_rating'), ('one_number', 'three_rating'), ('three_number', 'three_rating')]))
    num = d[d.cond.isin(['one_number', 'three_number'])]
    flat = {c: np.array([x for a in num[num.cond == c].answers.map(json.loads) for x in vals(a)]) for c in ['one_number', 'three_number']}
    print({c: {'mean pred': round(num[num.cond == c]['mean'].mean(), 5), 'cs sd': round(num[num.cond == c].groupby('date')['mean'].std().mean(), 5), 'sd of 30': round(num[num.cond == c].sd.mean(), 5), '>1%/day': round((np.abs(v) > 0.01).mean(), 3)} for c, v in flat.items()})
    for c in ['one_rating', 'three_rating']:
        v = np.array([x for a in d[d.cond == c].answers.map(json.loads) for x in vals(a)])
        print(c, 'rating shares', {k: round((v == k).mean(), 2) for k in [-2, -1, 0, 1, 2]}, '| mean', round(v.mean(), 2), '| cs sd of mean', round(d[d.cond == c].groupby('date')['mean'].std().mean(), 2), '| sd of 30', round(d[d.cond == c].sd.mean(), 2))
    res[g] = tab
# multiple regression on ranks: which inputs carry the prediction when taken together (industrials)
d = t[t.group == 'industrials']
X = ['return_1m', 'momentum_12m', 'volatility_30d', 'pe_trailing', 'pb', 'revenue_growth_yoy', 'net_income_growth_yoy', 'roe']
print("\nstandardised coefficients of a regression of the prediction rank on input ranks (industrials, ranks within date)")
out = {}
for c in order:
    x = d[d.cond == c].dropna(subset=X + ['mean']).copy()
    r = x.groupby('date')[X + ['mean']].rank(pct=True); r = (r - r.mean()) / r.std()
    beta, *_ = np.linalg.lstsq(r[X].values, r['mean'].values, rcond=None)
    pred = r[X].values @ beta; out[c] = dict(zip(X, beta.round(2)), R2=round(1 - ((r['mean'] - pred) ** 2).sum() / (r['mean'] ** 2).sum(), 2), n=len(r))
print(pd.DataFrame(out).to_string())
