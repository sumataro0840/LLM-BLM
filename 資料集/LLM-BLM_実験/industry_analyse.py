import sys, json, numpy as np, pandas as pd
t = pd.read_pickle(sys.argv[1])
ans = t.answers.map(json.loads)
t['ok'] = ans.map(lambda a: isinstance(a, list))
t['mean'] = [np.mean([x for x in a if x is not None]) if isinstance(a, list) else np.nan for a in ans]
t['sd'] = [np.std([x for x in a if x is not None]) if isinstance(a, list) else np.nan for a in ans]
print("failed:", (~t.ok).sum(), "| answers per request ok")
w = t.pivot_table(index=['group', 'date', 'symbol'], columns='cond', values='mean').dropna()
meta = t[t.cond == 'with'].set_index(['group', 'date', 'symbol'])
w['effect'] = w['with'] - w['na']          # what showing the metrics does
w['noise'] = w['na2'] - w['na']            # same prompt asked twice
HID = {'reit': ['pe_trailing', 'net_income_growth_yoy', 'net_margin', 'roe'], 'bank': ['net_margin', 'debt_to_equity'],
       'industrials': ['pe_trailing', 'net_income_growth_yoy', 'net_margin', 'roe']}
def rank_corr(df, a, b):
    """Spearman within each date, averaged (and the number of dates where it is positive)."""
    cs = [g[a].rank().corr(g[b].rank()) for _, g in df.groupby('date') if g[[a, b]].dropna().shape[0] >= 8]
    return np.nanmean(cs), sum(c > 0 for c in cs), len(cs)
for group in ['reit', 'bank', 'industrials']:
    d = w.loc[group].join(meta.loc[group][[c for c in meta.columns if c.endswith('_pct')]])
    print(f"\n=== {group}: {len(d)} stock-dates ===")
    print(f"prediction level (daily): with {d['with'].mean():.5f}  n/a {d['na'].mean():.5f} | cross-sectional sd of predictions {d['na'].groupby('date').std().mean():.5f}")
    print(f"size of change |with - n/a|: {d.effect.abs().mean():.5f}   vs noise |n/a2 - n/a|: {d.noise.abs().mean():.5f}   ratio {d.effect.abs().mean() / d.noise.abs().mean():.2f}")
    print(f"agreement of rankings: with~n/a {rank_corr(d, 'with', 'na')[0]:.2f}   n/a~n/a2 {rank_corr(d, 'na', 'na2')[0]:.2f}")
    for m in HID[group]:
        e = rank_corr(d, 'effect', m + '_pct'); n = rank_corr(d, 'noise', m + '_pct'); lv = rank_corr(d, 'with', m + '_pct'); l0 = rank_corr(d, 'na', m + '_pct')
        print(f"  {m:24s} pull of the metric on the change: {e[0]:+.2f} (positive in {e[1]}/{e[2]} dates) | noise {n[0]:+.2f} | prediction~metric: shown {lv[0]:+.2f}, hidden {l0[0]:+.2f}")
    for m, label in [('momentum_12m', 'momentum'), ('return_1m', '1m return'), ('pb', 'P/B')]:
        lv = rank_corr(d, 'with', m + '_pct'); print(f"  (always shown) prediction~{label}: {lv[0]:+.2f}")
print("\nanswer spread within a request (sd of 30 answers), mean by group/cond:")
print(t.groupby(['group', 'cond']).sd.mean().unstack().round(5).to_string())
