"""
H02: does the LLM forecast (run 1) know more than a simple function of its own input and a statistical model?

    python experiments/H02_statistical_baseline/run.py

No API calls. Writes results/summary.json, results/monthly_ic.csv, results/fama_macbeth.csv.
Decision rules: 資料集/LLM-BLM_実験仮説.pdf, chapter H02.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import data as D
from experiments.common.metrics import fama_macbeth, t_summary

OUT = Path(__file__).resolve().parent / 'results'
FEATURES = ['momentum_12m', 'return_1m', 'relative_momentum_12m', 'volatility_30d', 'beta_1y', 'earnings_yield',
            'book_to_price', 'revenue_growth_yoy', 'net_income_growth_yoy', 'net_margin', 'roe', 'debt_to_equity']
RIDGE_ALPHA = 100.0          # fixed before the run; the training set has ~50,000 rows, so this is a light shrinkage
MIN_EFFECT = 0.03            # smallest IC that matters (fixed in the design)


def zscore(s):
    """Cross-sectional z-score after clipping at the 1st / 99th percentiles; missing -> 0 (the average)."""
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index)
    lo, hi = s.quantile([0.01, 0.99])
    s = s.clip(lo, hi)
    sd = s.std()
    return ((s - s.mean()) / sd).fillna(0.0) if sd > 0 else pd.Series(0.0, index=s.index)


def rank_score(s):
    """Ranks scaled to (-0.5, 0.5), ties averaged."""
    r = s.rank(method='average')
    return (r - 0.5) / r.count() - 0.5


def load_metric_panel(conn):
    """Features of every member on every month-end since 2015 and the next month's holding return."""
    m = pd.read_sql("SELECT as_of_date, symbol, metric, value FROM stock_metrics", conn)
    wide = m.pivot_table(index=['as_of_date', 'symbol'], columns='metric', values='value')
    wide['earnings_yield'] = 1 / wide['pe_trailing']        # pe exists only for positive earnings
    wide['book_to_price'] = 1 / wide['pb']
    dates = sorted(wide.index.get_level_values(0).unique())
    # prices at every month-end (forward-filled, so a stock that stopped trading keeps its last price)
    px = pd.read_sql("SELECT date, symbol, adj_close FROM prices WHERE date >= ?", conn, params=(dates[0],))
    px = px.pivot(index='date', columns='symbol', values='adj_close').sort_index().ffill(limit=None)
    rows = []
    for i, d in enumerate(dates[:-1]):
        end = dates[i + 1]   # the next month-end is the last trading day of the next month
        start_px, end_px = px.loc[d], px.loc[end]
        alive = px.loc[d].notna()
        r = (end_px / start_px - 1).where(alive)
        rows.append(pd.DataFrame({'as_of_date': d, 'symbol': r.index, 'R_next': r.values}))
    nxt = pd.concat(rows).set_index(['as_of_date', 'symbol'])
    members = pd.read_sql("SELECT m.symbol, m.start_date, m.end_date FROM index_membership m JOIN tickers t "
                          "ON t.symbol = m.symbol AND t.excluded_reason IS NULL", conn)
    panel = wide.join(nxt, how='left').reset_index()
    ok = []
    for d, g in panel.groupby('as_of_date'):
        mem = members[(members.start_date.isna() | (members.start_date <= d)) & (members.end_date.isna() | (members.end_date > d))]
        ok.append(g[g.symbol.isin(set(mem.symbol))])
    return pd.concat(ok), dates


def statistical_scores(panel, eval_dates, outcome_known):
    """Walk-forward ridge: for each evaluation date, fit on months whose outcome is known before it."""
    X_all, y_all = {}, {}
    for d, g in panel.groupby('as_of_date'):
        g = g.set_index('symbol')
        X_all[d] = pd.DataFrame({f: zscore(g[f]) for f in FEATURES})
        y_all[d] = rank_score(g['R_next']) if g['R_next'].notna().sum() > 50 else None
    scores, coefs = {}, {}
    for d in eval_dates:
        train = [m for m in X_all if outcome_known(m, d) and y_all[m] is not None]
        X = pd.concat([X_all[m].loc[y_all[m].dropna().index] for m in train])
        y = pd.concat([y_all[m].dropna() for m in train])
        model = Ridge(alpha=RIDGE_ALPHA).fit(X.values, y.values)
        scores[d] = pd.Series(model.predict(X_all[d].values), index=X_all[d].index)
        coefs[d] = dict(zip(FEATURES, model.coef_)) | {'n_train_months': len(train), 'n_train_rows': len(y)}
    return scores, pd.DataFrame(coefs).T


def residual_on(y, xs):
    """Residual of a cross-sectional OLS of y on xs (with an intercept)."""
    X = np.column_stack([np.ones(len(y)), *xs])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta, 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    all_data = D.load_all(conn)
    eval_dates = list(all_data)
    panel, metric_dates = load_metric_panel(conn)
    # the outcome of month m is known on date d when the next month-end after m is on or before d
    nxt = {m: metric_dates[i + 1] for i, m in enumerate(metric_dates[:-1])}
    known = lambda m, d: m in nxt and nxt[m] <= d
    stat, coefs = statistical_scores(panel, eval_dates, known)
    coefs.to_csv(OUT / 'ridge_coefficients.csv')

    frames, monthly = {}, []
    for d in eval_dates:
        x = all_data[d]
        u = x['universe']
        df = pd.DataFrame({
            'Q': x['view_mean'], 'r_in': x['input_mean'].reindex(u), 'sd_in': x['input_sd'].reindex(u),
            'R_next': x['next_return'].reindex(u), 'r_next': x['next_mean_daily'].reindex(u),
            'stat': stat[d].reindex(u),
        })
        met = panel[panel.as_of_date == d].set_index('symbol')
        df['rev'] = -met['return_1m'].reindex(u)
        df['mom'] = met['momentum_12m'].reindex(u)
        df = df.dropna(subset=['Q', 'r_in', 'sd_in', 'R_next'])
        # ranks (-0.5..0.5) used for the regressions
        rk = df.apply(rank_score)
        # LLM replica: the LLM forecast explained by its input (same month, no outcome used)
        fitted_resid, r2 = residual_on(df.Q.values, [df.r_in.values, df.sd_in.values])
        df['replica'] = df.Q - fitted_resid
        res_rank, r2_rank = residual_on(rk.Q.values, [rk.r_in.values, rk.sd_in.values])
        df['llm_residual'] = res_rank
        rk['llm_residual'] = rank_score(pd.Series(res_rank, index=df.index))
        rk['y'] = rk['R_next']
        frames[d] = rk
        ic = lambda col: stats.spearmanr(df[col], df.R_next, nan_policy='omit')[0]
        dm = lambda s: s - s.mean()
        row = {'date': d, 'n': len(df), 'replica_r2': r2, 'replica_r2_rank': r2_rank,
               'ic_llm': ic('Q'), 'ic_replica': ic('replica'), 'ic_llm_residual': ic('llm_residual'),
               'ic_rev': ic('rev'), 'ic_mom': ic('mom'), 'ic_stat': ic('stat'), 'ic_input_mean': ic('r_in'),
               'pearson_llm': df[['Q', 'R_next']].corr().iloc[0, 1],
               'slope_llm': float((dm(df.Q) * dm(df.r_next)).sum() / (dm(df.Q) ** 2).sum()),
               'mae_llm': float((df.Q - df.r_next).abs().mean()), 'rmse_llm': float(np.sqrt(((df.Q - df.r_next) ** 2).mean())),
               'mae_replica': float((df.replica - df.r_next).abs().mean()),
               'rmse_replica': float(np.sqrt(((df.replica - df.r_next) ** 2).mean())),
               'mae_zero': float(df.r_next.abs().mean()),
               'direction_llm': float((np.sign(dm(df.Q)) == np.sign(dm(df.R_next))).mean()),
               'direction_stat': float((np.sign(dm(df.stat)) == np.sign(dm(df.R_next))).mean())}
        monthly.append(row)
    mon = pd.DataFrame(monthly).set_index('date')
    mon.to_csv(OUT / 'monthly_ic.csv')

    fm_table, fm = fama_macbeth(frames, 'y', ['Q', 'r_in', 'sd_in'])
    fm_table.to_csv(OUT / 'fama_macbeth.csv')
    _, fm_stat = fama_macbeth(frames, 'y', ['Q', 'stat'])

    primary = t_summary(mon.ic_llm_residual)
    if primary['p'] < 0.05 and primary['mean'] > 0:
        verdict = 'rejected (the LLM has predictive power beyond its input)'
    elif primary['ci_high'] < MIN_EFFECT:
        verdict = 'supported (no LLM-specific predictive power of 0.03 or more)'
    else:
        verdict = 'insufficient evidence (not significant, but an IC of 0.03 or more is not excluded)'
    summary = {
        'months': len(mon), 'stocks_per_month': float(mon.n.mean()),
        'replica_r2_mean': float(mon.replica_r2.mean()), 'replica_r2_rank_mean': float(mon.replica_r2_rank.mean()),
        'primary_ic_llm_residual': primary, 'verdict': verdict,
        'ic': {c: t_summary(mon[c]) for c in ['ic_llm', 'ic_replica', 'ic_llm_residual', 'ic_rev', 'ic_mom', 'ic_stat', 'ic_input_mean']},
        'ic_difference_stat_minus_llm': t_summary(mon.ic_stat - mon.ic_llm),
        'fama_macbeth_llm_with_inputs': fm,
        'fama_macbeth_llm_with_stat': fm_stat,
        'slope_llm_mean': float(mon.slope_llm.mean()),
        'pearson_llm_mean': float(mon.pearson_llm.mean()),
        'mae': {'llm': float(mon.mae_llm.mean()), 'replica': float(mon.mae_replica.mean()), 'zero': float(mon.mae_zero.mean())},
        'rmse': {'llm': float(mon.rmse_llm.mean()), 'replica': float(mon.rmse_replica.mean())},
        'direction_accuracy': {'llm': float(mon.direction_llm.mean()), 'stat': float(mon.direction_stat.mean())},
        'ridge': {'alpha': RIDGE_ALPHA, 'features': FEATURES,
                  'first_training_months': int(coefs.n_train_months.iloc[0]), 'last_training_months': int(coefs.n_train_months.iloc[-1]),
                  'coef_last': {f: float(coefs[f].iloc[-1]) for f in FEATURES}},
    }
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
