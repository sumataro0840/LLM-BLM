"""
Figures shared by the experiments: forecast accuracy (IC), portfolio performance, simple tests.
"""
import numpy as np
import pandas as pd
from scipy import stats

MONTHS = 12


def t_summary(values):
    """Mean, standard deviation, t-value, two-sided p-value and 95% interval of monthly values."""
    v = pd.Series(values, dtype=float).dropna()
    n = len(v)
    mean, sd = v.mean(), v.std(ddof=1)
    se = sd / np.sqrt(n)
    t = mean / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else np.nan
    half = stats.t.ppf(0.975, n - 1) * se
    return {'n': n, 'mean': mean, 'sd': sd, 't': t, 'p': p, 'ci_low': mean - half, 'ci_high': mean + half,
            'positive': int((v > 0).sum())}


def rank_ic(signal, outcome):
    """Spearman correlation over the stocks present in both (ties get average ranks)."""
    df = pd.concat([signal, outcome], axis=1).dropna()
    if len(df) < 5 or df.iloc[:, 0].nunique() < 2:
        return np.nan
    return stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])[0]


def ic_by_date(signals, outcomes):
    """signals, outcomes: {date: Series by symbol}. Returns a Series of the monthly rank IC."""
    return pd.Series({d: rank_ic(signals[d], outcomes[d]) for d in signals if d in outcomes})


def fama_macbeth(frames, y, xs):
    """Monthly cross-sectional OLS of y on xs (with an intercept); the time series of the coefficients
    gives the mean and its t-value. frames: {date: DataFrame}."""
    coefs = []
    for d, df in frames.items():
        df = df[[y, *xs]].dropna()
        if len(df) < len(xs) + 5:
            continue
        X = np.column_stack([np.ones(len(df)), df[xs].values])
        beta = np.linalg.lstsq(X, df[y].values, rcond=None)[0]
        coefs.append(pd.Series(beta[1:], index=xs, name=d))
    table = pd.DataFrame(coefs)
    return table, {x: t_summary(table[x]) for x in xs}


def performance(result, benchmark, use='net'):
    """result, benchmark: frames from backtest.run on the same dates."""
    r, b = result[use], benchmark[use]
    rf = result['risk_free']
    wealth = (1 + r).cumprod()
    active = r - b
    te = active.std(ddof=1) * np.sqrt(MONTHS)
    excess = r - rf
    return {
        'annual_return': wealth.iloc[-1] ** (MONTHS / len(r)) - 1,
        'annual_volatility': r.std(ddof=1) * np.sqrt(MONTHS),
        'sharpe': excess.mean() / r.std(ddof=1) * np.sqrt(MONTHS),
        'max_drawdown': (wealth / wealth.cummax() - 1).min(),
        'active_return': active.mean() * MONTHS,
        'tracking_error': te,
        'information_ratio': active.mean() * MONTHS / te if te > 0 else np.nan,
        't_active': t_summary(active)['t'],
        'p_active': t_summary(active)['p'],
        'turnover': result['turnover'].iloc[1:].mean(),
        'effective_n': result['effective_n'].mean(),
        'active_share': result['active_share'].mean(),
        'active_beta': result['active_beta'].mean(),
        'abs_active_beta': result['active_beta'].abs().mean(),
        'max_active_weight': result['max_active_weight'].mean(),
    }


def holm(pvalues):
    """Holm-adjusted p-values (same order as given)."""
    p = pd.Series(pvalues, dtype=float)
    order = p.sort_values().index
    m = len(p)
    adjusted, running = pd.Series(index=p.index, dtype=float), 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        adjusted[k] = running
    return adjusted
