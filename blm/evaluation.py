"""
Performance figures from monthly returns.
"""
import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def performance(returns, risk_free, benchmark=None):
    """returns, risk_free, benchmark: Series of monthly returns on the same dates."""
    excess = returns - risk_free
    wealth = (1 + returns).cumprod()
    figures = {
        'months': len(returns),
        'total_return': wealth.iloc[-1] - 1,
        'annual_return': wealth.iloc[-1] ** (MONTHS_PER_YEAR / len(returns)) - 1,
        'annual_volatility': returns.std() * np.sqrt(MONTHS_PER_YEAR),
        'sharpe_ratio': excess.mean() / returns.std() * np.sqrt(MONTHS_PER_YEAR),
        'max_drawdown': (wealth / wealth.cummax() - 1).min(),
        'worst_month': returns.min(),
    }
    if benchmark is not None:
        active = returns - benchmark
        figures['annual_active_return'] = active.mean() * MONTHS_PER_YEAR
        figures['tracking_error'] = active.std() * np.sqrt(MONTHS_PER_YEAR)
        figures['months_above_benchmark'] = int((active > 0).sum())
    return pd.Series(figures)


def turnover(weights):
    """Average share of the portfolio replaced per month (weights: rows = dates, columns = symbols)."""
    return weights.fillna(0).diff().abs().sum(axis=1).iloc[1:].mean() / 2


def effective_holdings(weights):
    """1 / sum(w^2), averaged over the dates: the number of equally weighted stocks with the same concentration."""
    return (1 / (weights.fillna(0) ** 2).sum(axis=1)).mean()
