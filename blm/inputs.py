"""
Everything the strategies need for one decision date, read from the database.
Only data up to the decision date is used.
"""
from dataclasses import dataclass

import pandas as pd

from marketdb.db.index import load_index_weights, load_members
from marketdb.db.llm import load_view_stats
from marketdb.db.market import last_trading_day_before, last_trading_days, load_prices

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Inputs:
    as_of_date: str
    returns: pd.DataFrame          # daily returns of the covariance window (rows = days, columns = universe)
    month_returns: pd.DataFrame    # daily returns of the decision month only (what the LLM saw)
    market_weights: pd.Series      # market-cap weights within the universe (sums to 1)
    view_mean: pd.Series           # LLM view: predicted average daily return
    view_variance: pd.Series       # variance of the LLM answers
    risk_free_daily: float         # 13-week T-bill yield on the decision date, per trading day


def load_inputs(conn, as_of_date, llm_run_id, window_days, min_history_ratio):
    """Universe = index members on the date that have an LLM view, a market cap and enough price history."""
    views = load_view_stats(conn, llm_run_id, as_of_date)
    weights = load_index_weights(conn, as_of_date)
    members = [s for s in load_members(conn, as_of_date) if s in views.index and s in weights.index]

    days = last_trading_days(conn, as_of_date, window_days + 1)
    prices = load_prices(conn, members, days[0], as_of_date)
    returns = prices.pct_change().iloc[1:]
    enough = returns.notna().mean() >= min_history_ratio
    has_last_price = prices.iloc[-1].notna()
    universe = sorted(s for s in members if enough.get(s, False) and has_last_price.get(s, False))
    returns = returns[universe]
    # days before a stock was listed: use the average return of the other stocks on that day
    returns = returns.apply(lambda col: col.fillna(returns.mean(axis=1)))

    month_start = pd.Timestamp(as_of_date).replace(day=1).strftime('%Y-%m-%d')
    month_first_day = pd.Timestamp(last_trading_day_before(conn, month_start))
    month_returns = returns[returns.index > month_first_day]

    risk_free = load_prices(conn, ['^IRX'], days[0], as_of_date, column='close')['^IRX'].dropna().iloc[-1]
    return Inputs(
        as_of_date=as_of_date,
        returns=returns,
        month_returns=month_returns,
        market_weights=weights[universe] / weights[universe].sum(),
        view_mean=views.loc[universe, 'mean_return'],
        view_variance=views.loc[universe, 'variance'],
        risk_free_daily=float(risk_free) / 100 / TRADING_DAYS_PER_YEAR,
    )
