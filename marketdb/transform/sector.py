"""
Sector indicators on a decision date: momentum and volatility from the sector ETF,
valuation from the financial statements of the members.
"""
import numpy as np
import pandas as pd

MOMENTUM_MONTHS = [3, 6, 12]
VOLATILITY_DAYS = 30
VOLATILITY_SMOOTHING_DAYS = 5
TRADING_DAYS_PER_YEAR = 252


def momentum(month_end_prices, months):
    """Return over `months` months that ends one month before the last row
    (the latest month is left out because short-term returns tend to reverse).

    month_end_prices: rows = last trading day of each month up to the decision date, columns = sectors.
    NaN when the history is too short.
    """
    if len(month_end_prices) < months + 2:
        return pd.Series(np.nan, index=month_end_prices.columns)
    return month_end_prices.iloc[-2] / month_end_prices.iloc[-2 - months] - 1


def volatility(daily_prices):
    """Yearly volatility of the daily returns over VOLATILITY_DAYS trading days,
    averaged over the last VOLATILITY_SMOOTHING_DAYS days (one unusual day moves it less)."""
    rolling = daily_prices.pct_change().rolling(VOLATILITY_DAYS).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return rolling.iloc[-VOLATILITY_SMOOTHING_DAYS:].mean(skipna=False)


def trailing_year(quarters):
    """Last four quarters of each company -> net income and revenue summed, equity of the latest quarter.

    quarters: columns symbol, period_end, revenue, net_income, total_equity; only quarters filed by the decision date.
    A company without four quarters spanning about a year is left out.
    """
    rows = {}
    for symbol, q in quarters.sort_values('period_end').groupby('symbol'):
        q = q.tail(4)
        span = (pd.Timestamp(q['period_end'].iloc[-1]) - pd.Timestamp(q['period_end'].iloc[0])).days
        if len(q) == 4 and 250 <= span <= 300:
            # min_count: a missing quarter makes the sum missing instead of too small
            rows[symbol] = {'net_income': q['net_income'].sum(min_count=4), 'revenue': q['revenue'].sum(min_count=4),
                            'total_equity': q['total_equity'].iloc[-1]}
    return pd.DataFrame.from_dict(rows, orient='index', columns=['net_income', 'revenue', 'total_equity'])


def valuation(market_caps, sectors, fundamentals):
    """Sector ratios: market cap of the members / their net income (pe_trailing), equity (pb), revenue (ps).
    Each ratio uses the members that have the item, so the numerator and the denominator cover the same companies.

    market_caps, sectors: Series indexed by symbol (members on the decision date); fundamentals: trailing_year().
    """
    table = pd.DataFrame({'market_cap': market_caps, 'sector': sectors}).join(fundamentals)
    out = {}
    for sector, members in table.groupby('sector'):
        row = {}
        for metric, item in [('pe_trailing', 'net_income'), ('pb', 'total_equity'), ('ps', 'revenue')]:
            known = members.dropna(subset=[item])
            if known[item].sum() > 0:
                row[metric] = known['market_cap'].sum() / known[item].sum()
        row['valuation_coverage'] = members.dropna(subset=['net_income'])['market_cap'].sum() / members['market_cap'].sum()
        out[sector] = row
    return pd.DataFrame.from_dict(out, orient='index')
