"""
Monthly price download (yfinance) and daily returns.
"""
from datetime import timedelta

import pandas as pd
import yfinance as yf


def month_ranges(start, end):
    """[(month_start, month_end), ...] for every month from start to end."""
    ranges = []
    for current_date in pd.date_range(start=start, end=end, freq='MS'):
        month_start = current_date.strftime('%Y-%m-%d')
        month_end = (current_date + pd.DateOffset(months=1) - timedelta(days=1)).strftime('%Y-%m-%d')
        ranges.append((month_start, month_end))
    return ranges


def download_prices(tickers, month_start, month_end):
    """Daily close prices (rows: dates, columns: tickers)."""
    return yf.download(tickers, start=month_start, end=month_end)['Close']


def to_daily_returns(prices):
    """Daily returns (the first day of the month is dropped)."""
    return prices.pct_change().iloc[1:]
