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


def load_monthly_returns(tickers, month_start, month_end, save_dir):
    """Download close prices, save prices and daily returns as CSV, and return the returns."""
    data = yf.download(tickers, start=month_start, end=month_end)['Close']
    data.to_csv(f'{save_dir}/data_{month_start}_{month_end}.csv')

    returns = data.pct_change().iloc[1:]
    returns.to_csv(f'{save_dir}/returns_{month_start}_{month_end}.csv')
    return returns
