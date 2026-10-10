"""
Months to process and daily returns given to the LLM.
"""
from datetime import timedelta

import pandas as pd


def month_ranges(start, end):
    """[(month_start, month_end), ...] for every month from start to end."""
    ranges = []
    for current_date in pd.date_range(start=start, end=end, freq='MS'):
        month_start = current_date.strftime('%Y-%m-%d')
        month_end = (current_date + pd.DateOffset(months=1) - timedelta(days=1)).strftime('%Y-%m-%d')
        ranges.append((month_start, month_end))
    return ranges


def to_daily_returns(prices):
    """Daily returns (the first day of the month is dropped)."""
    return prices.pct_change().iloc[1:]
