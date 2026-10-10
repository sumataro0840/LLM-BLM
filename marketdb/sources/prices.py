"""
Daily price download (yfinance).
"""
from datetime import timedelta

import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end):
    """Daily prices from start to end, both inclusive.

    Columns are (field, ticker) with field 'Close' (split-adjusted only)
    and 'Adj Close' (split- and dividend-adjusted, used for returns).
    """
    # yfinance excludes the end date, so ask for one day more to keep the last day of the month
    end_exclusive = (pd.Timestamp(end) + timedelta(days=1)).strftime('%Y-%m-%d')
    return yf.download(tickers, start=start, end=end_exclusive, auto_adjust=False)[['Close', 'Adj Close']]
