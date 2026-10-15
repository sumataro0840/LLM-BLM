"""
Shares outstanding and split history from yfinance.
"""
import time

import pandas as pd
import yfinance as yf


def download_shares(symbol, start, end, attempts=5, wait=30):
    """Raw reported shares (may hold several values per day); yfinance has them from about late 2015.
    yfinance refuses requests for a while after many in a row, so a failed request is tried again later."""
    for attempt in range(attempts):
        try:
            shares = yf.Ticker(symbol).get_shares_full(start=start, end=end)
            break
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(wait * (attempt + 1))
    shares = pd.Series(dtype=float) if shares is None else shares
    shares.index = pd.to_datetime(shares.index).strftime('%Y-%m-%d')
    return shares


def download_splits(symbol):
    """Split history (split date -> ratio); empty when yfinance does not know the ticker."""
    try:
        splits = yf.Ticker(symbol).splits
    except Exception:
        splits = None
    if splits is None or splits.empty:
        return pd.Series(dtype=float, index=pd.Index([], dtype=object))
    splits.index = pd.to_datetime(splits.index).strftime('%Y-%m-%d')
    return splits


def class_ratios(symbols):
    """Share of each listed class in their combined current shares outstanding, e.g. {'GOOGL': 0.51, 'GOOG': 0.49}."""
    counts = {s: yf.Ticker(s).info['sharesOutstanding'] for s in symbols}
    total = sum(counts.values())
    return {s: c / total for s, c in counts.items()}
