"""
Shares outstanding and split history from yfinance.
"""
import pandas as pd
import yfinance as yf


def download_shares_and_splits(symbol, start, end):
    """Raw reported shares (may hold several values per day) and the split history."""
    ticker = yf.Ticker(symbol)
    shares = ticker.get_shares_full(start=start, end=end)
    splits = ticker.splits
    shares = pd.Series(dtype=float) if shares is None else shares
    shares.index = pd.to_datetime(shares.index).strftime('%Y-%m-%d')
    splits.index = pd.to_datetime(splits.index).strftime('%Y-%m-%d')
    return shares, splits


def class_ratios(symbols):
    """Share of each listed class in their combined current shares outstanding, e.g. {'GOOGL': 0.51, 'GOOG': 0.49}."""
    counts = {s: yf.Ticker(s).info['sharesOutstanding'] for s in symbols}
    total = sum(counts.values())
    return {s: c / total for s, c in counts.items()}
