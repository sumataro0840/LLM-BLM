"""
Shares outstanding from yfinance, converted to the current split basis.
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


def to_current_split_basis(shares, splits, window_days=30, max_ratio=2.0):
    """One value per report date on today's split basis.

    1. multiply each report by every split that happened after it
    2. drop values more than max_ratio away from the median of reports within +-window_days
       (some reports before a split are already on the new basis)
    3. keep the daily median
    """
    if shares.empty:
        return pd.Series(dtype='int64')
    df = pd.DataFrame({'date': shares.index, 'shares': shares.values.astype(float)})
    df['shares'] *= [splits[splits.index > d].prod() if (splits.index > d).any() else 1.0 for d in df['date']]
    dates = pd.to_datetime(df['date'])
    near_median = [df.loc[(dates - d).abs() <= pd.Timedelta(days=window_days), 'shares'].median() for d in dates]
    ratio = df['shares'] / near_median
    df = df[(ratio <= max_ratio) & (ratio >= 1 / max_ratio)]
    return df.groupby('date')['shares'].median().round().astype('int64')


def class_ratios(symbols):
    """Share of each listed class in their combined current shares outstanding, e.g. {'GOOGL': 0.51, 'GOOG': 0.49}."""
    counts = {s: yf.Ticker(s).info['sharesOutstanding'] for s in symbols}
    total = sum(counts.values())
    return {s: c / total for s, c in counts.items()}


def clip_to_period(shares, start, end):
    """Reports between start and end, plus the last report before start (the value in force on day one)."""
    if shares.empty:
        return shares
    before = shares[shares.index < start]
    inside = shares[(shares.index >= start) & (shares.index <= end)]
    return pd.concat([before.iloc[-1:], inside])
