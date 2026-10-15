"""
Shares outstanding converted to the current split basis.
"""
import pandas as pd


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


def extend_back(recent, older):
    """`recent` (yfinance), with the values of `older` (EDGAR) before its first date.

    The two can count differently (share classes, units), so `older` is scaled to the level of `recent`:
    the median ratio of the two on the dates of `older` that `recent` also covers. Without such dates it is kept as is.
    """
    if recent.empty or older.empty:
        return older if recent.empty else recent
    first = recent.index[0]
    ratios = [recent[recent.index <= d].iloc[-1] / v for d, v in older[older.index >= first].items()]
    scale = pd.Series(ratios).median() if ratios else 1.0
    before = (older[older.index < first] * scale).round().astype('int64')
    return pd.concat([before, recent])


def clip_to_period(shares, start, end):
    """Reports between start and end, plus the last report before start (the value in force on day one)."""
    if shares.empty:
        return shares
    before = shares[shares.index < start]
    inside = shares[(shares.index >= start) & (shares.index <= end)]
    return pd.concat([before.iloc[-1:], inside])
