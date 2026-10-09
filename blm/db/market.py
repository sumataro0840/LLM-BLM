"""
Save / load tickers and daily close prices.

Policy: prices of a ticker are fetched once for the whole period and then kept as they are
(INSERT OR IGNORE), so yfinance's later re-adjustments never change stored values.
"""
from datetime import datetime, timezone

import pandas as pd


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save_tickers(conn, sp500_table):
    """sp500_table: columns Symbol, Security, GICS Sector, GICS Sub-Industry."""
    fetched_at = _now()
    conn.executemany(
        "INSERT OR IGNORE INTO tickers (symbol, security, gics_sector, gics_sub_industry, fetched_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [(r['Symbol'], r['Security'], r['GICS Sector'], r['GICS Sub-Industry'], fetched_at)
         for _, r in sp500_table.iterrows()],
    )
    conn.commit()


def missing_tickers(conn, tickers, start, end):
    """Tickers that have no price at all between start and end (they need to be fetched)."""
    rows = conn.execute(
        f"SELECT DISTINCT symbol FROM prices WHERE date BETWEEN ? AND ? "
        f"AND symbol IN ({','.join('?' * len(tickers))})",
        [start, end, *tickers],
    ).fetchall()
    stored = {r[0] for r in rows}
    return [t for t in tickers if t not in stored]


def save_prices(conn, prices):
    """prices: rows = dates, columns = tickers (the shape returned by yfinance)."""
    fetched_at = _now()
    long = prices.rename_axis('date').reset_index().melt(id_vars='date', var_name='symbol', value_name='close').dropna()
    conn.executemany(
        "INSERT OR IGNORE INTO prices (symbol, date, close, fetched_at) VALUES (?, ?, ?, ?)",
        [(r.symbol, pd.Timestamp(r.date).strftime('%Y-%m-%d'), float(r.close), fetched_at)
         for r in long.itertuples()],
    )
    conn.commit()


def load_prices(conn, tickers, start, end):
    """Close prices between start and end, in the same shape as yfinance (rows = dates, columns = tickers)."""
    long = pd.read_sql(
        f"SELECT date, symbol, close FROM prices WHERE date BETWEEN ? AND ? "
        f"AND symbol IN ({','.join('?' * len(tickers))}) ORDER BY date",
        conn, params=[start, end, *tickers],
    )
    prices = long.pivot(index='date', columns='symbol', values='close')
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = 'Date'
    prices.columns.name = 'Ticker'
    return prices
