"""
Save / load tickers and daily close prices.

Policy: prices of a ticker are fetched once for the whole period and then kept as they are
(INSERT OR IGNORE), so yfinance's later re-adjustments never change stored values.
"""
from datetime import datetime, timezone

import pandas as pd


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def save_tickers(conn, tickers):
    """tickers: rows of (symbol, security, gics_sector, gics_sub_industry); GICS may be None."""
    fetched_at = now_utc()
    conn.executemany(
        "INSERT OR IGNORE INTO tickers (symbol, security, gics_sector, gics_sub_industry, fetched_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [(*row, fetched_at) for row in tickers],
    )
    conn.commit()


def mark_excluded(conn, symbols, reason):
    conn.executemany("UPDATE tickers SET excluded_reason = ? WHERE symbol = ?", [(reason, s) for s in symbols])
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
    """prices: the frame returned by blm.data.prices.download_prices (columns: ('Close' | 'Adj Close', ticker))."""
    fetched_at = now_utc()
    long = prices.stack(level=1, future_stack=True).rename_axis(['date', 'symbol']).reset_index().dropna()
    long['date'] = pd.to_datetime(long['date']).dt.strftime('%Y-%m-%d')
    long = long.rename(columns={'Close': 'close', 'Adj Close': 'adj_close'})
    conn.executemany(
        "INSERT OR IGNORE INTO prices (symbol, date, close, adj_close, fetched_at) VALUES (?, ?, ?, ?, ?)",
        [(r.symbol, r.date, float(r.close), float(r.adj_close), fetched_at)
         for r in long.itertuples(index=False)],
    )
    conn.commit()


def save_shares(conn, symbol, shares):
    """shares: Series indexed by report date (already on the current split basis)."""
    fetched_at = now_utc()
    conn.executemany(
        "INSERT OR IGNORE INTO shares_outstanding (symbol, date, shares, fetched_at) VALUES (?, ?, ?, ?)",
        [(symbol, d, int(v), fetched_at) for d, v in shares.items()],
    )
    conn.commit()


def load_prices(conn, tickers, start, end, column='adj_close'):
    """Prices between start and end, in the same shape as yfinance (rows = dates, columns = tickers).

    column: 'adj_close' (for returns) or 'close' (for market cap).
    """
    if column not in ('adj_close', 'close'):
        raise ValueError(f"column must be 'adj_close' or 'close': {column}")
    long = pd.read_sql(
        f"SELECT date, symbol, {column} AS value FROM prices WHERE date BETWEEN ? AND ? "
        f"AND symbol IN ({','.join('?' * len(tickers))}) ORDER BY date",
        conn, params=[start, end, *tickers],
    )
    prices = long.pivot(index='date', columns='symbol', values='value')
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = 'Date'
    prices.columns.name = 'Ticker'
    return prices
