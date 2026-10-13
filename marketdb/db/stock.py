"""
Save / load stock indicators (stock_metrics).
"""
import pandas as pd

from marketdb.db.market import now_utc


def load_quarters(conn):
    """Every quarter of every company with its filing date (dates as Timestamps)."""
    return pd.read_sql(
        "SELECT symbol, period_end, filed_date, revenue, net_income, total_equity, long_term_debt "
        "FROM fundamentals_quarterly", conn, parse_dates=['period_end', 'filed_date'])


def save_stock_metrics(conn, as_of_date, metrics):
    """metrics: rows = symbols, columns = metric names (missing values are not stored). Replaces the date."""
    computed_at = now_utc()
    long = metrics.rename_axis(index='symbol', columns='metric').stack().dropna()
    conn.execute("DELETE FROM stock_metrics WHERE as_of_date = ?", (as_of_date,))
    conn.executemany(
        "INSERT INTO stock_metrics (as_of_date, symbol, metric, value, computed_at) VALUES (?, ?, ?, ?, ?)",
        [(as_of_date, symbol, metric, float(value), computed_at) for (symbol, metric), value in long.items()],
    )
    conn.commit()


def load_stock_metrics(conn, as_of_date, column='value'):
    """Indicators of one date (rows = symbols, columns = metrics). column: 'value' or 'sector_percentile'."""
    if column not in ('value', 'sector_percentile'):
        raise ValueError(f"column must be 'value' or 'sector_percentile': {column}")
    long = pd.read_sql(f"SELECT symbol, metric, {column} AS value FROM stock_metric_scores WHERE as_of_date = ?",
                       conn, params=(as_of_date,))
    return long.pivot(index='symbol', columns='metric', values='value')
