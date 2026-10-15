"""
Save / load sector indicators (sector_metrics).
"""
import pandas as pd

from marketdb.db.market import now_utc


def load_sector_etfs(conn):
    """ETF symbol -> the sector it tracks."""
    rows = conn.execute("SELECT symbol, gics_sector FROM tickers WHERE gics_sector IS NOT NULL "
                        "AND symbol NOT IN (SELECT symbol FROM index_membership)")
    return dict(rows.fetchall())


def load_month_ends(conn):
    """Last trading day of every month in the database."""
    rows = conn.execute("SELECT MAX(date) FROM prices WHERE symbol = '^GSPC' GROUP BY substr(date, 1, 7) ORDER BY 1")
    return [r[0] for r in rows]


def load_member_caps(conn, date):
    """Market cap, sector and sub-industry of the index members on `date` (indexed by symbol)."""
    return pd.read_sql(
        "SELECT w.symbol, w.market_cap, t.gics_sector, t.gics_sub_industry "
        "FROM index_weights w JOIN tickers t ON t.symbol = w.symbol "
        "WHERE w.date = ?", conn, params=(date,), index_col='symbol')


def save_sector_metrics(conn, as_of_date, metrics):
    """metrics: rows = sectors, columns = metric names (missing values are not stored). Replaces the date."""
    computed_at = now_utc()
    long = metrics.rename_axis(index='gics_sector', columns='metric').stack().dropna()
    conn.execute("DELETE FROM sector_metrics WHERE as_of_date = ?", (as_of_date,))
    conn.executemany(
        "INSERT INTO sector_metrics (as_of_date, gics_sector, metric, value, computed_at) VALUES (?, ?, ?, ?, ?)",
        [(as_of_date, sector, metric, float(value), computed_at) for (sector, metric), value in long.items()],
    )
    conn.commit()


def load_sector_metrics(conn, as_of_date, column='value'):
    """Indicators of one date (rows = sectors, columns = metrics). column: 'value' or 'z_score'."""
    if column not in ('value', 'z_score'):
        raise ValueError(f"column must be 'value' or 'z_score': {column}")
    long = pd.read_sql(f"SELECT gics_sector, metric, {column} AS value FROM sector_metric_scores WHERE as_of_date = ?",
                       conn, params=(as_of_date,))
    return long.pivot(index='gics_sector', columns='metric', values='value')
