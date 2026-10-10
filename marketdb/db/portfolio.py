"""
Save / load portfolios (portfolio_runs / portfolio_weights).
"""
import json

import pandas as pd

from marketdb.db.market import now_utc


def create_portfolio(conn, strategy, llm_run_id, params, note):
    """params: dict of the coefficients used (stored as JSON)."""
    cursor = conn.execute(
        "INSERT INTO portfolio_runs (strategy, llm_run_id, params, note, created_at) VALUES (?, ?, ?, ?, ?)",
        (strategy, llm_run_id, json.dumps(params, sort_keys=True), note, now_utc()),
    )
    conn.commit()
    return cursor.lastrowid


def save_weights(conn, portfolio_id, as_of_date, weights):
    """weights: Series indexed by symbol (every stock of the universe, 0 included)."""
    conn.executemany(
        "INSERT INTO portfolio_weights (portfolio_id, as_of_date, symbol, weight) VALUES (?, ?, ?, ?)",
        [(portfolio_id, as_of_date, symbol, float(weight)) for symbol, weight in weights.items()],
    )
    conn.commit()


def load_portfolio_returns(conn, portfolio_ids):
    """Monthly returns, rows = decision dates, columns = portfolio ids."""
    long = pd.read_sql(
        f"SELECT portfolio_id, as_of_date, period_return FROM portfolio_returns "
        f"WHERE portfolio_id IN ({','.join('?' * len(portfolio_ids))}) ORDER BY as_of_date",
        conn, params=list(portfolio_ids),
    )
    return long.pivot(index='as_of_date', columns='portfolio_id', values='period_return')
