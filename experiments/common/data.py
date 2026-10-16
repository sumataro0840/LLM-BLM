"""
Everything an experiment needs for one decision date, built once from the database and cached.

Only data up to the decision date goes into the inputs (blm.inputs.load_inputs is reused as it is);
the outcomes (next month's returns) are kept in separate fields whose names start with 'next_'.
"""
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values

from blm.covariance import shrunk_covariance
from blm.inputs import load_inputs
from marketdb.db.market import load_prices, load_ticker_table

REPO = Path(__file__).resolve().parents[2]
ENV = dotenv_values(REPO / '.env')
DB_PATH = REPO / ENV['DB_PATH']
CACHE = REPO / 'data' / 'cache' / 'dates'
LLM_RUN_ID = 1                 # gpt-4o-mini, the answers every report is based on
COV_WINDOW_DAYS = 252          # same as BLM_COV_WINDOW_DAYS
MIN_HISTORY_RATIO = 0.5        # same as BLM_MIN_HISTORY_RATIO
TRADING_DAYS = 252
CACHE_VERSION = 2


def connect():
    """Read-only connection: the experiments never write to the project's database."""
    return sqlite3.connect(f'file:{DB_PATH.as_posix()}?mode=ro', uri=True)


def decision_dates(conn, run_id=LLM_RUN_ID):
    rows = conn.execute("SELECT DISTINCT as_of_date FROM llm_views WHERE run_id = ? ORDER BY as_of_date", (run_id,))
    return [r[0] for r in rows]


def holding_end(conn, as_of_date):
    """Last trading day of the month after the decision date (the same rule as the view portfolio_returns)."""
    return conn.execute(
        "SELECT MAX(date) FROM prices WHERE symbol = '^GSPC' AND date <= date(?, 'start of month', '+2 months', '-1 day')",
        (as_of_date,)).fetchone()[0]


def next_trading_day(conn, date):
    return conn.execute("SELECT MIN(date) FROM prices WHERE symbol = '^GSPC' AND date > ?", (date,)).fetchone()[0]


def _build(conn, as_of_date):
    inputs = load_inputs(conn, as_of_date, LLM_RUN_ID, COV_WINDOW_DAYS, MIN_HISTORY_RATIO)
    universe = list(inputs.market_weights.index)
    covariance = shrunk_covariance(inputs.returns)

    # betas against the cap-weighted universe over the covariance window
    market = inputs.returns @ inputs.market_weights
    beta = inputs.returns.apply(lambda col: col.cov(market)) / market.var()

    # what the LLM saw: the daily returns of the decision month
    month = inputs.month_returns

    end = holding_end(conn, as_of_date)
    entry_next = next_trading_day(conn, as_of_date)
    prices = load_prices(conn, universe, as_of_date, end)
    start_price = prices.loc[pd.Timestamp(as_of_date)]
    last_price = prices.ffill().iloc[-1]
    next_price = prices.loc[pd.Timestamp(entry_next)] if pd.Timestamp(entry_next) in prices.index else pd.Series(np.nan, index=universe)
    daily = prices.pct_change().iloc[1:]
    rf_annual = float(inputs.risk_free_daily * TRADING_DAYS)
    n_days = len(daily)

    tickers = load_ticker_table(conn).set_index('Symbol')
    return {
        'version': CACHE_VERSION,
        'as_of_date': as_of_date,
        'universe': universe,
        'sector': tickers['GICS Sector'].reindex(universe),
        'covariance': covariance,
        'market_weights': inputs.market_weights,
        'beta': beta,
        'view_mean': inputs.view_mean,            # LLM forecast of the average daily return (30 answers)
        'view_variance': inputs.view_variance,    # variance of the 30 answers
        'risk_free_daily': inputs.risk_free_daily,
        'input_mean': month.mean(),               # average daily return of the decision month
        'input_sd': month.std(),
        'input_days': month.notna().sum(),
        'vol_daily': inputs.returns.std(),        # daily volatility over the covariance window
        'month_returns': month,
        # outcomes (after the decision date)
        'next_end_date': end,
        'next_return': last_price / start_price - 1,                     # bought at the decision date's close
        'next_return_lagged': last_price / next_price - 1,               # bought one trading day later
        'next_mean_daily': daily.mean(),
        'next_days': daily.notna().sum(),
        'next_daily': daily,
        'next_risk_free': rf_annual * n_days / TRADING_DAYS,
    }


def load_date(conn, as_of_date, rebuild=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{as_of_date}.pkl'
    if path.exists() and not rebuild:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if data.get('version') == CACHE_VERSION:
            return data
    data = _build(conn, as_of_date)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return data


def load_all(conn=None, rebuild=False):
    """{decision date: data} for every decision date of the LLM run, oldest first."""
    conn = conn or connect()
    return {d: load_date(conn, d, rebuild) for d in decision_dates(conn)}


def forecast_panel(all_data):
    """Long table of (date, symbol, forecast, outcome) for calibration and IC.

    outcome = average daily return of the next month (as in the reports), kept when 15 days or more exist.
    """
    rows = []
    for d, x in all_data.items():
        df = pd.DataFrame({'date': d, 'symbol': x['universe'], 'Q': x['view_mean'].values,
                           'variance': x['view_variance'].values,
                           'r_in': x['input_mean'].reindex(x['universe']).values,
                           'r_next': x['next_mean_daily'].reindex(x['universe']).values,
                           'n_next': x['next_days'].reindex(x['universe']).values,
                           'R_next': x['next_return'].reindex(x['universe']).values})
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
