"""
Data settings (read from .env).
"""
import os

# S&P500 constituents
SP500_URL = os.environ["SP500_URL"]
# S&P500 constituent changes
SP500_HISTORY_URL = os.environ["SP500_HISTORY_URL"]
# Wikipedia rejects requests without a User-Agent (HTTP 403)
USER_AGENT = os.environ["USER_AGENT"]

# Months to process (first day of each month)
START_MONTH = os.environ["START_MONTH"]
END_MONTH = os.environ["END_MONTH"]

# Comma-separated tickers, or ALL for every S&P500 stock ([] means all)
_tickers = os.environ["TICKERS"]
TICKERS = [] if _tickers.upper() == "ALL" else [t.strip() for t in _tickers.split(",")]

# Base directory for yfinance/ and responses/
OUT_DIR = os.environ["OUT_DIR"]

# SQLite database file
DB_PATH = os.environ["DB_PATH"]

# Period stored in the database (scripts/build_db.py)
DB_START_DATE = os.environ["DB_START_DATE"]
DB_END_DATE = os.environ["DB_END_DATE"]
