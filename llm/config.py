"""
Settings of the LLM run: API / model / sampling, months, tickers and output
(read from LLM-BLM/.env, every value is expected to be set there).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# LLM-BLM/.env (found regardless of the current directory)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ["OPENAI_MODEL"]

# Number of LLM queries per ticker (the variance is used as the confidence of the view)
N_SAMPLES = int(os.environ["N_SAMPLES"])

# Months to process (first day of each month)
START_MONTH = os.environ["START_MONTH"]
END_MONTH = os.environ["END_MONTH"]

# Comma-separated tickers, or ALL for every S&P500 stock ([] means all)
_tickers = os.environ["TICKERS"]
TICKERS = [] if _tickers.upper() == "ALL" else [t.strip() for t in _tickers.split(",")]

# Base directory for yfinance/ and responses/
OUT_DIR = os.environ["OUT_DIR"]
