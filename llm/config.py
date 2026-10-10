"""
Settings of the LLM run: API / model / sampling, months and tickers
(read from LLM-BLM/.env, every value is expected to be set there).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# LLM-BLM/.env (found regardless of the current directory)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ["OPENAI_MODEL"]

# Number of answers per ticker (the variance is used as the confidence of the view)
N_SAMPLES = int(os.environ["N_SAMPLES"])
# Sampling temperature (must stay above 0, otherwise the answers do not vary)
TEMPERATURE = float(os.environ["TEMPERATURE"])
# Tickers asked at the same time
MAX_WORKERS = int(os.environ["MAX_WORKERS"])

# Empty = start a new run, a number = continue that run (answered tickers are skipped)
RUN_ID = int(os.environ["RUN_ID"]) if os.environ["RUN_ID"] else None
# Free memo stored with a new run
RUN_NOTE = os.environ["RUN_NOTE"]

# Months given to the LLM (first day of each month); the prediction is for the month after each
START_MONTH = os.environ["START_MONTH"]
END_MONTH = os.environ["END_MONTH"]

# Comma-separated tickers, or ALL for every S&P500 stock ([] means all)
_tickers = os.environ["TICKERS"]
TICKERS = [] if _tickers.upper() == "ALL" else [t.strip() for t in _tickers.split(",")]

# SQLite database file (input prices and the answers)
DB_PATH = os.environ["DB_PATH"]
