"""
Settings of the portfolio calculation (read from LLM-BLM/.env, every value is expected to be set there).
The coefficients used are also stored with each portfolio (portfolio_runs.params).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# LLM-BLM/.env (found regardless of the current directory)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# SQLite database file (prices, LLM answers and the portfolios)
DB_PATH = os.environ["DB_PATH"]

# LLM run whose answers are used as views; the portfolios are computed for its decision dates
LLM_RUN_ID = int(os.environ["BLM_LLM_RUN_ID"])
# Comma-separated strategies (names in blm/strategies.py)
STRATEGIES = [s.strip() for s in os.environ["BLM_STRATEGIES"].split(",")]
# Free memo stored with the portfolios
NOTE = os.environ["BLM_NOTE"]

# Weight of the views against the market equilibrium (uncertainty of the equilibrium = tau x covariance)
TAU = float(os.environ["BLM_TAU"])
# Risk aversion delta: equilibrium return = delta x covariance x market weights, objective = return - delta / 2 x variance
RISK_AVERSION = float(os.environ["BLM_RISK_AVERSION"])
# Trading days of daily returns used for the covariance
COV_WINDOW_DAYS = int(os.environ["BLM_COV_WINDOW_DAYS"])
# A stock needs prices for at least this share of the window to be in the universe
MIN_HISTORY_RATIO = float(os.environ["BLM_MIN_HISTORY_RATIO"])

# Coefficients of the original implementation (strategy 'black_litterman_original')
ORIGINAL_RISK_FREE_RATE = float(os.environ["BLM_ORIGINAL_RISK_FREE_RATE"])   # subtracted from daily returns as it is
ORIGINAL_RETURN_WEIGHT = float(os.environ["BLM_ORIGINAL_RETURN_WEIGHT"])     # objective = variance - weight x return
