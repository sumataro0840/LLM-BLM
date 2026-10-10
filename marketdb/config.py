"""
Settings of the market database: data sources, database file and stored period
(read from LLM-BLM/.env, every value is expected to be set there).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# LLM-BLM/.env (found regardless of the current directory)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# S&P500 constituents
SP500_URL = os.environ["SP500_URL"]
# S&P500 constituent changes
SP500_HISTORY_URL = os.environ["SP500_HISTORY_URL"]
# Old revisions of the constituents page ({revid} = revision id)
WIKIPEDIA_API_URL = os.environ["WIKIPEDIA_API_URL"]
WIKIPEDIA_REVISION_URL = os.environ["WIKIPEDIA_REVISION_URL"]
# Wikipedia rejects requests without a User-Agent (HTTP 403)
USER_AGENT = os.environ["USER_AGENT"]

# SQLite database file
DB_PATH = os.environ["DB_PATH"]

# Period stored in the database (marketdb/build_db.py)
DB_START_DATE = os.environ["DB_START_DATE"]
DB_END_DATE = os.environ["DB_END_DATE"]

# Financial statements (marketdb/build_fundamentals.py)
EDGAR_FACTS_URL = os.environ["EDGAR_FACTS_URL"]              # {cik} = 10-digit company id
EDGAR_SUBMISSIONS_URL = os.environ["EDGAR_SUBMISSIONS_URL"]
WIKIDATA_SPARQL_URL = os.environ["WIKIDATA_SPARQL_URL"]
FUNDAMENTALS_START_DATE = os.environ["FUNDAMENTALS_START_DATE"]
