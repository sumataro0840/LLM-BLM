"""
Data settings (defaults reproduce the full experiment).
"""

# S&P500 constituents
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia rejects requests without a User-Agent (HTTP 403)
USER_AGENT = "Mozilla/5.0 (LLM-BLM research script)"

# Months to process (first day of each month)
START_MONTH = "2024-06-01"
END_MONTH = "2024-12-01"

# Base directory for yfinance/ and responses/
OUT_DIR = "."
