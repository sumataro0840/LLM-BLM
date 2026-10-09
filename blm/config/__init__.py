"""
Settings read from LLM-BLM/.env (every value is expected to be set there).
llm.py: API / model / sampling, data.py: data source / period / tickers / output.
"""
from pathlib import Path

from dotenv import load_dotenv

# LLM-BLM/.env (found regardless of the current directory)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
