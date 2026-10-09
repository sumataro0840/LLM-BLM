"""
LLM settings (API key and model are read from .env).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# LLM-BLM/.env (found regardless of the current directory)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Number of LLM queries per ticker (the variance is used as the confidence of the view)
N_SAMPLES = 30
