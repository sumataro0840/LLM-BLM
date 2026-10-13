"""
Settings of the report analysis: report folder and database file
(read from LLM-BLM/.env, every value is expected to be set there).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# LLM-BLM/.env (found regardless of the current directory)
load_dotenv(ROOT / ".env")

# Folder of the report files, named <source>_<YYYY-MM>.pdf (relative to LLM-BLM)
REPORTS_DIR = ROOT / os.environ["REPORTS_DIR"]

# SQLite database file
DB_PATH = os.environ["DB_PATH"]
