"""
Settings of the report analysis: API / model, report folder, reports to analyse
(read from LLM-BLM/.env, every value is expected to be set there).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# LLM-BLM/.env (found regardless of the current directory)
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ["OPENAI_MODEL"]

# Folder of the report files, named <source>_<YYYY-MM>.pdf (relative to LLM-BLM)
REPORTS_DIR = ROOT / os.environ["REPORTS_DIR"]

# Reports to analyse: months on the cover (YYYY-MM), both inclusive
REPORT_START_MONTH = os.environ["REPORT_START_MONTH"]
REPORT_END_MONTH = os.environ["REPORT_END_MONTH"]
# Pages sent to the LLM, counted from the first page
REPORT_MAX_PAGES = int(os.environ["REPORT_MAX_PAGES"])
# 0 = the same report gives (almost) the same answer every time
REPORT_TEMPERATURE = float(os.environ["REPORT_TEMPERATURE"])
# Empty = start a new run, a number = continue that run (analysed reports are skipped)
REPORT_RUN_ID = int(os.environ["REPORT_RUN_ID"]) if os.environ["REPORT_RUN_ID"] else None
# Free memo stored with a new run
REPORT_RUN_NOTE = os.environ["REPORT_RUN_NOTE"]

# SQLite database file
DB_PATH = os.environ["DB_PATH"]
