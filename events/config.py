"""
Settings of the event study (read from LLM-BLM/.env; the EVENTS_* values have defaults).

SEC_USER_AGENT (a name and a contact e-mail, required by the SEC) is read from the environment only,
so that it is never written to a file of the repository.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")

DB_PATH = REPO / os.environ["DB_PATH"]
# Release texts (outside Git; the same place as H06 used)
TEXT_DIR = REPO / os.environ.get("EVENTS_TEXT_DIR", "data/cache/edgar")
# Loughran-McDonald master dictionary (free for academic research; outside Git). The 1993-2025 file of the
# authors' site (Google Drive) could not be downloaded on 2026-10-17 (quota exceeded); this is an earlier release of
# the same master dictionary, as shipped in the pysentiment2 0.1.1 package (same columns, words flagged by the year
# they were added)
LM_DICTIONARY = REPO / os.environ.get("EVENTS_LM_DICTIONARY", "data/cache/lm/LM_MasterDictionary_pysentiment2-0.1.1.csv")
# ChronoGPT-Instruct (code and weights outside the repository, see 工房/LLM/chrono/README.md)
CHRONO_DIR = (REPO / os.environ.get("EVENTS_CHRONO_DIR", "../LLM/chrono")).resolve()
# FinBERT of Huang, Wang and Yang (2023)
FINBERT_MODEL = os.environ.get("EVENTS_FINBERT_MODEL", "yiyanghkust/finbert-tone")
# LLM: gpt-4o-mini knows nothing after 2023-10, so it only reads releases filed from this day on
OPENAI_MODEL = os.environ.get("EVENTS_OPENAI_MODEL", "gpt-4o-mini")
LLM_FIRST_DATE = os.environ.get("EVENTS_LLM_FIRST_DATE", "2023-11-01")

SEC_MAX_PER_SECOND = 8      # SEC asks for at most 10


def sec_user_agent():
    agent = os.environ.get("SEC_USER_AGENT")
    if not agent:
        raise SystemExit('SEC_USER_AGENT="<name> <contact e-mail>" を環境変数で渡してください (SEC の求め)')
    return agent


def openai_api_key():
    return os.environ["OPENAI_API_KEY"]
