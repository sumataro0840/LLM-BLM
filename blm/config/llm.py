"""
LLM settings (read from .env).
"""
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ["OPENAI_MODEL"]

# Number of LLM queries per ticker (the variance is used as the confidence of the view)
N_SAMPLES = int(os.environ["N_SAMPLES"])
