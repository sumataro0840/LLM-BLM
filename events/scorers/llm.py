"""
gpt-4o-mini reads the release text and scores guidance, outlook and tone (the same question as H06, unchanged,
so that the 5,429 scores of H06 and new ones are on one scale; the block between the marks below is copied from
experiments/H06_earnings_text/score.py and tests/test_events_db.py checks that it stays identical).

gpt-4o-mini knows nothing after 2023-10: releases filed before config.LLM_FIRST_DATE are refused.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from events import config, texts
from events.scorers import Result, Scorer

# USD per 1M tokens of gpt-4o-mini (input, cached input, output)
PRICE = {"input": 0.15, "cached": 0.075, "output": 0.60}

# --- copied from experiments/H06_earnings_text/score.py ---
MAX_CHARS = 15000

SYSTEM = (
    "You read one quarterly earnings press release of a US company and score what it says. "
    "Use only the text of the release. Do not use anything you know about the company from elsewhere, "
    "about later events, or about how the stock price moved. Judge the substance, not the usual promotional wording "
    "that every release uses.")
QUESTION = """Score the release:
- guidance_change: the company's forward guidance compared with its previous guidance, as the release states it. -2 = cut sharply, -1 = lowered, 0 = reaffirmed or unchanged, 1 = raised, 2 = raised sharply. null when the release gives no forward guidance or does not say how it compares.
- outlook: what the release says about the business over the coming quarters (demand, orders, pipeline, margins). -2 = clearly deteriorating, -1 = somewhat weaker, 0 = stable or not discussed, 1 = somewhat stronger, 2 = clearly improving.
- tone: management's overall tone about the quarter and the business. -2 = clearly negative, -1 = cautious, 0 = neutral, 1 = positive, 2 = clearly positive.
- risk_flags: the problems the release itself reports (empty when none).

The release (it may be cut off):
"""
SCHEMA = {
    'name': 'earnings_release_scores', 'strict': True,
    'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['guidance_change', 'outlook', 'tone', 'risk_flags'],
        'properties': {
            'guidance_change': {'anyOf': [{'type': 'integer', 'enum': [-2, -1, 0, 1, 2]}, {'type': 'null'}]},
            'outlook': {'type': 'integer', 'enum': [-2, -1, 0, 1, 2]},
            'tone': {'type': 'integer', 'enum': [-2, -1, 0, 1, 2]},
            'risk_flags': {'type': 'array', 'items': {'type': 'string', 'enum': [
                'demand_weakness', 'margin_pressure', 'cost_inflation', 'restructuring_or_layoffs', 'impairment_or_writedown',
                'legal_or_regulatory', 'supply_chain', 'currency_or_macro', 'debt_or_liquidity', 'management_change']}},
        },
    },
}
PARAMS = {'temperature': 0, 'max_tokens': 120, 'response_format': {'type': 'json_schema', 'json_schema': SCHEMA}}
# --- end of the copy ---
ITEMS = ["guidance_change", "outlook", "tone", "risk_flags"]
EXPECTED_VERSION = "gpt-4o-mini-2024-07-18"    # the version that answered all of H06


def cost_of(usage):
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return ((usage.prompt_tokens - cached) * PRICE["input"] + cached * PRICE["cached"]
            + usage.completion_tokens * PRICE["output"]) / 1e6


class LLMScorer(Scorer):
    method = "llm"
    uses_outcomes = False

    def __init__(self, workers=8):
        self.model = config.OPENAI_MODEL
        self.workers = workers

    def params(self):
        return {"model": self.model, "system": SYSTEM, "question": QUESTION, "request": PARAMS, "max_chars": MAX_CHARS}

    def score(self, releases):
        early = releases[releases.filing_date < config.LLM_FIRST_DATE]
        if len(early):
            raise ValueError(f"{len(early)} releases were filed before {config.LLM_FIRST_DATE}; "
                             f"{self.model} may know what happened after them")
        api = OpenAI(api_key=config.openai_api_key(), max_retries=6, timeout=60)

        def one(r):
            text = texts.read(r.text_path, MAX_CHARS)
            started = time.time()
            response = api.chat.completions.create(
                model=self.model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION + text}],
                **PARAMS)
            if response.model != EXPECTED_VERSION:
                # 'gpt-4o-mini' may point to a newer snapshot some day; its scores would not be on H06's scale
                raise RuntimeError(f"answered by {response.model}, not {EXPECTED_VERSION}")
            answer = response.choices[0].message.content
            values = json.loads(answer)
            call = {"model_version": response.model, "raw_answer": answer, "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens, "cost_usd": cost_of(response.usage),
                    "seconds": round(time.time() - started, 2)}
            return Result(r.accession, {k: values.get(k) for k in ITEMS}, call)

        with ThreadPoolExecutor(self.workers) as pool:
            yield from pool.map(one, releases.itertuples())
