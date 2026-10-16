"""
OpenAI calls of the experiments: every request is logged (prompt, answer, model version, tokens, cost)
so that results can be recomputed from the log and the spending is known.

The API key is read from LLM-BLM/.env (OPENAI_API_KEY) and never written anywhere.
"""
import json
import threading
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
MODEL = 'gpt-4o-mini'                    # knowledge up to 2023-10, before every decision date
# USD per 1M tokens of gpt-4o-mini (input, cached input, output)
PRICE = {'input': 0.15, 'cached': 0.075, 'output': 0.60}
_lock = threading.Lock()


def client():
    # retries with growing waits are done by the client (at most 6), one request waits at most 60 s
    return OpenAI(api_key=dotenv_values(REPO / '.env')['OPENAI_API_KEY'], max_retries=6, timeout=60)


def cost_of(usage):
    cached = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0
    return ((usage.prompt_tokens - cached) * PRICE['input'] + cached * PRICE['cached']
            + usage.completion_tokens * PRICE['output']) / 1e6


def chat(api, messages, log_path, meta, **kwargs):
    """One chat completion; the request and the answer are appended to log_path (JSON lines)."""
    started = time.time()
    response = api.chat.completions.create(model=MODEL, messages=messages, **kwargs)
    usage = response.usage
    record = {
        **meta, 'model': response.model, 'requested_model': MODEL, 'params': kwargs, 'messages': messages,
        'answers': [c.message.content for c in response.choices],
        'prompt_tokens': usage.prompt_tokens, 'completion_tokens': usage.completion_tokens,
        'cost_usd': cost_of(usage), 'seconds': round(time.time() - started, 2),
        'at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    if kwargs.get('logprobs'):
        record['top_logprobs'] = [{t.token: t.logprob for t in pos.top_logprobs}
                                  for pos in response.choices[0].logprobs.content]
    with _lock, open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record


def done_keys(log_path, key):
    """Keys of the requests already answered (an interrupted run continues without asking them again)."""
    path = Path(log_path)
    if not path.exists():
        return set()
    with open(path, encoding='utf-8') as f:
        return {tuple(json.loads(line)[k] for k in key) for line in f if line.strip()}


def load_log(log_path):
    with open(log_path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def spending(log_path):
    rows = load_log(log_path)
    return {'requests': len(rows), 'prompt_tokens': sum(r['prompt_tokens'] for r in rows),
            'completion_tokens': sum(r['completion_tokens'] for r in rows),
            'cost_usd': round(sum(r['cost_usd'] for r in rows), 4),
            'models': sorted({r['model'] for r in rows})}
