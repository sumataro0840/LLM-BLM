"""
OpenAI API calls with structured output.
"""
import json

from openai import OpenAI
from pydantic import BaseModel


class ResearchPaperExtraction(BaseModel):
    expected_return: float


def make_client(api_key):
    # rate limits and timeouts are retried by the client (with increasing waits)
    return OpenAI(api_key=api_key, max_retries=6)


def sample_expected_returns(client, model, system_prompt, user_prompt, n, temperature):
    """Ask the question once and take n independent answers.

    Returns (answers, model_version): answers has n values, None where the answer could not be parsed.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # n answers are sampled independently from one request (the input is sent and billed once)
        n=n,
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "expected_return",
                "strict": True,
                "schema": {
                    **ResearchPaperExtraction.model_json_schema(),
                    "additionalProperties": False,
                },
            },
        },
    )
    answers = []
    for choice in completion.choices:
        try:
            answers.append(float(json.loads(choice.message.content)['expected_return']))
        except:
            answers.append(None)
    return answers, completion.model
