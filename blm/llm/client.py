"""
OpenAI API calls with structured output.
"""
import json

from openai import OpenAI
from pydantic import BaseModel


class ResearchPaperExtraction(BaseModel):
    expected_return: float


def make_client(api_key):
    return OpenAI(api_key=api_key)


def sample_expected_returns(client, model, system_prompt, user_prompt, n):
    """Ask the same question n times and return the predicted values (failed parses are skipped)."""
    answers = []
    for _ in range(n):
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
        try:
            expected_return_dict = json.loads(completion.choices[0].message.content)
            print(expected_return_dict)
        except:
            continue
        answers.append(expected_return_dict['expected_return'])
    return answers
