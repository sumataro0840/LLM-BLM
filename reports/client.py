"""
OpenAI API call that reads one report (the PDF pages are sent as they are, so charts and tables are seen too).
"""
import base64
import json

from openai import OpenAI


def make_client(api_key):
    # rate limits and timeouts are retried by the client; a report takes longer than a short question
    return OpenAI(api_key=api_key, max_retries=6, timeout=300)


def analyse_report(client, model, system_prompt, user_prompt, schema, file_name, pdf_bytes, temperature):
    """Returns (values, model_version, input_tokens): values = {item: number, text or None}."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "file", "file": {
                    "filename": file_name,
                    "file_data": "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode(),
                }},
                {"type": "text", "text": user_prompt},
            ]},
        ],
        temperature=temperature,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "report_items", "strict": True, "schema": schema}},
    )
    return json.loads(completion.choices[0].message.content), completion.model, completion.usage.prompt_tokens
