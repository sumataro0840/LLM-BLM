"""
Prompt and answer format for reading a monthly report, built from reports/items.py.
"""
from reports.items import ITEMS

SYSTEM_PROMPT = (
    "You read a monthly market outlook written in Japanese by an asset manager and fill in the requested items. "
    "Use only what the attached pages say: do not add anything you know from elsewhere. "
    "Copy figures exactly as printed. "
    "Answer null for an item the pages do not state; never guess. "
    "For sector scores, answer null for a sector the text does not discuss."
)

USER_PROMPT = ("Fill in the items from the attached report. It is the issue of {month}: "
               "values of {year} and later are forecasts, and the tables have one column per year.")

_JSON_TYPE = {'number': 'number', 'score': 'integer', 'text': 'string'}


def make_user_prompt(report_month):
    return USER_PROMPT.format(month=report_month, year=report_month[:4])


def answer_schema(report_month):
    """JSON schema of the answer: one property per item, null allowed everywhere.
    The years in the item descriptions are filled in from the month of the report ('YYYY-MM')."""
    year = int(report_month[:4])
    properties = {}
    for item in ITEMS:
        properties[item.name] = {'type': [_JSON_TYPE[item.type], 'null'],
                                 'description': item.description.format(year=year, next_year=year + 1)}
        if item.type == 'score':
            properties[item.name]['enum'] = [-2, -1, 0, 1, 2, None]
    return {'type': 'object', 'properties': properties, 'required': [item.name for item in ITEMS],
            'additionalProperties': False}
