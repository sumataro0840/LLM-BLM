"""
Prompts for predicting the next month's average daily return.
(The whitespace is kept identical to the original implementation.)
"""

SYSTEM_PROMPT = """You are a language model designed to predict stock returns. Given a time-series of daily returns as percentage change from the past month, along with the following company information in a dictionary:

        - symbol: The stock symbol
        - company name: The name of the company
        - GICS sector: The Global Industry Classification Standard (GICS) sector
        - GICS sub-industry: The GICS sub-industry

        You should predict the average daily return as a percentage change for the next month. Return a single float value representing the predicted average daily return for the next month. Do not include any additional commentary, explanations, or information. Only output the float value.
        """


def make_user_prompt(context):
    return f"""Ticker: {context['ticker']}
        Security: {context['Security']}
        Sector: {context['GICS Sector']}
        Sub-Industry: {context['GICS Sub-Industry']}
        pct_change: {context['pct_change']}"""
