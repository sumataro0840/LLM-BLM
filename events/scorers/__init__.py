"""
Ways of scoring the release texts. Every scorer has the same shape (設計.md, 8):

    method         name stored in release_score_runs.method
    model          model or resource
    uses_outcomes  True when it is trained on stock reactions (run only after the pre-registration)
    params()       everything needed to reproduce the scores (stored as JSON, its hash identifies the run)
    score(releases) yields one Result per release (releases: rows of earnings_releases with a text)
"""
from dataclasses import dataclass, field
from importlib import import_module


@dataclass
class Result:
    accession: str
    values: dict                     # {item: number | str | list | None}
    call: dict = field(default=None)  # model_version, raw_answer, tokens, cost, seconds (None = no model call)


class Scorer:
    method = None
    model = None
    uses_outcomes = False

    def params(self):
        raise NotImplementedError

    def score(self, releases):
        raise NotImplementedError


MODULES = {"llm": "llm.LLMScorer", "dictionary": "dictionary.DictionaryScorer", "finbert": "finbert.FinBERTScorer",
           "chrono": "chrono.ChronoScorer", "tfidf": "tfidf.TfidfScorer"}


def make(method, **kwargs):
    module, name = MODULES[method].split(".")
    return getattr(import_module(f"events.scorers.{module}"), name)(**kwargs)
