"""
Loughran-McDonald tone: shares of positive, negative and uncertainty words in the release.

Each word of the dictionary carries the year it was added to a list; a word added after the year of the release
is not counted (the dictionary itself was built from later filings, so this keeps it closer to point in time).
"""
import re

import pandas as pd

from events import config, texts
from events.scorers import Result, Scorer

CATEGORIES = ["Positive", "Negative", "Uncertainty"]
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def load_lists(path):
    """{category: {word: year added}} (a negative year marks a removed word, which is left out)."""
    df = pd.read_csv(path, usecols=["Word", *CATEGORIES], keep_default_na=False)
    return {c: {w.upper(): int(y) for w, y in zip(df.Word, df[c]) if int(y) > 0} for c in CATEGORIES}


def tone(text, lists, year):
    words = [w.upper() for w in WORD.findall(text)]
    n = len(words)
    counts = {}
    for c, added in lists.items():
        counts[c] = sum(1 for w in words if w in added and added[w] <= year)
    pos, neg = counts["Positive"], counts["Negative"]
    return {"positive": pos / n if n else None, "negative": neg / n if n else None,
            "uncertainty": counts["Uncertainty"] / n if n else None,
            "net_tone": (pos - neg) / (pos + neg) if pos + neg else 0.0, "n_words": n}


class DictionaryScorer(Scorer):
    method = "dictionary"
    uses_outcomes = False

    def __init__(self, path=None):
        self.path = path or config.LM_DICTIONARY
        self.model = "LM-MasterDictionary (pysentiment2 0.1.1)"
        self.sha256 = texts.sha256_of(self.path)
        self.lists = load_lists(self.path)

    def params(self):
        return {"dictionary_file": self.path.name, "dictionary_sha256": self.sha256, "categories": CATEGORIES,
                "tokens": WORD.pattern, "point_in_time": "words added after the year of the release are not counted"}

    def score(self, releases):
        for r in releases.itertuples():
            yield Result(r.accession, tone(texts.read(r.text_path), self.lists, int(r.accepted_et[:4])))
