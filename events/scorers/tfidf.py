"""
TF-IDF + ridge: a text model trained on the stock reactions of earlier releases (the non-LLM text baseline of
Meursault et al. 2023's kind). Uses outcomes, so it runs only after the pre-registration (設計.md, 9.2).

For the releases of each calendar quarter, the model is fit on the releases whose reaction window had closed
before the quarter began (expanding window), then predicts the reaction from the text.
The vocabulary size and the ridge penalty are fixed here and are not tuned.
"""
import pandas as pd

from events import texts
from events.scorers import Result, Scorer

MAX_FEATURES = 20000
MIN_DF = 5
ALPHA = 10.0
MIN_TRAIN = 500
MAX_CHARS = 15000


class TfidfScorer(Scorer):
    method = "tfidf"
    model = "sklearn TfidfVectorizer + Ridge"
    uses_outcomes = True

    def __init__(self, reactions=None, target="abn_r_0_1"):
        """reactions: frame indexed by accession with the target column and the last day of its window
        ('window_end'); supplied by run_events.py after the pre-registration."""
        self.reactions = reactions
        self.target = target

    def params(self):
        return {"max_features": MAX_FEATURES, "min_df": MIN_DF, "sublinear_tf": True, "stop_words": "english",
                "alpha": ALPHA, "min_train": MIN_TRAIN, "max_chars": MAX_CHARS, "target": self.target,
                "refit": "every calendar quarter, on releases whose window closed before it"}

    def score(self, releases):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        if self.reactions is None:
            raise ValueError("tfidf needs the reactions of earlier releases")
        data = releases.join(self.reactions[[self.target, "window_end"]], on="accession")
        data["quarter"] = pd.PeriodIndex(data.accepted_et.str[:10], freq="Q")
        text_of = {}
        for q, group in data.groupby("quarter"):
            start = q.start_time.strftime("%Y-%m-%d")
            train = data[(data.window_end < start) & data[self.target].notna()]
            if len(train) < MIN_TRAIN:
                for r in group.itertuples():
                    yield Result(r.accession, {"predicted_reaction": None, "n_train": len(train)})
                continue
            for a, p in zip(train.accession, train.text_path):
                if a not in text_of:
                    text_of[a] = texts.read(p, MAX_CHARS)
            vec = TfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF, sublinear_tf=True, stop_words="english")
            X = vec.fit_transform([text_of[a] for a in train.accession])
            model = Ridge(alpha=ALPHA).fit(X, train[self.target].to_numpy())
            preds = model.predict(vec.transform([texts.read(p, MAX_CHARS) for p in group.text_path]))
            for r, pred in zip(group.itertuples(), preds):
                yield Result(r.accession, {"predicted_reaction": float(pred), "n_train": len(train)})
