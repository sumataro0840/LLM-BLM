"""
ChronoGPT-Instruct (He, Lv, Manela, Wu 2025) reads the opening of the release with zero-shot questions.

Each release is read by the vintage trained on text up to the end of the year before its acceptance
(chrono.vintage_for); releases after 2025 are read by the newest published vintage (2024). The model only sees 1,792 tokens, so only the
opening of the release is read. Answers are the probabilities of fixed choices, calibrated against content-free
inputs (see 工房/LLM/chrono/README.md; without it the model answers 'lower' to almost everything).
"""
import sys
import time

from events import config, texts
from events.scorers import Result, Scorer

QUESTIONS = {
    "guidance": ("Does the company raise, lower, or keep its guidance in the following earnings press release excerpt? "
                 "Answer with one word: raise, lower, or keep.", ["raise", "lower", "keep"]),
    "tone": ("Classify management's tone in the following earnings press release excerpt.",
             ["very negative", "negative", "neutral", "positive", "very positive"]),
}
MAX_CHARS = 6000             # far more than fits; chrono.fit cuts the text to the context
LATEST_VINTAGE = 2024        # newest vintage published (2026-10); later releases are read by it


def _chrono():
    if str(config.CHRONO_DIR) not in sys.path:
        sys.path.insert(0, str(config.CHRONO_DIR))
    import chrono
    return chrono


class ChronoScorer(Scorer):
    method = "chrono"
    uses_outcomes = False
    model = "manelalab/chrono-gpt-instruct-v1"

    def __init__(self):
        self.chrono = _chrono()
        self.available = self.chrono.available()
        if not self.available:
            raise SystemExit(f"ChronoGPT の版がありません ({config.CHRONO_DIR}/models)")

    def params(self):
        return {"model": self.model, "questions": QUESTIONS, "max_chars": MAX_CHARS, "calibration": self.chrono.CONTENT_FREE,
                "vintage": f"end of the year before the acceptance date, at most {LATEST_VINTAGE}",
                "scoring_code_sha256": self.scoring_code_hash()}

    def scoring_code_hash(self):
        """Hash of the functions that decide the scores (prompt, cut, choice likelihoods, model pass, calibration),
        so that changes elsewhere in chrono.py (e.g. memory handling) do not start a new run."""
        import hashlib
        import inspect
        parts = [inspect.getsource(getattr(self.chrono, name)) for name in
                 ("format_prompt", "fit", "score_choices", "_choice_probs", "final_hidden", "head")]
        parts += [repr(self.chrono.SYSTEM), repr(self.chrono.CONTEXT), repr(self.chrono.CONTENT_FREE)]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def vintage(self, accepted_et):
        """The vintage depends on the date only (never on what happens to be downloaded), so a run is reproducible."""
        year = min(self.chrono.vintage_for(accepted_et[:10]), LATEST_VINTAGE)
        if year not in self.available:
            raise SystemExit(f"ChronoGPT の {year} 年末の版がありません: cd {config.CHRONO_DIR} && python chrono.py download {year}")
        return year

    def score(self, releases):
        releases = releases.assign(vintage=[self.vintage(t) for t in releases.accepted_et])
        model = None
        for year, group in releases.groupby("vintage"):
            model = None                     # drop the previous vintage before the next one is put on the GPU
            model, tok = self.chrono.load(year)
            for r in group.itertuples():
                started = time.time()
                text = texts.read(r.text_path, MAX_CHARS)
                values, truncated = {}, False
                for name, (question, choices) in QUESTIONS.items():
                    out = self.chrono.score_choices(model, tok, question, choices, text=text)
                    truncated |= out["truncated"]
                    for c, p in out["probs"].items():
                        values[f"{name}_{c.replace(' ', '_')}"] = p
                values["guidance_net"] = values["guidance_raise"] - values["guidance_lower"]
                values["tone_net"] = (values["tone_positive"] + values["tone_very_positive"]
                                      - values["tone_negative"] - values["tone_very_negative"])
                yield Result(r.accession, values, {"model_version": str(year), "seconds": round(time.time() - started, 2),
                                                   "cost_usd": 0.0, "raw_answer": f"truncated={truncated}"})
