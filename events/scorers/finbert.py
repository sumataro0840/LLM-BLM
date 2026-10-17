"""
FinBERT of Huang, Wang and Yang (2023): sentence-level tone (positive / negative / neutral), averaged over the
first sentences of the release. Runs on this PC's GPU.
"""
import re
import time

from events import config, texts
from events.scorers import Result, Scorer

MAX_SENTENCES = 200          # the opening of a release carries the news; tables at the end are mostly numbers
MIN_CHARS, MAX_CHARS = 20, 1000


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if MIN_CHARS <= len(p.strip()) <= MAX_CHARS and re.search(r"[a-z]", p)][:MAX_SENTENCES]


class FinBERTScorer(Scorer):
    method = "finbert"
    uses_outcomes = False

    def __init__(self, batch_size=32):
        from huggingface_hub import model_info
        self.model = config.FINBERT_MODEL
        self.revision = model_info(self.model).sha
        self.batch_size = batch_size

    def params(self):
        return {"model": self.model, "revision": self.revision, "max_sentences": MAX_SENTENCES,
                "sentence_chars": [MIN_CHARS, MAX_CHARS], "aggregate": "mean of probabilities over sentences"}

    def score(self, releases):
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer
        # the config of finbert-tone has no model_type, so the BERT classes are named explicitly
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BertTokenizer.from_pretrained(self.model, revision=self.revision)
        model = BertForSequenceClassification.from_pretrained(self.model, revision=self.revision).to(device).eval()
        labels = {i: name.lower() for i, name in model.config.id2label.items()}   # 0 neutral, 1 positive, 2 negative
        for r in releases.itertuples():
            started = time.time()
            sents = sentences(texts.read(r.text_path))
            if not sents:
                yield Result(r.accession, {"positive": None, "negative": None, "net_tone": None, "n_sentences": 0})
                continue
            chunks = []
            with torch.no_grad():
                for i in range(0, len(sents), self.batch_size):
                    batch = tokenizer(sents[i:i + self.batch_size], padding=True, truncation=True, max_length=512,
                                      return_tensors="pt").to(device)
                    chunks.append(torch.softmax(model(**batch).logits.float(), -1).cpu())
            p = torch.cat(chunks).mean(0)
            pos = p[[k for k, v in labels.items() if v == "positive"][0]].item()
            neg = p[[k for k, v in labels.items() if v == "negative"][0]].item()
            yield Result(r.accession, {"positive": pos, "negative": neg, "net_tone": pos - neg, "n_sentences": len(sents)},
                         {"model_version": self.revision, "seconds": round(time.time() - started, 2), "cost_usd": 0.0})
