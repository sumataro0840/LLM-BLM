"""
The event tables: releases are updated in place, a score run is found again by its settings,
scores keep 'no answer' apart from 'answered nothing'; the llm scorer asks H06's question word for word.
"""
import sqlite3
import unittest
from pathlib import Path

import pandas as pd

from marketdb.db import events as db
from marketdb.db.connection import init_db

REPO = Path(__file__).resolve().parents[1]


def release(accession="0000000001-24-000001", event_date="2024-01-12"):
    return {"accession": accession, "symbol": "AAA", "cik": 1, "filing_date": "2024-01-11",
            "accepted_at": "2024-01-11T21:05:00+00:00", "accepted_et": "2024-01-11T16:05:00", "session": "post",
            "event_date": event_date, "document": "ex991.htm", "doc_type": "EX-99.1", "how_chosen": "named",
            "text_path": None, "text_sha256": None, "n_chars": None}


class TestEventTables(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA foreign_keys = ON")
        init_db(self.conn)
        self.conn.execute("INSERT INTO tickers (symbol, security, fetched_at) VALUES ('AAA', 'A Corp', 'x')")

    def test_release_is_updated_in_place(self):
        db.save_releases(self.conn, pd.DataFrame([release(event_date=None)]))
        db.save_releases(self.conn, pd.DataFrame([release(event_date="2024-01-12")]))
        rows = self.conn.execute("SELECT accession, event_date FROM earnings_releases").fetchall()
        self.assertEqual(rows, [("0000000001-24-000001", "2024-01-12")])

    def test_score_run_found_by_its_settings(self):
        a = db.get_or_create_score_run(self.conn, "dictionary", "LM", {"x": 1, "y": [1, 2]}, False)
        b = db.get_or_create_score_run(self.conn, "dictionary", "LM", {"y": [1, 2], "x": 1}, False)
        c = db.get_or_create_score_run(self.conn, "dictionary", "LM", {"x": 2}, False)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        with self.assertRaises(ValueError):
            db.get_or_create_score_run(self.conn, "finbert", "LM", {"x": 1, "y": [1, 2]}, False)

    def test_scores_and_calls(self):
        db.save_releases(self.conn, pd.DataFrame([release()]))
        run = db.get_or_create_score_run(self.conn, "llm", "gpt-4o-mini", {"q": "?"}, False)
        db.save_scores(self.conn, run, "0000000001-24-000001",
                       {"tone": 1, "guidance_change": None, "risk_flags": ["supply_chain"]},
                       {"model_version": "v", "cost_usd": 0.001, "prompt_tokens": 10})
        rows = dict((r[0], r[1:]) for r in self.conn.execute(
            "SELECT item, value_num, value_text FROM release_scores ORDER BY item"))
        self.assertEqual(rows["tone"], (1.0, None))
        self.assertEqual(rows["guidance_change"], (None, None))
        self.assertEqual(rows["risk_flags"], (None, '["supply_chain"]'))
        self.assertEqual(db.scored_accessions(self.conn, run), {"0000000001-24-000001"})
        self.assertEqual(self.conn.execute("SELECT cost_usd FROM release_score_calls").fetchone()[0], 0.001)
        self.assertEqual(db.load_scores(self.conn, run)["tone"].iloc[0], 1.0)

    def test_score_needs_a_known_release(self):
        run = db.get_or_create_score_run(self.conn, "llm", "m", {}, False)
        with self.assertRaises(sqlite3.IntegrityError):
            db.save_scores(self.conn, run, "unknown", {"tone": 1})

    def test_llm_question_is_h06_word_for_word(self):
        h06 = (REPO / "experiments" / "H06_earnings_text" / "score.py").read_text(encoding="utf-8").splitlines()
        start = h06.index("MAX_CHARS = 15000")
        end = next(i for i, line in enumerate(h06) if line.startswith("PARAMS = "))
        block = "\n".join(h06[start:end + 1])
        self.assertIn(block, (REPO / "events" / "scorers" / "llm.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
