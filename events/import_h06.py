"""
Move the releases and the gpt-4o-mini scores of H06 into the database (once; 設計.md, 10).

    experiments/H06_earnings_text/results/filings.csv  -> earnings_releases
    experiments/H06_earnings_text/results/scores.jsonl -> release_score_runs / release_scores / release_score_calls

Every logged request is checked to have asked exactly the question of events/scorers/llm.py, so the imported run
and new scores of the same scorer share one run (same params hash).
"""
import json
from datetime import datetime, timezone

import pandas as pd

from events import config, texts
from events.scorers.llm import ITEMS, PARAMS, QUESTION, SYSTEM, LLMScorer
from events.timing import timing
from marketdb.db import events as db

H06 = config.REPO / "experiments" / "H06_earnings_text" / "results"


def trading_days(conn):
    return [r[0] for r in conn.execute("SELECT date FROM trading_days ORDER BY date")]


def symbol_chooser(conn):
    """Dual-class companies share one CIK (GOOG/GOOGL, FOX/FOXA, NWS/NWSA). A release is filed to the symbol
    that was in the S&P500 on the filing date, the first in alphabetical order when both were."""
    members = pd.read_sql("SELECT symbol, start_date, end_date FROM index_membership", conn)

    def choose(candidates, date):
        m = members[members.symbol.isin(candidates) & (members.start_date.isna() | (members.start_date <= date))
                    & (members.end_date.isna() | (members.end_date > date))]
        return sorted(m.symbol)[0] if len(m) else sorted(candidates)[0]
    return choose


def releases_from_filings(conn, filings, days):
    choose = symbol_chooser(conn)
    rows = []
    for f in filings.itertuples():
        symbol = choose(f.symbols.split(","), f.filing_date)
        accepted_at, accepted_et, session, event_date = timing(f.accepted, days)
        text_path = None if pd.isna(f.file) else f.file.replace("\\", "/")
        sha, n_chars, kind = None, None, None
        if text_path and texts.absolute(text_path).exists():
            body = texts.read(text_path)
            sha, n_chars, kind = texts.sha256_of(texts.absolute(text_path)), len(body), texts.doc_type(body)
        elif text_path:
            text_path = None
        rows.append({"accession": f.accession, "symbol": symbol, "cik": int(f.cik), "filing_date": f.filing_date,
                     "accepted_at": accepted_at, "accepted_et": accepted_et, "session": session, "event_date": event_date,
                     "document": None if pd.isna(f.document) else f.document, "doc_type": kind, "how_chosen": f.how,
                     "text_path": text_path, "text_sha256": sha, "n_chars": n_chars})
    return pd.DataFrame(rows)


def check_request(rec):
    """The logged request must be the question of the llm scorer, word for word."""
    system, user = rec["messages"][0]["content"], rec["messages"][1]["content"]
    if system != SYSTEM or not user.startswith(QUESTION) or rec["params"] != json.loads(json.dumps(PARAMS)) \
            or rec["requested_model"] != config.OPENAI_MODEL:
        raise ValueError(f"{rec['accession']}: the logged request differs from events/scorers/llm.py")


def to_utc(local):
    """'2026-10-16T22:11:45+0900' -> '2026-10-16T13:11:45+00:00'."""
    return datetime.strptime(local, "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc).isoformat(timespec="seconds")


def run(conn):
    filings = pd.read_csv(H06 / "filings.csv", dtype={"cik": int})
    known = {r[0] for r in conn.execute("SELECT symbol FROM tickers")}
    missing = {s for joined in filings.symbols for s in joined.split(",")} - known
    if missing:
        raise ValueError(f"symbols not in tickers: {sorted(missing)[:5]}")
    releases = releases_from_filings(conn, filings, trading_days(conn))
    db.save_releases(conn, releases)
    print(f"releases: {len(releases)} (text {releases.text_path.notna().sum()})", flush=True)

    records = [json.loads(line) for line in open(H06 / "scores.jsonl", encoding="utf-8") if line.strip()]
    for rec in records:
        check_request(rec)
    scorer = LLMScorer()
    run_id = db.get_or_create_score_run(conn, scorer.method, scorer.model, scorer.params(), scorer.uses_outcomes,
                                        note="imported from experiments/H06_earnings_text/results/scores.jsonl")
    done = db.scored_accessions(conn, run_id)
    n = 0
    for rec in records:
        if rec["accession"] in done:
            continue
        answer = rec["answers"][0]
        values = json.loads(answer)
        db.save_scores(conn, run_id, rec["accession"], {k: values.get(k) for k in ITEMS}, {
            "model_version": rec["model"], "raw_answer": answer, "prompt_tokens": rec["prompt_tokens"],
            "completion_tokens": rec["completion_tokens"], "cost_usd": rec["cost_usd"], "seconds": rec["seconds"],
            "answered_at": to_utc(rec["at"])})
        n += 1
    total = conn.execute("SELECT COUNT(*), ROUND(SUM(cost_usd), 4) FROM release_score_calls WHERE score_run_id = ?",
                         (run_id,)).fetchone()
    print(f"score run {run_id}: {n} new, {total[0]} in all, cost {total[1]} USD (log: {len(records)} requests, "
          f"{round(sum(r['cost_usd'] for r in records), 4)} USD)")
    return run_id
