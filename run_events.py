"""
Earnings releases as events (design: 設計.md).

    python run_events.py import-h06                          # H06's releases and gpt-4o-mini scores -> DB (once)
    SEC_USER_AGENT="<name> <e-mail>" python run_events.py fetch --from 2015-01-01 --to 2023-10-31
    python run_events.py timing                              # acceptance time -> session and event date (again after a price update)
    python run_events.py surprise                            # earnings and revenue surprises
    python run_events.py score dictionary [--from ... --to ... --limit N]   # llm / chrono / dictionary / finbert / tfidf
    python run_events.py status

Scorers trained on stock reactions (tfidf) need --i-have-preregistered: they look at outcomes, which is allowed only
after experiments/E2_event_reaction/preregistration.md has been committed (設計.md, 9.2).
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from events import config, edgar, surprise, texts
from events.import_h06 import run as import_h06, symbol_chooser, trading_days
from events.scorers import make
from events.timing import timing
from marketdb.db import events as db
from marketdb.db.connection import connect, init_db


def cmd_fetch(conn, args):
    tickers = pd.read_sql("SELECT t.symbol, t.cik FROM tickers t WHERE t.cik IS NOT NULL AND EXISTS "
                          "(SELECT 1 FROM index_membership m WHERE m.symbol = t.symbol)", conn)
    if args.symbols:
        tickers = tickers[tickers.symbol.isin(args.symbols.split(","))]
    choose = symbol_chooser(conn)
    symbols_of = tickers.groupby("cik").symbol.apply(sorted).to_dict()
    known = {r[0] for r in conn.execute("SELECT accession FROM earnings_releases WHERE text_path IS NOT NULL")}
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:
        lists = list(pool.map(lambda c: edgar.filings_of(c, args.start, args.end), sorted(symbols_of)))
    filings = [f for lst in lists for f in lst if f["accession"] not in known]
    print(f"{len(symbols_of)} companies, {len(filings)} new filings ({time.time() - t0:.0f}s)", flush=True)
    days = trading_days(conn)
    batch = []
    with ThreadPoolExecutor(4) as pool:
        for i, row in enumerate(pool.map(edgar.fetch_text, filings), 1):
            accepted_at, accepted_et, session, event_date = timing(row["accepted"], days)
            sha = n_chars = kind = None
            if row.get("text_path"):
                body = texts.read(row["text_path"])
                sha, n_chars, kind = texts.sha256_of(texts.absolute(row["text_path"])), len(body), texts.doc_type(body)
            batch.append({"accession": row["accession"], "symbol": choose(symbols_of[row["cik"]], row["filing_date"]),
                          "cik": row["cik"], "filing_date": row["filing_date"], "accepted_at": accepted_at,
                          "accepted_et": accepted_et, "session": session, "event_date": event_date,
                          "document": row.get("document"), "doc_type": kind, "how_chosen": row["how_chosen"],
                          "text_path": row.get("text_path"), "text_sha256": sha, "n_chars": n_chars})
            if len(batch) >= 200 or i == len(filings):
                db.save_releases(conn, pd.DataFrame(batch))
                batch = []
                print(f"  {i}/{len(filings)} ({time.time() - t0:.0f}s)", flush=True)


def cmd_timing(conn, args):
    days = trading_days(conn)
    rel = db.load_releases(conn)
    rows = []
    for r in rel.itertuples():
        _, accepted_et, session, event_date = timing(r.accepted_at, days)
        rows.append((accepted_et, session, event_date, r.accession))
    db.update_timing(conn, rows)
    print(f"timing of {len(rows)} releases; without an event date: {sum(r[2] is None for r in rows)}")


def cmd_surprise(conn, args):
    rel = db.load_releases(conn, args.start, args.end)
    out = surprise.compute(conn, rel)
    db.save_surprises(conn, out)
    print(f"surprises of {len(out)} releases; sue {out.sue.notna().sum()}, revenue {out.revenue_surprise.notna().sum()}")


def cmd_score(conn, args):
    scorer = make(args.method)
    if scorer.uses_outcomes:
        if not args.i_have_preregistered:
            raise SystemExit(f"{args.method} は株価の反応で学習します。事前登録をコミットしてから --i-have-preregistered を付けて実行する")
        raise SystemExit("反応を使う scorer の実行は、事前登録の後に run_events.py に足す (設計.md 9.2)")
    run_id = db.get_or_create_score_run(conn, scorer.method, scorer.model, scorer.params(), scorer.uses_outcomes, args.note)
    start = args.start
    if args.method == "llm":
        start = max(start or config.LLM_FIRST_DATE, config.LLM_FIRST_DATE)
    rel = db.load_releases(conn, start, args.end, with_text=True)
    done = db.scored_accessions(conn, run_id)
    todo = rel[~rel.accession.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"score run {run_id} ({args.method}): {len(rel)} releases, {len(todo)} to score", flush=True)
    t0 = time.time()
    for i, result in enumerate(scorer.score(todo), 1):
        db.save_scores(conn, run_id, result.accession, result.values, result.call)
        if i % 100 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


def cmd_status(conn, args):
    s = db.status(conn)
    n, with_text, no_event, first, last = s["releases"]
    print(f"releases: {n} (text {with_text}, no event date {no_event}), filed {first} .. {last}")
    print(f"sessions: {s['sessions']}")
    print(f"surprises: {s['surprises']}")
    for run in s["runs"]:
        print(f"score run {run[0]}: {run[1]} / {run[2]} / uses_outcomes={run[3]}: {run[4]} releases, cost {run[5]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="earnings releases as events")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("import-h06")
    f = sub.add_parser("fetch")
    f.add_argument("--from", dest="start", required=True)
    f.add_argument("--to", dest="end", required=True)
    f.add_argument("--symbols", help="comma separated (default: every S&P500 member with a CIK)")
    sub.add_parser("timing")
    su = sub.add_parser("surprise")
    sc = sub.add_parser("score")
    sc.add_argument("method", choices=["llm", "chrono", "dictionary", "finbert", "tfidf"])
    sc.add_argument("--limit", type=int)
    sc.add_argument("--note")
    sc.add_argument("--i-have-preregistered", action="store_true")
    for p in (su, sc):
        p.add_argument("--from", dest="start")
        p.add_argument("--to", dest="end")
    sub.add_parser("status")
    args = parser.parse_args()

    conn = connect(config.DB_PATH)
    # several commands may run at once (e.g. scoring on the GPU while surprises are recomputed): wait for the
    # other writer instead of failing after SQLite's default 5 seconds
    conn.execute("PRAGMA busy_timeout = 60000")
    init_db(conn)
    {"import-h06": lambda c, a: import_h06(c), "fetch": cmd_fetch, "timing": cmd_timing, "surprise": cmd_surprise,
     "score": cmd_score, "status": cmd_status}[args.command](conn, args)


if __name__ == "__main__":
    main()
